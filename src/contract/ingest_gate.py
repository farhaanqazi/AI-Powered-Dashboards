"""Phase 1 — the Ingest Contract Gate.

``run_ingest_gate`` is the single doorway every dataset passes through before
any profiling, classification or LLM narration. It:

1. coerces thousands/currency/percent-formatted text columns to real numbers,
2. replaces sentinel tokens (``"NA"``, ``"-"``, ``"#N/A"`` …) with ``pd.NA``,
3. drops (near-)entirely-null rows and rejects datasets left empty,
4. scans for PII (Presidio when installed, regex fallback otherwise) and sets
   a fail-closed ``sensitivity`` / ``pii_blocked`` verdict.

Every mutation is recorded in a :class:`CleaningManifest`; nothing is cleaned
silently. Wiring into the pipeline is Phase 5 — this module is self-contained.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List

import pandas as pd

from src import config
from src.contract.models import CleaningManifest, IngestResult

logger = logging.getLogger(__name__)

# Strip everything that is not a digit, sign, decimal point or exponent so
# "$1,234.50", "1 234,50 €" (after the comma swap below) and "(1,200)" parse.
_CURRENCY_GROUPING_RE = re.compile(r"[^\d.\-eE]")
_PARENS_NEGATIVE_RE = re.compile(r"^\((.*)\)$")

# Regex PII fallback (used when Presidio is not installed).
_REGEX_PII = {
    "EMAIL_ADDRESS": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "PHONE_NUMBER": re.compile(
        r"(?<!\d)(?:\+?\d{1,3}[ .\-]?)?(?:\(\d{2,4}\)[ .\-]?)?\d{3}[ .\-]?\d{3,4}(?!\d)"
    ),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def _looks_numeric_after_strip(series: pd.Series) -> tuple[bool, pd.Series]:
    """Return (is_coercible, coerced_series) for a candidate text column."""
    s = series.dropna().astype(str).str.strip()
    if s.empty:
        return False, series
    # "(1,200)" -> "-1,200"
    s_signed = s.str.replace(_PARENS_NEGATIVE_RE, r"-\1", regex=True)
    stripped = s_signed.str.replace(_CURRENCY_GROUPING_RE, "", regex=True)
    coerced = pd.to_numeric(stripped, errors="coerce")
    frac_ok = coerced.notna().mean()
    if frac_ok >= config.INGEST_NUMERIC_COERCE_FRACTION:
        full = series.astype(str).str.strip()
        full = full.str.replace(_PARENS_NEGATIVE_RE, r"-\1", regex=True)
        full = full.str.replace(_CURRENCY_GROUPING_RE, "", regex=True)
        return True, pd.to_numeric(full, errors="coerce")
    return False, series


def _coerce_numeric(df: pd.DataFrame, manifest: CleaningManifest) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype != object:
            continue
        sample = df[col].dropna()
        if sample.empty:
            continue
        # Skip columns that are clearly free text (no digits at all).
        if not sample.astype(str).str.contains(r"\d", regex=True).any():
            continue
        ok, coerced = _looks_numeric_after_strip(df[col])
        if ok:
            df[col] = coerced
            manifest.coerced_numeric[col] = (
                "removed currency/grouping symbols and converted to numbers"
            )
    return df


def _null_sentinels(df: pd.DataFrame, manifest: CleaningManifest) -> pd.DataFrame:
    sentinels = {s.lower() for s in config.INGEST_SENTINELS}
    for col in df.columns:
        if df[col].dtype != object:
            continue
        as_str = df[col].astype(str).str.strip().str.lower()
        mask = as_str.isin(sentinels) & df[col].notna()
        n = int(mask.sum())
        if n:
            df.loc[mask, col] = pd.NA
            manifest.sentinels_nulled[col] = n
    return df


def _drop_null_rows(df: pd.DataFrame, manifest: CleaningManifest) -> pd.DataFrame:
    if df.empty:
        return df
    na_frac = df.isna().mean(axis=1)
    keep = na_frac < config.INGEST_NULL_ROW_FRACTION
    dropped = int((~keep).sum())
    if dropped:
        manifest.dropped_null_rows = dropped
        df = df.loc[keep].reset_index(drop=True)
    return df


def _detect_pii(df: pd.DataFrame) -> tuple[Dict[str, List[str]], str]:
    """Return (column -> entity types, engine name)."""
    sample_rows = config.PII_SAMPLE_ROWS
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if not obj_cols:
        return {}, "none"

    samples = {
        c: df[c].dropna().astype(str).head(sample_rows).tolist() for c in obj_cols
    }
    samples = {c: v for c, v in samples.items() if v}
    if not samples:
        return {}, "none"

    # Preferred: Presidio (optional dependency).
    try:
        from presidio_analyzer import AnalyzerEngine

        analyzer = AnalyzerEngine()
        wanted = set(config.PII_ENTITY_TYPES)
        found: Dict[str, set] = {}
        for col, values in samples.items():
            text = "\n".join(values)
            results = analyzer.analyze(text=text, language="en")
            hits = {
                r.entity_type
                for r in results
                if r.score >= config.PII_SCORE_THRESHOLD
                and r.entity_type in wanted
            }
            if hits:
                found[col] = hits
        return {c: sorted(v) for c, v in found.items()}, "presidio"
    except Exception as exc:  # presidio missing or failed -> regex fallback
        logger.info("Presidio unavailable (%s); using regex PII fallback", exc)

    found = {}
    for col, values in samples.items():
        text = "\n".join(values)
        hits = {
            ent for ent, rx in _REGEX_PII.items() if rx.search(text)
        }
        if hits:
            found[col] = hits
    return {c: sorted(v) for c, v in found.items()}, "regex"


def run_ingest_gate(df: pd.DataFrame) -> IngestResult:
    """Clean, screen and classify a freshly-loaded DataFrame.

    Pure: it copies the input and never mutates the caller's frame.
    """
    manifest = CleaningManifest()
    warnings: List[str] = []

    if df is None or df.shape[1] == 0:
        return IngestResult(
            ok=False,
            rejected=True,
            reject_reason="Empty dataset (no columns).",
            sensitivity="sensitive" if config.SENSITIVITY_FAIL_CLOSED else "public",
        )

    work = df.copy()
    manifest.original_shape = (int(work.shape[0]), int(work.shape[1]))

    work = _coerce_numeric(work, manifest)
    work = _null_sentinels(work, manifest)
    work = _drop_null_rows(work, manifest)
    manifest.cleaned_shape = (int(work.shape[0]), int(work.shape[1]))

    if work.empty:
        return IngestResult(
            ok=False,
            rejected=True,
            reject_reason="Every row was empty or contained only placeholder values (like 'NA' or '-'), so there's nothing to analyze.",
            manifest=manifest,
            sensitivity="sensitive" if config.SENSITIVITY_FAIL_CLOSED else "public",
        )

    pii_columns, engine = _detect_pii(work)

    sensitive = bool(pii_columns)
    if not sensitive and config.SENSITIVITY_FAIL_CLOSED and engine == "regex":
        # Regex is best-effort; flag the dataset sensitive but do not block
        # egress unless actual PII was detected (invariant: pii_blocked only
        # on detected PII).
        warnings.append(
            "Used basic pattern detection for personal data (the advanced "
            "detector isn’t available here); treating this dataset as "
            "sensitive to be safe."
        )
        sensitive = True

    pii_blocked = bool(pii_columns) and config.PII_BLOCK_EGRESS

    return IngestResult(
        ok=True,
        rejected=False,
        df=work,
        manifest=manifest,
        sensitivity="sensitive" if sensitive else "public",
        pii_blocked=pii_blocked,
        pii_columns=pii_columns,
        pii_scan_engine=engine,
        warnings=warnings,
    )
