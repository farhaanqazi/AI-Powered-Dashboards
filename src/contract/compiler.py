"""Phase 2 — contract compiler.

``compile_contract`` turns Layer-1/2 profiles + the cleaned frame into an
immutable :class:`DatasetContract`: a schema fingerprint, the row grain,
aggregate-row detection, year/ratio refinement, and per-field aggregation /
chart allow-lists. It computes only deterministic facts — no LLM.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src import config
from src.analysis.data_structures import EnrichedProfile
from src.contract.models import DatasetContract, FieldContract
from src.contract.models import IngestResult

_AGG_ROW_RE = re.compile(
    r"^\s*(grand\s+total|sub\s*total|subtotal|total|totals|all|sum|average|avg|mean)\s*$",
    re.IGNORECASE,
)
_YEAR_NAME_RE = re.compile(r"(?i)(?<![a-z])(year|yr|fy|fiscal_year)(?![a-z])")
_RATIO_NAME_RE = re.compile(r"(?i)(rate|ratio|pct|percent|percentage|share)")

# role/domain -> allowed deterministic aggregations and chart kinds.
_AGG_ALLOW = {
    "identifier": ("count", "nunique"),
    "year": ("min", "max", "range", "nunique"),
    "datetime": ("min", "max", "range"),
    "ratio": ("mean", "median", "min", "max"),
    "numeric_additive": ("sum", "mean", "min", "max", "median"),
    "numeric": ("mean", "median", "min", "max"),
    "categorical": ("count", "nunique"),
    "boolean": ("count", "sum", "mean"),
    "text": ("count", "nunique"),
}
_CHART_ALLOW = {
    "identifier": (),
    "year": ("line", "bar"),
    "datetime": ("line", "area"),
    "ratio": ("line", "bar", "box"),
    "numeric_additive": ("bar", "line", "histogram", "box", "scatter"),
    "numeric": ("histogram", "box", "scatter", "line"),
    "categorical": ("bar", "pie"),
    "boolean": ("bar", "pie"),
    "text": (),
}


def schema_fingerprint(df: pd.DataFrame, profiles: Dict[str, EnrichedProfile]) -> str:
    """Stable hash over (column, dtype, role) — order-independent."""
    items = sorted(
        (str(c), str(df[c].dtype), profiles.get(c).role if profiles.get(c) else "")
        for c in df.columns
    )
    return hashlib.sha256(
        json.dumps(items, separators=(",", ":")).encode()
    ).hexdigest()


def _classify_year_ratio(
    name: str, series: pd.Series, prof: EnrichedProfile
) -> Tuple[bool, bool]:
    is_year = is_ratio = False
    tags = set(prof.semantic_tags)
    if prof.role == "numeric":
        s = pd.to_numeric(series, errors="coerce").dropna()
        if not s.empty:
            int_like = bool((s == s.round()).all())
            in_year_range = bool(
                s.min() >= config.MIN_DATE and s.max() <= config.MAX_DATE
            )
            if int_like and in_year_range and (
                _YEAR_NAME_RE.search(name) or s.nunique() <= (config.MAX_DATE - config.MIN_DATE)
            ) and _YEAR_NAME_RE.search(name):
                is_year = True
            bounded = bool(s.min() >= 0 and s.max() <= 1) or bool(
                s.min() >= 0 and s.max() <= 100 and _RATIO_NAME_RE.search(name)
            )
            if not is_year and ("rate" in tags or bounded or _RATIO_NAME_RE.search(name)):
                is_ratio = True
    return is_year, is_ratio


def _field_contract(
    name: str, df: pd.DataFrame, prof: EnrichedProfile, ingest: Optional[IngestResult]
) -> FieldContract:
    tags = set(prof.semantic_tags)
    is_year, is_ratio = _classify_year_ratio(name, df[name], prof)

    if prof.role == "numeric" and "additive" in tags:
        aggregation = "additive"
    elif is_ratio or "rate" in tags:
        aggregation = "rate"
    elif prof.role == "boolean":
        aggregation = "additive"
    else:
        aggregation = "none"

    if is_year:
        allow_key, role_out, domain = "year", "year", "year"
    elif is_ratio:
        allow_key, role_out, domain = "ratio", "ratio", "ratio"
    elif prof.role == "numeric" and aggregation == "additive":
        allow_key = "numeric_additive"
        role_out = "numeric"
        domain = "monetary" if "monetary" in tags else "measure"
    else:
        allow_key = prof.role if prof.role in _AGG_ALLOW else "text"
        role_out = prof.role
        domain = "dimension" if prof.role in ("categorical", "identifier") else "generic"

    pii_entities: Tuple[str, ...] = ()
    sensitivity = "public"
    if ingest is not None:
        ents = ingest.pii_columns.get(name)
        if ents:
            pii_entities = tuple(ents)
            sensitivity = "sensitive"
        elif ingest.sensitivity == "sensitive":
            sensitivity = "sensitive"

    return FieldContract(
        name=name,
        dtype=str(df[name].dtype),
        role=role_out,
        domain=domain,
        confidence=float(prof.confidence),
        alternatives=tuple(
            {a["role"]: float(a["confidence"])} for a in (prof.alternatives or [])
        ),
        aggregation=aggregation,
        is_identifier=(prof.role == "identifier"),
        is_year=is_year,
        is_ratio=is_ratio,
        sensitivity=sensitivity,
        pii_entities=pii_entities,
        allowed_aggregations=_AGG_ALLOW.get(allow_key, ("count",)),
        allowed_charts=_CHART_ALLOW.get(allow_key, ()),
    )


def _detect_grain(
    df: pd.DataFrame, profiles: Dict[str, EnrichedProfile], max_combo: int = 2
) -> Tuple[str, ...]:
    """Smallest set of dimension columns whose tuple is unique per row."""
    if df.empty:
        return ()
    n = len(df)
    dims = [
        c
        for c in df.columns
        if profiles.get(c)
        and profiles[c].role in ("identifier", "categorical", "datetime", "year")
    ][:8]  # bound combinatorics
    for size in range(1, max_combo + 1):
        for combo in itertools.combinations(dims, size):
            try:
                if not df.duplicated(subset=list(combo)).any() and df[list(combo)].notna().all(axis=None):
                    return tuple(combo)
            except Exception:
                continue
    return ()


def _count_aggregate_rows(df: pd.DataFrame, profiles: Dict[str, EnrichedProfile]) -> int:
    dim_cols = [
        c
        for c in df.columns
        if profiles.get(c) and profiles[c].role in ("categorical", "text", "identifier")
    ]
    if not dim_cols or df.empty:
        return 0
    mask = pd.Series(False, index=df.index)
    for c in dim_cols:
        col = df[c].astype(str)
        mask = mask | col.str.match(_AGG_ROW_RE)
    return int(mask.sum())


def compile_contract(
    df: pd.DataFrame,
    profiles: Dict[str, EnrichedProfile],
    ingest: Optional[IngestResult] = None,
) -> DatasetContract:
    """Compile an immutable :class:`DatasetContract` from profiles + frame."""
    fp = schema_fingerprint(df, profiles)
    fields = {
        name: _field_contract(name, df, prof, ingest)
        for name, prof in profiles.items()
        if name in df.columns
    }
    grain = _detect_grain(df, profiles)
    agg_rows = _count_aggregate_rows(df, profiles)

    sensitivity = "public"
    pii_blocked = False
    if ingest is not None:
        sensitivity = ingest.sensitivity
        pii_blocked = ingest.pii_blocked
    elif any(f.sensitivity == "sensitive" for f in fields.values()):
        sensitivity = "sensitive"

    return DatasetContract(
        schema_fingerprint=fp,
        version=1,
        locked=False,
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        grain=grain,
        has_aggregate_rows=agg_rows > 0,
        aggregate_row_count=agg_rows,
        sensitivity=sensitivity,
        pii_blocked=pii_blocked,
        fields=fields,
    )
