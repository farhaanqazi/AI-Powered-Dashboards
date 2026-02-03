"""
Layer 2: Semantic Classifier

Responsibilities:
-   Consumes the `SyntacticProfile` from Layer 1.
-   Applies heuristics and rules to infer the semantic role of each column.
-   Assigns semantic tags.
-   Produces an `EnrichedProfile`.
"""
import logging
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import re

from src.analysis.data_structures import SyntacticProfile, EnrichedProfile
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_bool_dtype
from src.utils.identifier_detector import is_likely_identifier_with_confidence

logger = logging.getLogger(__name__)

_CURRENCY_RE = re.compile(r"[$\\u00A3\\u20AC\\u00A5\\u20B9\\u20A9\\u20A6\\u20BD\\u0E3F\\u20AB\\u20B4\\u20B1\\u20AA\\u20A1]")
_AMBIGUOUS_DATE_RE = re.compile(r"\\b(\\d{1,2})[/-](\\d{1,2})[/-](\\d{2,4})\\b")
_DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
    "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
    "%m-%d-%Y", "%m/%d/%Y", "%m.%d.%Y",
]

def _clean_numeric_series(series: pd.Series) -> pd.Series:
    """
    Normalize numeric-like strings by stripping currency symbols, commas,
    whitespace, percent signs, and handling parentheses for negatives.
    """
    s = series.astype(str).str.strip()
    s = s.replace({"": None, "nan": None, "NaN": None, "None": None})
    s = s.str.replace(r"^\\((.*)\\)$", r"-\\1", regex=True)
    s = s.str.replace(_CURRENCY_RE, "", regex=True)
    s = s.str.replace("%", "", regex=False)
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace(r"\\s+", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def _numeric_conversion_rate(series: pd.Series) -> Tuple[pd.Series, float, bool]:
    """
    Attempt numeric conversion and return series, conversion rate, and currency flag.
    """
    raw = series.astype(str)
    has_currency = raw.str.contains(_CURRENCY_RE, regex=True, na=False).any()
    numeric = _clean_numeric_series(series)
    non_null = series.dropna()
    rate = 0.0
    if len(non_null) > 0:
        rate = numeric.notna().sum() / len(non_null)
    return numeric, rate, has_currency

def _compute_numeric_stats(series: pd.Series) -> Dict[str, Any]:
    """
    Compute numeric stats from a cleaned numeric series.
    """
    s_clean = series.dropna()
    if len(s_clean) == 0:
        return {}
    return {
        "count": int(len(s_clean)),
        "min": float(s_clean.min()),
        "max": float(s_clean.max()),
        "mean": float(s_clean.mean()),
        "std": float(s_clean.std()) if len(s_clean) > 1 else 0.0,
        "median": float(s_clean.median()),
        "q25": float(s_clean.quantile(0.25)),
        "q75": float(s_clean.quantile(0.75)),
        "sum": float(s_clean.sum()),
        "variance": float(s_clean.var()) if len(s_clean) > 1 else 0.0,
    }

def _is_ambiguous_date_strings(series: pd.Series, max_samples: int = 200, threshold: float = 0.2) -> bool:
    """
    Detect ambiguous day/month ordering (e.g., 01/02/2020 could be MM/DD or DD/MM).
    """
    sample = series.dropna().astype(str).head(max_samples)
    if sample.empty:
        return False
    ambiguous = 0
    total = 0
    for val in sample:
        m = _AMBIGUOUS_DATE_RE.search(val)
        if not m:
            continue
        total += 1
        try:
            a = int(m.group(1))
            b = int(m.group(2))
            if 1 <= a <= 12 and 1 <= b <= 12:
                ambiguous += 1
        except Exception:
            continue
    if total == 0:
        return False
    return (ambiguous / total) >= threshold

def _detect_datetime(series: pd.Series, min_rate: float = 0.85) -> Tuple[bool, Optional[str]]:
    """
    Try explicit formats first; only fall back to dateutil if confidence is high.
    Returns (is_datetime, reason_tag).
    """
    if _is_ambiguous_date_strings(series):
        return False, "ambiguous_date"

    s = series.dropna().astype(str)
    if s.empty:
        return False, None

    best_rate = 0.0
    best_fmt = None
    for fmt in _DATE_FORMATS:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        rate = parsed.notna().mean() if len(parsed) > 0 else 0.0
        if rate > best_rate:
            best_rate = rate
            best_fmt = fmt

    if best_rate >= min_rate:
        return True, f"format:{best_fmt}"

    parsed = pd.to_datetime(s, errors="coerce")
    rate = parsed.notna().mean() if len(parsed) > 0 else 0.0
    if rate >= 0.9 and not _is_ambiguous_date_strings(series, threshold=0.1):
        return True, "dateutil_high_confidence"

    return False, None

def run_semantic_classification(
    profiles: Dict[str, SyntacticProfile],
    df: pd.DataFrame
) -> Dict[str, EnrichedProfile]:
    """
    Performs Layer 2 analysis: assigns semantic roles to profiled columns.
    """
    enriched_profiles: Dict[str, EnrichedProfile] = {}

    for name, profile in profiles.items():
        role = "unknown"
        semantic_tags = []

        # --- Role Inference Hierarchy ---
        # The order of these checks is critical.

        # 1. Handle unambiguous data types first.
        if is_bool_dtype(df[name]):
            role = "boolean"
        elif is_datetime64_any_dtype(df[name]):
            role = "datetime"

        # 2. **CRITICAL FIX**: Check for identifiers *before* checking for generic numeric types.
        # This prevents numeric IDs from being misclassified as aggregatable measures.
        elif is_likely_identifier_with_confidence(df[name], name)[0]:
            role = "identifier"

        # 3. Check for generic numerics if it's not an identifier.
        elif is_numeric_dtype(df[name]):
            role = "numeric"

        # 4. For 'object' types, perform more detailed checks.
        elif profile.dtype == 'object':
            # Attempt to parse as datetime (explicit formats first)
            is_dt, dt_tag = _detect_datetime(df[name])
            if is_dt:
                role = "datetime"
                if dt_tag:
                    semantic_tags.append(dt_tag)
            elif dt_tag == "ambiguous_date":
                semantic_tags.append("ambiguous_date")
                role = "text"
            # Check for identifiers before numeric conversion
            elif is_likely_identifier_with_confidence(df[name], name)[0]:
                role = "identifier"
            else:
                # Try numeric normalization on object columns
                numeric_series, conv_rate, has_currency = _numeric_conversion_rate(df[name])
                if conv_rate >= 0.85:
                    role = "numeric"
                    if has_currency:
                        semantic_tags.append("monetary")
                    numeric_stats = _compute_numeric_stats(numeric_series)
                    if numeric_stats:
                        profile.stats.update(numeric_stats)
                # Check for low-cardinality strings
                elif profile.unique_count < 50 and profile.unique_count / profile.stats['count'] < 0.2: # Explicit count limit and lower ratio
                    role = "categorical"
                # Fallback for high-cardinality potential categorical (e.g., more than 50 unique values but still a small ratio)
                elif profile.unique_count / profile.stats['count'] < 0.05 and profile.unique_count > 50:
                    role = "categorical"
                # Otherwise, it's high-cardinality free text.
                else:
                    role = "text"

        # 5. Fallback for any other types.
        else:
            role = "text"

        # --- Semantic Tagging ---
        # Apply semantic tags only if the role is numeric, to avoid tagging text fields with currency symbols
        if role == 'numeric' and False:
            # Check for monetary symbols more robustly by looking at string representations
            sample_values = df[name].dropna().astype(str)
            if len(sample_values) > 0:
                sample_values = sample_values.sample(min(len(sample_values), 100), random_state=42)
            if any(re.search(r'[\$£€¥₹₩₦₽฿₫₴₱₪₡]', val) for val in sample_values):
                semantic_tags.append('monetary')

            # Ensure numeric stats exist for numeric-like object columns
            if profile.stats and "mean" not in profile.stats:
                numeric_series, _, _ = _numeric_conversion_rate(df[name])
                numeric_stats = _compute_numeric_stats(numeric_series)
                if numeric_stats:
                    profile.stats.update(numeric_stats)

        # Normalized numeric tagging (active path)
        if role == 'numeric':
            sample_values = df[name].dropna().astype(str)
            if len(sample_values) > 0:
                sample_values = sample_values.sample(min(len(sample_values), 100), random_state=42)
            if any(_CURRENCY_RE.search(val) for val in sample_values):
                semantic_tags.append('monetary')

            if profile.stats and "mean" not in profile.stats:
                numeric_series, _, _ = _numeric_conversion_rate(df[name])
                numeric_stats = _compute_numeric_stats(numeric_series)
                if numeric_stats:
                    profile.stats.update(numeric_stats)

        enriched_profiles[name] = EnrichedProfile(
            role=role,
            semantic_tags=semantic_tags,
            **profile.__dict__
        )

    logger.info(f"Layer 2: Semantic classification complete. Assigned roles to {len(enriched_profiles)} columns.")
    return enriched_profiles
