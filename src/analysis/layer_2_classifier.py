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

_CURRENCY_RE = re.compile("[$£€¥₹₩₦₽฿₫₴₱₪₡]")
_AMBIGUOUS_DATE_RE = re.compile(r"\\b(\\d{1,2})[/-](\\d{1,2})[/-](\\d{2,4})\\b")

# Numeric columns that aggregate by SUMMING across time:
# revenue, cost, sales, quantity, units, total_*, *_amount, fees, etc.
_ADDITIVE_NAME_RE = re.compile(
    r"(?i)(?<![a-zA-Z])("
    r"revenue|sales|cost|costs|expense|expenses|spend|spending|"
    r"amount|amounts|total|totals|subtotal|"
    r"qty|quantity|quantities|unit|units|count|counts|order|orders|"
    r"profit|loss|losses|income|earnings|"
    r"fee|fees|tax|taxes|discount|discounts|"
    r"volume|gross|net|cashflow|cash_flow|"
    r"transactions|invoice|invoices|payment|payments"
    r")(?![a-zA-Z])"
)
_ADDITIVE_SUFFIX_RE = re.compile(
    r"(?i)("
    r"_amount|_total|_count|_qty|_quantity|_units?|_fees?|_sum|"
    r"_revenue|_sales|_cost|_costs|_spend|_orders?|_volume|_income"
    r")$"
)

# Numeric columns that aggregate by AVERAGING (rates, ratios, scores, measurements):
_RATE_NAME_RE = re.compile(
    r"(?i)(?<![a-zA-Z])("
    r"rate|ratio|pct|percent|percentage|"
    r"score|scoring|index|"
    r"temperature|temp|pressure|humidity|"
    r"avg|average|mean|median|"
    r"speed|velocity|frequency|latency|duration|"
    r"age|year|hour|minute|second"
    r")(?![a-zA-Z])"
)
_RATE_SUFFIX_RE = re.compile(
    r"(?i)("
    r"_rate|_pct|_percent|_ratio|_score|_index|"
    r"_avg|_average|_mean|_median|"
    r"_per_.+"
    r")$"
)


def _normalize_name_for_match(name: str) -> str:
    """
    Normalize a column name so the additive/rate regexes can find tokens
    whether the column uses snake_case, camelCase, PascalCase, or kebab-case.
    Inserts spaces at lower->upper boundaries and replaces non-letters with
    spaces.
    """
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    spaced = re.sub(r"[^A-Za-z]+", " ", spaced)
    return spaced


def _detect_aggregation_semantics(
    name: str,
    series: pd.Series,
    is_monetary: bool,
) -> Optional[str]:
    """
    Decide whether a numeric column should be aggregated by sum or by mean
    when collapsed across time / categories.

    Returns 'additive', 'rate', or None (caller defaults to mean).

    Precedence:
      1. Monetary (currency-tagged) -> additive.
      2. Name regex match for rates (e.g. 'temperature', 'age', '*_rate') -> rate.
         Checked BEFORE additive because 'age' must not be summed.
      3. Name regex match for additives (e.g. 'revenue', '*_amount') -> additive.
      4. Bounded [0, 1] or [0, 100] floats -> rate.
      5. None.
    """
    if is_monetary:
        return "additive"
    normalized = _normalize_name_for_match(name)
    if _RATE_NAME_RE.search(normalized) or _RATE_SUFFIX_RE.search(name):
        return "rate"
    if _ADDITIVE_NAME_RE.search(normalized) or _ADDITIVE_SUFFIX_RE.search(name):
        return "additive"
    try:
        if pd.api.types.is_numeric_dtype(series):
            s = pd.to_numeric(series, errors="coerce").dropna()
            if len(s) > 10:
                mn, mx = float(s.min()), float(s.max())
                if mn >= 0.0 and mx <= 1.0 and pd.api.types.is_float_dtype(s):
                    return "rate"
    except Exception:
        pass
    return None

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

def _confidence_and_alternatives(
    name: str,
    series: pd.Series,
    profile: SyntacticProfile,
    role: str,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Derive a calibrated confidence for the chosen ``role`` plus the top-2
    runner-up roles. Metadata only — never changes the role decision, so
    existing classification behaviour is preserved.
    """
    count = max(int(profile.stats.get("count", profile.unique_count or 1)), 1)
    unique_ratio = (profile.unique_count or 0) / count

    scores: Dict[str, float] = {}
    if role == "identifier":
        try:
            scores["identifier"] = float(
                is_likely_identifier_with_confidence(series, name)[2]
            )
        except Exception:
            scores["identifier"] = 0.8
        scores["numeric" if is_numeric_dtype(series) else "categorical"] = 0.3
        scores["text"] = 0.2
    elif role in ("numeric", "boolean", "datetime"):
        scores[role] = 0.92
        scores["identifier"] = 0.35 if unique_ratio > 0.9 else 0.1
        scores["categorical"] = 0.2
    elif role == "categorical":
        scores["categorical"] = max(0.55, min(0.95, 1.0 - unique_ratio))
        scores["text"] = unique_ratio
        scores["identifier"] = 0.4 if unique_ratio > 0.9 else 0.1
    elif role == "text":
        scores["text"] = 0.6
        scores["categorical"] = max(0.0, 0.5 - unique_ratio)
        scores["identifier"] = 0.5 if unique_ratio > 0.95 else 0.1
    else:  # unknown / other
        scores[role] = 0.3
        scores["text"] = 0.25

    chosen = float(scores.get(role, 0.5))
    alternatives = [
        {"role": r, "confidence": round(float(s), 4)}
        for r, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if r != role
    ][:2]
    return round(chosen, 4), alternatives


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
        # GUARD: float columns are continuous measurements and are NEVER identifiers.
        elif is_numeric_dtype(df[name]) and df[name].dtype.kind == 'f':
            role = "numeric"

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

            # Aggregation-semantics tag: additive (sum across time) vs rate (mean).
            agg_hint = _detect_aggregation_semantics(
                name, df[name], is_monetary=('monetary' in semantic_tags)
            )
            if agg_hint == 'additive':
                semantic_tags.append('additive')
            elif agg_hint == 'rate':
                semantic_tags.append('rate')

        confidence, alternatives = _confidence_and_alternatives(
            name, df[name], profile, role
        )
        enriched_profiles[name] = EnrichedProfile(
            role=role,
            confidence=confidence,
            alternatives=alternatives,
            semantic_tags=semantic_tags,
            **profile.__dict__
        )

    logger.info(f"Layer 2: Semantic classification complete. Assigned roles to {len(enriched_profiles)} columns.")
    return enriched_profiles
