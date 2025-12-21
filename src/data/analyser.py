import logging
import pandas as pd
import numpy as np
import re
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_bool_dtype
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import math
from src.utils.identifier_detector import is_likely_identifier
from src import config

logger = logging.getLogger(__name__)

# Configuration parameters with defaults
UNIQUENESS_CUTOFF = config.UNIQUENESS_CUTOFF
AVG_LENGTH_CUTOFF = config.AVG_LENGTH_CUTOFF
MIN_DATE = config.MIN_DATE
MAX_DATE = config.MAX_DATE

# Regex patterns for identifying different field types
ID_PATTERNS = [
    r'^[0-9]+$',  # Pure numeric sequences
    r'^[a-zA-Z0-9]{8,}$',  # Long alphanumeric strings (possible UUIDs)
    r'^[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}$',  # Standard UUID format
    r'^[a-zA-Z0-9]{20,}$',  # Very long alphanumeric (possible hash)
]

# Patterns for mixed units (duration, currency, etc.)
MIXED_UNIT_PATTERNS = {
    'duration': [
        r'\d+\s*(min|mins|minute|minutes|h|hour|hours|s|sec|second|seconds|d|day|days)',  # Duration indicators
        r'[0-9:]{4,}'  # HH:MM format
    ],
    'currency': [
        r'\$\s*\d+\.?\d*',  # Dollar amounts
        r'€\s*\d+\.?\d*',  # Euro amounts
        r'£\s*\d+\.?\d*',  # Pound amounts
        r'\d+\.?\d*\s*(USD|EUR|GBP|JPY|CAD|AUD|CHF|CNY|INR)' # Currency codes
    ],
    'percentage': [
        r'\d+\.?\d*\s*%',  # Percentages
        r'\d+\.?\d*/\d+\.?\d*'  # Fractions (e.g., 3/4)
    ]
}

def _calculate_confidence(confidence_factors: Dict[str, float]) -> float:
    """
    Combine different confidence factors into a single score.
    """
    if not confidence_factors:
        return 0.5  # Default medium confidence

    weights = {
        'pattern_match': 0.4,
        'data_consistency': 0.3,
        'cardinality': 0.2,
        'semantic_context': 0.1,
    }

    total_confidence = 0.0
    total_weight = 0.0

    for factor, value in confidence_factors.items():
        weight = weights.get(factor, 0.1)
        total_confidence += value * weight
        total_weight += weight

    return total_confidence / total_weight if total_weight > 0 else 0.5


def _extract_numeric_from_mixed(text_values: pd.Series) -> Tuple[Optional[pd.Series], float]:
    """
    Extract numeric values from mixed-unit text fields (e.g. '90 min', '$123', '45%').

    Returns:
        parsed_series: Series aligned with the sample index, or None if parsing is not meaningful
        confidence: ratio of successfully parsed non-NaN values in the sample
    """
    if text_values.empty:
        return None, 0.0

    # Sample for detection (not full column conversion, just enough to decide)
    sample_size = min(100, len(text_values))
    sample_values = text_values.dropna().head(sample_size)

    if sample_values.empty:
        return None, 0.0

    parsed_values = []
    success_flags = []

    def _extract_first_number(s: str) -> Optional[float]:
        nums = re.findall(r'\d+\.?\d*', s)
        if not nums:
            return None
        try:
            return float(nums[0])
        except (ValueError, IndexError):
            return None

    for val in sample_values:
        if pd.isna(val):
            parsed_values.append(np.nan)
            success_flags.append(False)
            continue

        str_val = str(val).strip().lower()
        parsed_num = None

        # Define helper function inside
        def _extract_first_number(s: str) -> Optional[float]:
            nums = re.findall(r'\d+\.?\d*', s)
            if not nums:
                return None
            try:
                return float(nums[0])
            except (ValueError, IndexError):
                return None

        # Try duration, currency, percentage patterns in that order
        for unit_type in ["duration", "currency", "percentage"]:
            matched = False
            for pattern in MIXED_UNIT_PATTERNS[unit_type]:
                if re.search(pattern, str_val):
                    num = _extract_first_number(str_val)
                    if num is not None:
                        parsed_num = num
                        matched = True
                        break
            if matched:
                break

        if parsed_num is not None:
            parsed_values.append(parsed_num)
            success_flags.append(True)
        else:
            parsed_values.append(np.nan)
            success_flags.append(False)

    # Compute confidence based on non-NaN parses
    successful_parses = sum(success_flags)
    confidence = successful_parses / len(success_flags) if success_flags else 0.0

    # If too little was successfully parsed, this is not a reliable mixed-numeric field
    if successful_parses == 0 or confidence < 0.3:
        return None, 0.0

    parsed_series = pd.Series(parsed_values, index=sample_values.index)
    return parsed_series, confidence


def _is_multi_value_field(series: pd.Series, delimiter_chars: List[str] = [',', ';', '|', '/']) -> Tuple[bool, str, float]:
    """
    Detect if a field contains multiple values separated by delimiters.
    Optimized for large text columns to prevent memory overload.

    Args:
        series: The pandas Series to analyze
        delimiter_chars: List of potential delimiter characters

    Returns:
        Tuple of (is_multi_value, primary_delimiter, confidence_score)
    """
    # Ensure series is a pandas Series
    if not isinstance(series, pd.Series):
        try:
            series = pd.Series(series) if hasattr(series, '__iter__') and not isinstance(series, str) else pd.Series([series])
        except Exception as e:
            logger.warning(f"Could not convert input to a pandas Series in _is_multi_value_field: {e}")
            return False, "", 0.0

    if series.dtype != 'object' and not str(series.dtype).startswith('string'):
        return False, "", 0.0

    # Limit the sample size to prevent memory issues with very large series
    sample_size = min(100, len(series))
    sample_values = series.dropna().head(sample_size)

    if len(sample_values) == 0:
        return False, "", 0.0

    delimiter_scores = {}

    for delim in delimiter_chars:
        # Check if the delimiter exists in the sample values before counting
        try:
            splits = sample_values.astype(str).str.contains(delim, na=False)
            if splits.any():
                split_ratio = splits.sum() / len(sample_values)
                # Consider it multi-value if at least 20% of values contain the delimiter
                if split_ratio >= 0.2:
                    delimiter_scores[delim] = split_ratio
        except Exception as e:
            logger.warning(f"Error checking delimiter '{delim}' in series: {e}")
            continue  # Skip this delimiter if it causes issues

    if delimiter_scores:
        best_delimiter = max(delimiter_scores, key=delimiter_scores.get)
        confidence = delimiter_scores[best_delimiter]

        # Additional validation: check if the splits seem meaningful
        try:
            sample_with_delim = sample_values[sample_values.astype(str).str.contains(best_delimiter, na=False)]
            if len(sample_with_delim) > 0:
                # For text optimization: only calculate count on a subset if the sample is large
                sample_to_check = sample_with_delim.head(20) if len(sample_with_delim) > 20 else sample_with_delim
                avg_split_count = sample_to_check.astype(str).str.count(best_delimiter)

                # Ensure avg_split_count is a pandas Series before calling .mean()
                if not isinstance(avg_split_count, pd.Series):
                    avg_split_count = pd.Series(avg_split_count)
                avg_splits = avg_split_count.mean() + 1

                # If on average there are more than 2 values per field, it's likely multi-value
                if avg_splits >= 2:
                    return True, best_delimiter, confidence
        except Exception as e:
            logger.warning(f"Error validating multi-value field with delimiter '{best_delimiter}': {e}")

    return False, "", 0.0


def _is_likely_identifier(
    series: pd.Series,
    uniqueness_threshold: float = 0.95
) -> Tuple[bool, float]:
    """
    Detect if a series is likely an identifier based on various heuristics.

    We now combine:
      - high uniqueness
      - ID-like patterns
      - column name hints

    This is to avoid misclassifying genuine numeric metrics as identifiers.
    """
    n_total = len(series)
    if n_total == 0:
        return False, 0.0

    n_unique = series.nunique(dropna=True)
    unique_ratio = n_unique / n_total if n_total > 0 else 0.0

    # Check if column name suggests it's an ID
    name_lower = (series.name or "").lower()
    id_name_tokens = [
        "id", "identifier", "uuid", "guid", "key", "account", "user", "customer",
        "client", "booking", "transaction", "order", "invoice", "code", "number"
    ]
    looks_like_id_name = any(token in name_lower for token in id_name_tokens)

    # Check for ID-like patterns in the values
    sample_values = series.dropna().head(min(100, len(series))).astype(str)
    id_pattern_matches = 0

    for val in sample_values:
        for pattern in ID_PATTERNS:
            if re.fullmatch(pattern, val.strip(), re.IGNORECASE):
                id_pattern_matches += 1
                break

    pattern_confidence = id_pattern_matches / len(sample_values) if len(sample_values) > 0 else 0.0

    # High uniqueness is necessary for ID classification, but not sufficient on its own
    # Only classify as ID if either name looks like ID or strong pattern matching occurs
    if unique_ratio > uniqueness_threshold:
        if looks_like_id_name:
            # Strong confidence if both high uniqueness and name suggests ID
            confidence = min(1.0, unique_ratio * 1.2)  # Boost for name match
            return True, confidence
        elif pattern_confidence > 0.5:
            # Moderate confidence if high uniqueness and strong pattern match
            confidence = min(0.9, unique_ratio * pattern_confidence * 1.5)
            return True, confidence
        elif unique_ratio > 0.99:  # Very high uniqueness might indicate ID anyway
            # Check with the centralized identifier detector as well
            centralized_is_id = is_likely_identifier(series, name=series.name or "")
            if centralized_is_id:
                # Lower confidence if just high uniqueness but no other indicators
                confidence = min(0.8, unique_ratio)
                return True, confidence
            else:
                # Lower confidence if just high uniqueness but no other indicators
                confidence = min(0.7, unique_ratio)
                return True, confidence

    return False, 0.0


def _infer_role_advanced(
    series: pd.Series,
    uniqueness_cutoff: float = 0.5,
    avg_length_cutoff: int = 30
) -> Tuple[str, float, str, Dict[str, float], List[str]]:
    """
    Advanced role inference with confidence scoring and semantic tags.

    Returns:
        role: core role ("numeric", "categorical", "datetime", "text", "boolean", "identifier", "ordinal")
        confidence: 0–1
        provenance: short string indicating the main decision path
        confidence_factors: breakdown of the confidence calculation
        semantic_tags: list of semantic hints like ["geographic"], ["duration"], ["monetary"], ["percentage"], ["multi_value"]
    """
    # Ensure series is a pandas Series
    if not isinstance(series, pd.Series):
        series = pd.Series(series) if hasattr(series, '__iter__') and not isinstance(series, str) else pd.Series([series])

    semantic_tags = []

    if series.empty:
        return "text", 0.5, "empty_series", {"data_consistency": 0.0}, semantic_tags

    # 0) Identifier detection (uses name + uniqueness + patterns)
    is_id, id_confidence = _is_likely_identifier(series)
    if is_id and id_confidence > 0.7:
        return "identifier", id_confidence, f"identifier_detection_{id_confidence:.2f}", {
            "pattern_match": id_confidence,
            "data_consistency": 1.0,
            "cardinality": 1.0
        }, semantic_tags

    # 1) Multi-value text field detection
    is_multi, delimiter, multi_confidence = _is_multi_value_field(series)
    if is_multi and multi_confidence > 0.5:
        semantic_tags.append("multi_value")
        return "text", multi_confidence, f"multivalue_delim_{delimiter}_conf_{multi_confidence:.2f}", {
            "pattern_match": multi_confidence,
            "data_consistency": 0.8,
            "cardinality": 0.3
        }, semantic_tags

    # 2) Boolean detection
    unique_vals = series.dropna().unique()
    if len(unique_vals) <= 2:
        unique_str = [str(val).lower() for val in unique_vals if pd.notna(val)]
        boolean_indicators = {'true', 'false', 'yes', 'no', '1', '0', 't', 'f', 'y', 'n', '1.0', '0.0', 'on', 'off', 'true.', 'false.'}
        if set(unique_str).issubset(boolean_indicators) or pd.api.types.is_bool_dtype(series):
            return "boolean", 0.9, f"boolean_values_{len(unique_vals)}", {
                "pattern_match": 0.9,
                "data_consistency": 1.0,
                "cardinality": 0.1
            }, semantic_tags

    name_lower = (series.name or "").lower()

    # 3) Native numeric dtype (includes potential geo/monetary/unit values which we tag separately)
    if pd.api.types.is_numeric_dtype(series):
        # Check for potential datetime in numeric format (e.g. years)
        s_nonnull = series.dropna()
        if not s_nonnull.empty:
            # Use pd.to_numeric to ensure comparison is safe
            try:
                s_numeric = pd.to_numeric(s_nonnull)
                col_min = float(s_numeric.min())
                col_max = float(s_numeric.max())
            except (ValueError, TypeError):
                logger.warning(f"Could not convert column {series.name} to numeric for year check.")
                col_min = float('inf')
                col_max = float('-inf')

            # Year-like numeric datetime
            if (
                MIN_DATE <= col_min <= MAX_DATE and
                MIN_DATE <= col_max <= MAX_DATE and
                any(keyword in name_lower for keyword in ["year", "yr"])
            ):
                return "datetime", 0.8, "numeric_year_keyword", {
                    "pattern_match": 0.7,
                    "data_consistency": 0.9,
                    "semantic_context": 0.8
                }, semantic_tags

        # For numeric data, check for semantic tags based on name
        # Geographic indicators
        geo_indicators = ["lat", "lon", "long", "latitude", "longitude", "coord", "x_", "y_", "x", "y"]
        if any(indicator in name_lower for indicator in geo_indicators):
            semantic_tags.append("geographic")

        # Currency indicators
        money_tokens = ["price", "cost", "revenue", "sales", "amount", "income", "expense", "salary", "wage", "fee",
                        "charge", "payment", "profit", "budget", "fund", "investment", "capital", "value", "total"]
        if any(token in name_lower for token in money_tokens):
            semantic_tags.append("monetary")

        # Percentage indicators
        perc_tokens = ["percent", "percentage", "pct", "ratio", "rate", "_pct", "prop", "proportion"]
        if any(token in name_lower for token in perc_tokens):
            semantic_tags.append("percentage")

        # Duration/timing indicators
        dur_tokens = ["duration", "length", "time", "period", "span", "interval", "delay", "lag", "gap"]
        if any(token in name_lower for token in dur_tokens):
            semantic_tags.append("duration")

        return "numeric", 0.8, "numeric_dtype", {
            "pattern_match": 0.8,
            "data_consistency": 0.9,
            "cardinality": 0.5
        }, semantic_tags

    # 4) Native datetime dtype
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime", 0.9, "datetime_dtype", {
            "pattern_match": 0.9,
            "data_consistency": 1.0,
            "cardinality": 0.5
        }, semantic_tags

    # 5) Attempt to parse datetime from object/string
    if series.dtype == "object":
        sample = series.dropna().astype(str).head(50)  # Sample for performance
        if not sample.empty:
            try:
                parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
                valid_dates = parsed.notna().mean()
                if valid_dates > 0.7:  # Majority are parseable as dates
                    return "datetime", valid_dates, f"datetime_parsed_{valid_dates:.2f}", {
                        "pattern_match": valid_dates,
                        "data_consistency": 0.8,
                        "cardinality": 0.5
                    }, semantic_tags
            except Exception as e:
                logger.debug(f"Date parsing failed for column {series.name}: {e}")

    # 6) Mixed-unit numeric embedded in text (duration, currency, percentage)
    if series.dtype == "object":
        parsed_nums, mixed_confidence = _extract_numeric_from_mixed(series)
        if parsed_nums is not None and mixed_confidence > 0.6:
            # Determine the semantic tag based on patterns in the original text
            sample_values = series.dropna().head(min(50, len(series))).astype(str)
            unit_votes = Counter()

            for val in sample_values:
                val_lower = val.lower()
                # Check for duration patterns
                for pattern in MIXED_UNIT_PATTERNS['duration']:
                    if re.search(pattern, val_lower):
                        unit_votes['duration'] += 1
                        break
                # Check for currency patterns
                for pattern in MIXED_UNIT_PATTERNS['currency']:
                    if re.search(pattern, val_lower):
                        unit_votes['currency'] += 1
                        break
                # Check for percentage patterns
                for pattern in MIXED_UNIT_PATTERNS['percentage']:
                    if re.search(pattern, val_lower):
                        unit_votes['percentage'] += 1
                        break

            # Assign semantic tag based on most common pattern
            if unit_votes:
                dominant_unit, _ = unit_votes.most_common(1)[0]
                if dominant_unit == 'duration':
                    semantic_tags.append("duration")
                elif dominant_unit == 'currency':
                    semantic_tags.append("monetary")
                elif dominant_unit == 'percentage':
                    semantic_tags.append("percentage")

            # Still return numeric role but with semantic tags
            return "numeric", mixed_confidence, "mixed_unit_numeric", {
                "pattern_match": mixed_confidence,
                "data_consistency": 0.7,
                "cardinality": 0.4
            }, semantic_tags

    # 7) Check for ordinal categorical patterns (e.g. small sets of ordered categories)
    if series.dtype == "object":
        str_values = series.dropna().astype(str).str.lower().unique()
        ordinal_patterns = [
            {'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth'},
            {'low', 'medium', 'high'},
            {'small', 'medium', 'large'},
            {'beginner', 'intermediate', 'advanced'},
            {'junior', 'senior', 'expert'},
            {'one', 'two', 'three', 'four', 'five'},
            {'none', 'low', 'medium', 'high', 'maximum'},
            {'min', 'mid', 'max'},
            {'start', 'middle', 'end'},
            {'level 1', 'level 2', 'level 3', 'level 4', 'level 5'},
            {'grade 1', 'grade 2', 'grade 3', 'grade 4', 'grade 5'},
            {'tier 1', 'tier 2', 'tier 3'},
            {'class 1', 'class 2', 'class 3'}
        ]

        for pattern_set in ordinal_patterns:
            if set(str_values).issubset(pattern_set):
                return "ordinal", 0.85, f"ordinal_pattern_{len(set(str_values))}", {
                    "pattern_match": 0.85,
                    "data_consistency": 0.8,
                    "semantic_context": 0.6
                }, semantic_tags

    # 8) For text/object types, check semantic hints in name and decide role based on characteristics
    # Geographic semantic tag based on column name (but keep text role if not obviously categorical)
    geo_name_tokens = ["country", "city", "state", "province", "address", "location", "region", "zone", "lat", "lon", "long", "coord"]
    if any(indicator in name_lower for indicator in geo_name_tokens):
        semantic_tags.append("geographic")

    text_indicators = ["desc", "comment", "note", "text", "message", "review", "comments", "notes", "info", "details"]
    if any(indicator in name_lower for indicator in text_indicators):
        semantic_tags.append("textual")

    # 9) Use cardinality and length heuristics to decide between text and categorical
    n_rows = len(series)
    n_unique = series.nunique(dropna=True)
    unique_ratio = n_unique / n_rows if n_rows > 0 else 0.0

    if series.notna().any():
        try:
            if not isinstance(series, pd.Series):
                series = pd.Series(series)
            avg_len = series.dropna().astype(str).str.len().mean()
        except Exception as e:
            logger.warning(f"Could not compute average length for series {series.name}: {e}")
            avg_len = 0
    else:
        avg_len = 0

    # Very high cardinality with long values = text
    if unique_ratio > 0.9 and n_unique > 100 and avg_len > 20:
        return "text", 0.9, f"high_cardinality_long_text_unique_ratio_{unique_ratio:.2f}_avg_len_{avg_len:.1f}", {
            "pattern_match": 0.9,
            "data_consistency": 0.9,
            "cardinality": 1.0
        }, semantic_tags
    # High cardinality or long values = text
    elif unique_ratio > uniqueness_cutoff or avg_len > avg_length_cutoff:
        return "text", max(0.6, min(0.9, unique_ratio)), f"high_cardinality_or_long_values_unique_ratio_{unique_ratio:.2f}_avg_len_{avg_len:.1f}", {
            "pattern_match": min(0.9, unique_ratio),
            "data_consistency": 0.7,
            "cardinality": unique_ratio
        }, semantic_tags
    # Low cardinality = categorical
    elif unique_ratio < 0.05 and n_unique <= 10:
        return "categorical", max(0.7, 1.0 - unique_ratio), f"low_cardinality_n_{n_unique}_unique_ratio_{unique_ratio:.2f}", {
            "pattern_match": 0.8,
            "data_consistency": 0.8,
            "cardinality": 1.0 - unique_ratio
        }, semantic_tags
    else:
        # Medium cardinality - categorical with lower confidence
        return "categorical", 0.6, f"medium_cardinality_n_{n_unique}_unique_ratio_{unique_ratio:.2f}", {
            "pattern_match": 0.6,
            "data_consistency": 0.7,
            "cardinality": 0.6
        }, semantic_tags


def build_dataset_profile(df: pd.DataFrame, max_cols: int = 50, sample_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Build a rich dataset profile with confidence scores and semantic tags.
    Optionally samples the DataFrame for performance.

    Args:
        df: Input DataFrame
        max_cols: Maximum number of columns to profile
        sample_size: If provided, profile based on a random sample of this size

    Returns:
        A dict with structured profile information.
    """
    if df.empty:
        logger.warning("DataFrame is empty, returning empty profile")
        return {
            "n_rows": 0,
            "n_cols": 0,
            "role_counts": {"numeric": 0, "datetime": 0, "categorical": 0, "text": 0, "identifier": 0, "boolean": 0, "ordinal": 0, "other": 0},
            "columns": []
        }

    n_rows_original = int(len(df))
    n_cols = int(df.shape[1])

    # Apply sampling if requested
    if sample_size and sample_size < n_rows_original:
        logger.info(f"Sampling {sample_size} rows from {n_rows_original} for profiling.")
        df_to_profile = df.sample(n=sample_size, random_state=42).copy()
        n_rows = sample_size
    else:
        df_to_profile = df
        n_rows = n_rows_original

    # Set limits
    max_cols = min(max_cols, n_cols)

    columns = []
    role_counts = {
        "numeric": 0,
        "datetime": 0,
        "categorical": 0,
        "text": 0,
        "identifier": 0,
        "boolean": 0,
        "ordinal": 0,
        "other": 0,
    }

    for i, col in enumerate(df_to_profile.columns):
        if i >= max_cols:
            break

        # Skip columns with no data or that are entirely null
        if df_to_profile[col].isna().all():
            logger.info(f"Skipping column {col} as it contains only null values")
            continue

        s = df_to_profile[col]

        # Handle edge case where column has all NaN values
        if s.isna().all():
            role = "text"
            confidence = 0.3
            provenance = "all_nan"
            confidence_factors = {"data_consistency": 0.0}
            semantic_tags = []
        else:
            role, confidence, provenance, confidence_factors, semantic_tags = _infer_role_advanced(
                s, uniqueness_cutoff=UNIQUENESS_CUTOFF, avg_length_cutoff=AVG_LENGTH_CUTOFF
            )

        # Track role counts
        if role in role_counts:
            role_counts[role] += 1
        else:
            role_counts["other"] += 1

        # Default: no stats and no top categories
        stats = None
        top_categories = []

        # Compute column statistics based on role
        if role in ("numeric", "numeric_duration", "numeric_currency", "numeric_percentage", "numeric_mixed"):
            # Process as numeric
            # Ensure s is a pandas Series before processing
            if not isinstance(s, pd.Series):
                s = pd.Series(s) if hasattr(s, '__iter__') and not isinstance(s, str) else pd.Series([s])
            s_clean = pd.to_numeric(s, errors='coerce')
            s_clean = s_clean.dropna()

            if len(s_clean) > 0:
                # Compute all statistics in one pass for efficiency
                try:
                    stats = {
                        "min": float(s_clean.min()) if len(s_clean) > 0 else None,
                        "max": float(s_clean.max()) if len(s_clean) > 0 else None,
                        "mean": float(s_clean.mean()) if len(s_clean) > 0 else None,
                        "std": float(s_clean.std()) if len(s_clean) > 1 else 0.0,
                        "median": float(s_clean.median()) if len(s_clean) > 0 else None,
                        "q25": float(s_clean.quantile(0.25)) if len(s_clean) > 0 else None,
                        "q75": float(s_clean.quantile(0.75)) if len(s_clean) > 0 else None,
                        "sum": float(s_clean.sum()) if len(s_clean) > 0 else None,
                        "variance": float(s_clean.var()) if len(s_clean) > 1 else 0.0,
                        "skewness": float(s_clean.skew()) if len(s_clean) > 2 else 0.0,
                        "kurtosis": float(s_clean.kurtosis()) if len(s_clean) > 3 else 0.0,
                        "count": int(len(s_clean))
                    }
                except (TypeError, ValueError) as e:
                    logger.warning(f"Error calculating statistics for numeric column '{col}': {e}")
                    stats = None

                # Add specific semantic-based stats if stats were calculated successfully
                if stats and "monetary" in semantic_tags:
                    stats["currency_units"] = float(s_clean.abs().sum())  # Total monetary value
                elif stats and "percentage" in semantic_tags:
                    # Make sure s_clean is still a Series before calculating mean again
                    if isinstance(s_clean, pd.Series):
                        stats["average_percentage"] = float(s_clean.mean()) if s_clean.empty is False else 0.0
                    else:
                        stats["average_percentage"] = 0.0
                elif stats and "duration" in semantic_tags:
                    stats["total_duration"] = float(s_clean.sum())

        elif role == "datetime":
            try:
                # Ensure s is a pandas Series before processing
                if not isinstance(s, pd.Series):
                    s = pd.Series(s) if hasattr(s, '__iter__') and not isinstance(s, str) else pd.Series([s])

                s_dt = pd.to_datetime(s, errors="coerce")
                s_dt_clean = s_dt.dropna()
                if len(s_dt_clean) > 0:
                    stats = {
                        "min": s_dt_clean.min().isoformat() if not s_dt_clean.empty else None,
                        "max": s_dt_clean.max().isoformat() if not s_dt_clean.empty else None,
                        "range_days": (s_dt_clean.max() - s_dt_clean.min()).days if not s_dt_clean.empty else 0,
                        "count": int(len(s_dt_clean))
                    }
            except Exception as e:
                logger.warning(f"Error computing datetime stats for {col}: {e}")
                stats = None

        elif role in ("categorical", "ordinal", "identifier", "boolean", "text"):
            try:
                # Ensure s is a pandas Series before processing
                if not isinstance(s, pd.Series):
                    s = pd.Series(s) if hasattr(s, '__iter__') and not isinstance(s, str) else pd.Series([s])

                value_counts = s.value_counts(dropna=True)
                top_categories = [
                    {"value": str(idx), "count": int(cnt), "percentage": f"{(cnt/len(s))*100:.2f}%"}
                    for idx, cnt in value_counts.head(10).items()  # Top 10 categories
                ]

                # For categorical and ordinal roles, also compute additional stats
                if role in ("categorical", "ordinal"):
                    stats = {
                        "unique_count": int(s.nunique()) if hasattr(s, 'nunique') else len(set(s)) if isinstance(s, (list, tuple)) else 0,
                        "unique_ratio": float(s.nunique() / len(s)) if hasattr(s, 'nunique') and len(s) > 0 else
                                         float(len(set(s)) / len(s)) if isinstance(s, (list, tuple)) and len(s) > 0 else 0.0,
                        "top_category": str(value_counts.index[0]) if not value_counts.empty else None,
                        "top_category_count": int(value_counts.iloc[0]) if not value_counts.empty else 0,
                        "top_category_percentage": float((value_counts.iloc[0] / len(s)) * 100) if len(s) > 0 and not value_counts.empty else 0.0,
                        "count": int(len(s))
                    }
                elif role == "identifier":
                    stats = {
                        "unique_count": int(s.nunique()) if hasattr(s, 'nunique') else len(set(s)) if isinstance(s, (list, tuple)) else 0,
                        "unique_ratio": float(s.nunique() / len(s)) if hasattr(s, 'nunique') and len(s) > 0 else
                                         float(len(set(s)) / len(s)) if isinstance(s, (list, tuple)) and len(s) > 0 else 0.0,
                        "is_unique": bool(s.nunique() == len(s)) if hasattr(s, 'nunique') else
                                    bool(len(set(s)) == len(s)) if isinstance(s, (list, tuple)) else False,
                        "count": int(len(s))
                    }
                elif role == "boolean":
                    stats = {
                        "true_count": int((s == True).sum()) if hasattr(s, 'sum') else sum(1 for x in s if x is True or x == 1) if isinstance(s, (list, tuple)) else 0,
                        "false_count": int((s == False).sum()) if hasattr(s, 'sum') else sum(1 for x in s if x is False or x == 0) if isinstance(s, (list, tuple)) else 0,
                        "true_ratio": float((s == True).sum() / len(s)) if hasattr(s, 'sum') and len(s) > 0 else
                                   float(sum(1 for x in s if x is True or x == 1) / len(s)) if isinstance(s, (list, tuple)) and len(s) > 0 else 0.0,
                        "count": int(len(s))
                    }
            except Exception as e:
                logger.warning(f"Error computing categorical stats for {col}: {e}")
                top_categories = []
                stats = None

        # Build the column profile
        column_profile = {
            "name": col,
            "dtype": str(s.dtype),
            "role": role,
            "confidence": float(confidence),
            "confidence_factors": confidence_factors,
            "semantic_tags": semantic_tags,
            "missing_count": int(s.isna().sum()),
            "unique_count": int(s.nunique()),
            "stats": stats,
            "top_categories": top_categories,
            "provenance": provenance
        }

        columns.append(column_profile)

    logger.info(f"Dataset profile built for {len(columns)} out of {n_cols} total columns")
    logger.info(f"Role counts: {role_counts}")
    logger.info(f"Profiled on {n_rows} rows (original: {n_rows_original})")

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "role_counts": role_counts,
        "columns": columns,
    }


def basic_profile(df: pd.DataFrame, max_cols: int = 10) -> List[Dict[str, Any]]:
    """
    Basic profile per column (maintained for API compatibility).
    Uses the advanced profiling logic internally.
    """
    if df is None or df.empty:
        logger.warning("DataFrame is None or empty, returning empty profile")
        return []

    # Ensure df is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # Use the advanced profile function, then extract basic info
    advanced_profile = build_dataset_profile(df, max_cols=max_cols, sample_size=None) # Don't sample for basic profile
    basic_profile_list = []
    for col_info in advanced_profile.get("columns", []):
        basic_stats = col_info.get("stats", {}) or {}
        basic_profile_list.append({
            "column": col_info["name"],
            "dtype": col_info["dtype"],
            "missing": col_info["missing_count"],
            "unique": col_info["unique_count"],
            "role": col_info["role"],
            "confidence": col_info["confidence"],
            "semantic_tags": col_info["semantic_tags"],
            "stats": basic_stats,
            "provenance": col_info["provenance"]
        })

    logger.info(f"Basic profile generated for {len(basic_profile_list)} columns")
    return basic_profile_list
