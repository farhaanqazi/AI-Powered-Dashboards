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
        # This regex will find the first floating point number in a string
        nums = re.findall(r'-?\d+\.?\d*', str(s))
        if not nums:
            return None
        try:
            return float(nums[0])
        except (ValueError, IndexError):
            return None

    for val in sample_values:
        parsed_num = _extract_first_number(val)
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

    # Return a full series of parsed numbers, not just a sample
    return text_values.apply(_extract_first_number), confidence


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


def _is_likely_identifier_local(series: pd.Series) -> Tuple[bool, float]:
    """
    More robust identifier detection. Requires more than just high uniqueness
    to classify a column as an identifier, preventing misclassification of
    high-cardinality text or numeric fields.
    """
    n_total = len(series)
    if n_total == 0:
        return False, 0.0

    n_unique = series.nunique(dropna=True)
    unique_ratio = n_unique / n_total if n_total > 0 else 0.0

    # High uniqueness is a prerequisite, but not sufficient on its own.
    if unique_ratio < 0.9:
        return False, 0.0

    # Strong signals: Name or format patterns
    name_lower = (series.name or "").lower()
    id_name_keywords = ["id", "key", "uuid", "guid", "code", "token", "hash"]
    if any(keyword in name_lower for keyword in id_name_keywords):
        return True, 0.9 # High confidence if name matches

    # Check for formats like UUIDs in a sample
    if series.dtype == 'object':
        sample = series.dropna().head(20).astype(str)
        uuid_pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
        if not sample.empty and sample.str.match(uuid_pattern).mean() > 0.5:
            return True, 0.95 # Very high confidence for UUID format

    # If we only have high uniqueness but no other signals, it's not a confident identifier.
    # This prevents long text or unique floats from being misclassified.
    return False, 0.0


def _infer_role_advanced(
    series: pd.Series,
    uniqueness_cutoff: float = 0.95, # Adjusted threshold
    avg_length_cutoff: int = 100
) -> Tuple[str, float, str, Dict[str, float], List[str]]:
    """
    Final, most robust version of the role inference function with a corrected
    classification hierarchy to properly handle mixed-numeric object types.
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    semantic_tags = []
    if series.empty or series.isna().all():
        return "text", 0.3, "all_nan", {}, semantic_tags
    if series.dropna().nunique() <= 1:
        return "text", 0.4, "near_constant_text", {}, semantic_tags

    # --- Start Classification Hierarchy ---

    # 1. Unambiguous types first
    if is_bool_dtype(series):
        return "boolean", 0.9, "boolean_native", {}, semantic_tags
    if is_datetime64_any_dtype(series):
        return "datetime", 0.9, "datetime_native", {}, semantic_tags

    # 2. Check for identifiers (works for ALL dtypes, including numeric)
    is_id, id_confidence = _is_likely_identifier_local(series)
    if is_id:
        return "identifier", id_confidence, "identifier_detection", {}, semantic_tags
    
    # 3. Now check numeric dtype (after ruling out identifiers)
    if is_numeric_dtype(series):
        return "numeric", 0.9, "numeric_native", {}, semantic_tags


    # 4. If it's an object, perform further multi-stage checks
    if series.dtype == "object":
        # Check for datetime strings (if not already handled as native datetime)
        try:
            sample = series.dropna().head(50)
            if not sample.empty and pd.to_datetime(sample, errors='coerce').notna().mean() > 0.7:
                return "datetime", 0.8, "datetime_parsed", {}, semantic_tags
        except Exception:
            pass
        
        # Second, try to coerce to pure numeric strings (if not already handled as native numeric)
        s_numeric_coerced = pd.to_numeric(series, errors='coerce')
        if series.notna().sum() > 0 and (s_numeric_coerced.notna().sum() / series.notna().sum()) > 0.85:
            return "numeric", 0.85, "numeric_coerced", {}, semantic_tags

        # Third, check for mixed formats like currency ('$100')
        parsed_nums, mixed_confidence = _extract_numeric_from_mixed(series)
        if parsed_nums is not None and mixed_confidence > 0.6:
            if "$" in series.to_string() or "€" in series.to_string() or "£" in series.to_string():
                 semantic_tags.append("monetary")
            return "numeric", mixed_confidence, "mixed_unit_numeric", {}, semantic_tags

        # Finally, distinguish between categorical and text if still an object
        n_rows = len(series)
        n_unique = series.nunique(dropna=True)
        unique_ratio = n_unique / n_rows if n_rows > 0 else 0.0
        
        if unique_ratio < uniqueness_cutoff and n_unique < 500:
            return "categorical", 0.7, "low_cardinality_categorical", {}, semantic_tags
        else:
            return "text", 0.7, "high_cardinality_text", {}, semantic_tags

    # Fallback for any other unhandled dtypes (should be rare)
    return "text", 0.2, "fallback", {}, semantic_tags


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
        stats = {}
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
                    # Calculate basic statistics
                    min_val = float(s_clean.min()) if len(s_clean) > 0 else None
                    max_val = float(s_clean.max()) if len(s_clean) > 0 else None
                    mean_val = float(s_clean.mean()) if len(s_clean) > 0 else None
                    std_val = float(s_clean.std()) if len(s_clean) > 1 else 0.0
                    median_val = float(s_clean.median()) if len(s_clean) > 0 else None
                    q25_val = float(s_clean.quantile(0.25)) if len(s_clean) > 0 else None
                    q75_val = float(s_clean.quantile(0.75)) if len(s_clean) > 0 else None
                    sum_val = float(s_clean.sum()) if len(s_clean) > 0 else None
                    variance_val = float(s_clean.var()) if len(s_clean) > 1 else 0.0

                    # Calculate skewness and kurtosis with proper error handling
                    if len(s_clean) > 2:
                        try:
                            skewness_val = float(s_clean.skew())
                            if pd.isna(skewness_val) or not np.isfinite(skewness_val):
                                skewness_val = 0.0
                        except (TypeError, ValueError, RuntimeWarning):
                            skewness_val = 0.0
                    else:
                        skewness_val = 0.0

                    if len(s_clean) > 3:
                        try:
                            kurtosis_val = float(s_clean.kurtosis())
                            if pd.isna(kurtosis_val) or not np.isfinite(kurtosis_val):
                                kurtosis_val = 0.0
                        except (TypeError, ValueError, RuntimeWarning):
                            kurtosis_val = 0.0
                    else:
                        kurtosis_val = 0.0

                    stats = {
                        "min": min_val,
                        "max": max_val,
                        "mean": mean_val,
                        "std": std_val,
                        "median": median_val,
                        "q25": q25_val,
                        "q75": q75_val,
                        "sum": sum_val,
                        "variance": variance_val,
                        "skewness": skewness_val,
                        "kurtosis": kurtosis_val,
                        "count": int(len(s_clean))
                    }
                except (TypeError, ValueError) as e:
                    logger.warning(f"Error calculating statistics for numeric column '{col}': {e}")
                    stats = {}

                # Add specific semantic-based stats if stats were calculated successfully
                if stats and "monetary" in semantic_tags and len(s_clean) > 0:
                    try:
                        stats["currency_units"] = float(s_clean.abs().sum())  # Total monetary value
                    except (TypeError, ValueError):
                        logger.warning(f"Could not calculate currency units for column '{col}'")
                elif stats and "percentage" in semantic_tags and len(s_clean) > 0:
                    try:
                        # Make sure s_clean is still a Series before calculating mean again
                        if isinstance(s_clean, pd.Series):
                            stats["average_percentage"] = float(s_clean.mean()) if s_clean.empty is False else 0.0
                        else:
                            stats["average_percentage"] = 0.0
                    except (TypeError, ValueError):
                        logger.warning(f"Could not calculate average percentage for column '{col}'")
                elif stats and "duration" in semantic_tags and len(s_clean) > 0:
                    try:
                        stats["total_duration"] = float(s_clean.sum())
                    except (TypeError, ValueError):
                        logger.warning(f"Could not calculate total duration for column '{col}'")

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
                stats = {}

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
                stats = {}

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