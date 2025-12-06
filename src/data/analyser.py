# src/data/analyser.py
import logging
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from functools import lru_cache

logger = logging.getLogger(__name__)

# Configuration parameters with defaults
UNIQUENESS_CUTOFF = 0.5  # Ratio where text vs categorical is determined
AVG_LENGTH_CUTOFF = 30  # Character length where text vs categorical is determined
MIN_DATE = 1900
MAX_DATE = 2100

def basic_profile(df: pd.DataFrame, max_cols: int = 10) -> List[Dict[str, Any]]:
    """
    Simple profile per column:
    - column name
    - dtype
    - missing values
    - unique values
    - role: 'numeric' or 'non-numeric'
    - min, max, mean: only for numeric columns

    Args:
        df: Input DataFrame
        max_cols: Maximum number of columns to analyze
    """
    if df.empty:
        logger.warning("DataFrame is empty, returning empty profile")
        return []

    profile = []
    for i, col in enumerate(df.columns):
        if i >= max_cols:
            break
        series = df[col]

        if is_numeric_dtype(series):
            role = "numeric"
            # Validate that we have non-NaN values before computing stats
            non_nan_series = series.dropna()
            if len(non_nan_series) > 0:
                col_min = float(non_nan_series.min()) if non_nan_series.notna().any() else None
                col_max = float(non_nan_series.max()) if non_nan_series.notna().any() else None
                col_mean = float(non_nan_series.mean()) if non_nan_series.notna().any() else None
            else:
                col_min = None
                col_max = None
                col_mean = None
        else:
            role = "non-numeric"
            col_min = None
            col_max = None
            col_mean = None

        profile.append({
            "column": col,
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique()),
            "role": role,
            "min": col_min,
            "max": col_max,
            "mean": col_mean,
            "provenance": "basic_profile"  # Add provenance info
        })

    logger.info(f"Basic profile generated for {len(profile)} columns")
    return profile


def _infer_role(series: pd.Series, uniqueness_cutoff: float = UNIQUENESS_CUTOFF,
                avg_length_cutoff: int = AVG_LENGTH_CUTOFF,
                min_date: int = MIN_DATE,
                max_date: int = MAX_DATE) -> Tuple[str, str]:
    """
    Determine role with configurable thresholds and provenance info:
    - numeric
    - datetime
    - categorical
    - text
    - boolean (new)
    - ordered categorical (new)

    Args:
        series: Input pandas Series
        uniqueness_cutoff: Ratio threshold to distinguish categorical vs text
        avg_length_cutoff: Average character length threshold to distinguish categorical vs text
        min_date: Minimum year value to be considered as datetime
        max_date: Maximum year value to be considered as datetime

    Returns:
        Tuple of (role, provenance_info)
    """
    if series.empty:
        logger.warning(f"Empty series, defaulting to 'text' role")
        return "text", "empty_series"

    # Check for boolean type
    unique_vals = series.dropna().unique()
    if len(unique_vals) <= 2:
        unique_str = [str(val).lower() for val in unique_vals if pd.notna(val)]
        boolean_indicators = {'true', 'false', 'yes', 'no', '1', '0', 't', 'f', 'y', 'n', '1.0', '0.0'}
        if set(unique_str).issubset(boolean_indicators) or series.dtype == bool:
            return "boolean", f"boolean_values_{len(unique_vals)}"

    # 1) Numeric
    if is_numeric_dtype(series):
        # year-like numeric → datetime
        s_nonnull = series.dropna()
        if not s_nonnull.empty:
            try:
                col_min = float(s_nonnull.min())
                col_max = float(s_nonnull.max())
            except Exception:
                logger.warning(f"Could not compute min/max for numeric series: {series.name}")
                col_min = None
                col_max = None

            name_lower = (series.name or "").lower()
            if (col_min is not None and col_max is not None and
                min_date <= col_min <= max_date and
                min_date <= col_max <= max_date and
                any(keyword in name_lower for keyword in ["year", "yr"])):
                return "datetime", "numeric_year_keyword"

        return "numeric", "numeric_dtype"

    # 2) Native datetime dtype
    if is_datetime64_any_dtype(series):
        return "datetime", "datetime_dtype"

    # 3) Try to detect datetime even if stored as object
    if series.dtype == "object":
        sample = series.dropna().astype(str).head(50)
        if not sample.empty:
            try:
                parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
                if parsed.notna().mean() > 0.7:
                    return "datetime", "datetime_parsed"
            except Exception as e:
                logger.warning(f"Error parsing datetime for series {series.name}: {e}")

    # 4) Check for ordered categorical/ordinal based on common patterns
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

        for pattern in ordinal_patterns:
            if set(str_values).issubset(pattern):
                return "ordinal", f"ordinal_pattern_{len(str_values)}"

    # 5) Try to detect geo/text language hints
    # Check if series name suggests geographic content
    name_lower = (series.name or "").lower()
    geo_indicators = ["country", "city", "state", "province", "address", "location", "region", "zone", "lat", "lon", "long", "coord"]
    if any(indicator in name_lower for indicator in geo_indicators):
        return "geographic", "name_geographic_hint"

    text_indicators = ["desc", "comment", "note", "text", "message", "review", "comment"]
    if any(indicator in name_lower for indicator in text_indicators):
        return "text", "name_text_hint"

    # 6) Distinguish categorical vs text based on characteristics
    n_rows = len(series)
    n_unique = series.nunique(dropna=True)
    unique_ratio = n_unique / n_rows if n_rows > 0 else 0

    if series.notna().any():
        # Calculate average length, but handle potential non-string values
        try:
            avg_len = series.dropna().astype(str).str.len().mean()
        except:
            logger.warning(f"Could not compute average length for series {series.name}")
            avg_len = 0
    else:
        avg_len = 0

    # Long / high-cardinality strings → text
    if avg_len > avg_length_cutoff or unique_ratio > uniqueness_cutoff:
        return "text", f"high_cardinality_avg_len_{avg_len:.1f}_ratio_{unique_ratio:.2f}"

    # Otherwise categorical
    return "categorical", f"low_cardinality_avg_len_{avg_len:.1f}_ratio_{unique_ratio:.2f}"


def build_dataset_profile(df: pd.DataFrame, max_cols: int = 50,
                         uniqueness_cutoff: float = UNIQUENESS_CUTOFF,
                         avg_length_cutoff: int = AVG_LENGTH_CUTOFF) -> Dict[str, Any]:
    """
    Build a richer dataset profile used across the app with configurable thresholds.
    Returns a dict shaped like DatasetProfile.

    Args:
        df: Input DataFrame
        max_cols: Maximum number of columns to profile
        uniqueness_cutoff: Ratio threshold for categorical vs text
        avg_length_cutoff: Length threshold for categorical vs text
    """
    if df.empty:
        logger.warning("DataFrame is empty, returning empty profile")
        return {
            "n_rows": 0,
            "n_cols": 0,
            "role_counts": {"numeric": 0, "datetime": 0, "categorical": 0, "text": 0},
            "columns": []
        }

    n_rows = int(len(df))
    n_cols = int(df.shape[1])

    columns = []
    role_counts = {
        "numeric": 0,
        "datetime": 0,
        "categorical": 0,
        "text": 0,
    }

    for i, col in enumerate(df.columns):
        if i >= max_cols:
            break

        s = df[col]

        # Handle edge case where column has all NaN values
        if s.isna().all():
            logger.info(f"Column {col} has all NaN values, defaulting to 'text' role")
            role = "text"
            provenance = "all_nan"
        else:
            role, provenance = _infer_role(s, uniqueness_cutoff, avg_length_cutoff)

        # Track role counts
        if role in role_counts:
            role_counts[role] += 1

        # Default: no stats
        stats = None
        top_categories = []

        # Numeric stats
        if role == "numeric" and s.notna().any():
            s_numeric = pd.to_numeric(s, errors='coerce')
            s_clean = s_numeric.dropna()
            if len(s_clean) > 0:
                # Compute quantiles in a single operation for efficiency
                quantiles = s_clean.quantile([0.25, 0.5, 0.75]).values
                stats = {
                    "min": float(s_clean.min()),
                    "max": float(s_clean.max()),
                    "mean": float(s_clean.mean()),
                    "std": float(s_clean.std()) if len(s_clean) > 1 else 0.0,
                    "sum": float(s_clean.sum()),
                    "variance": float(s_clean.var()) if len(s_clean) > 1 else 0.0,
                    "q25": float(quantiles[0]),
                    "q50": float(quantiles[1]),  # median
                    "q75": float(quantiles[2]),
                    "skew": float(s_clean.skew()) if len(s_clean) > 2 else 0.0,
                }
            else:
                logger.warning(f"Column {col} has no valid numeric values after cleaning")

        # Datetime stats (min/max as strings)
        elif role == "datetime" and s.notna().any():
            try:
                s_dt = pd.to_datetime(s, errors="coerce")
                s_dt = s_dt.dropna()
                if not s_dt.empty:
                    # Compute datetime quantiles if possible
                    stats = {
                        "min": s_dt.min().isoformat(),
                        "max": s_dt.max().isoformat(),
                        "mean": s_dt.mean().isoformat() if len(s_dt) > 0 else None,
                        "std": None,
                        "sum": None,
                    }
            except Exception as e:
                logger.warning(f"Error computing datetime stats for {col}: {e}")

        # Boolean stats
        elif role == "boolean":
            try:
                value_counts = s.value_counts(dropna=True)
                top_categories = [
                    {"value": str(idx), "count": int(cnt)}
                    for idx, cnt in value_counts.items()
                ]
            except Exception as e:
                logger.warning(f"Error computing boolean stats for {col}: {e}")

        # Ordinal stats
        elif role == "ordinal":
            try:
                value_counts = s.value_counts(dropna=True)
                top_categories = [
                    {"value": str(idx), "count": int(cnt), "order": list(value_counts.index).index(idx)}
                    for idx, cnt in value_counts.items()
                ]
                # Sort by the defined order
                top_categories.sort(key=lambda x: x["order"])
            except Exception as e:
                logger.warning(f"Error computing ordinal stats for {col}: {e}")

        # Geographic stats
        elif role == "geographic":
            try:
                value_counts = s.value_counts(dropna=True).head(5)  # More top values for geographic data
                top_categories = [
                    {"value": str(idx), "count": int(cnt)}
                    for idx, cnt in value_counts.items()
                ]
            except Exception as e:
                logger.warning(f"Error computing geographic stats for {col}: {e}")

        # Categorical stats: top 3 categories
        elif role == "categorical":
            try:
                value_counts = s.value_counts(dropna=True).head(3)
                top_categories = [
                    {"value": str(idx), "count": int(cnt)}
                    for idx, cnt in value_counts.items()
                ]
            except Exception as e:
                logger.warning(f"Error computing categorical stats for {col}: {e}")

        columns.append({
            "name": col,
            "dtype": str(s.dtype),
            "role": role,
            "missing_count": int(s.isna().sum()),
            "unique_count": int(s.nunique()),
            "stats": stats,
            "top_categories": top_categories,
            "provenance": provenance  # Add provenance info
        })

    logger.info(f"Dataset profile built for {len(columns)} out of {n_cols} total columns")
    logger.info(f"Role counts: {role_counts}")
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "role_counts": role_counts,
        "columns": columns,
    }
