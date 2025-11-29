# src/data/analyser.py

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype


def basic_profile(df, max_cols=10):
    """
    Simple profile per column:
    - column name
    - dtype
    - missing values
    - unique values
    - role: 'numeric' or 'non-numeric'
    - min, max, mean: only for numeric columns
    """
    profile = []
    for i, col in enumerate(df.columns):
        if i >= max_cols:
            break
        series = df[col]

        if is_numeric_dtype(series):
            role = "numeric"
            col_min = float(series.min()) if series.notna().any() else None
            col_max = float(series.max()) if series.notna().any() else None
            col_mean = float(series.mean()) if series.notna().any() else None
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
        })

    return profile


def _infer_role(series: pd.Series) -> str:
    """
    Determine role:
    - numeric
    - datetime
    - categorical
    - text
    """
    # 1) Numeric
    if is_numeric_dtype(series):
        # year-like numeric → datetime
        s_nonnull = series.dropna()
        if not s_nonnull.empty:
            try:
                col_min = float(s_nonnull.min())
                col_max = float(s_nonnull.max())
            except Exception:
                col_min = None
                col_max = None

            name_lower = (series.name or "").lower()
            if (
                col_min is not None and col_max is not None
                and 1900 <= col_min <= 2100
                and 1900 <= col_max <= 2100
                and any(keyword in name_lower for keyword in ["year", "yr"])
            ):
                return "datetime"

        return "numeric"

    # 2) Native datetime dtype
    if is_datetime64_any_dtype(series):
        return "datetime"

    # 3) Try to detect datetime even if stored as object
    if series.dtype == "object":
        sample = series.dropna().astype(str).head(50)
        if not sample.empty:
            parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
            if parsed.notna().mean() > 0.7:
                return "datetime"

    # ---- Distinguish categorical vs text ----
    n_rows = len(series)
    n_unique = series.nunique(dropna=True)
    unique_ratio = n_unique / n_rows if n_rows > 0 else 0

    if series.notna().any():
        avg_len = series.dropna().astype(str).str.len().mean()
    else:
        avg_len = 0

    # Long / high-cardinality strings → text
    if avg_len > 30 or unique_ratio > 0.5:
        return "text"

    # Otherwise categorical
    return "categorical"


def build_dataset_profile(df: pd.DataFrame, max_cols: int = 50):
    """
    Build a richer dataset profile used across the app.
    Returns a dict shaped like DatasetProfile.
    """
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
        role = _infer_role(s)

        # Track role counts
        if role in role_counts:
            role_counts[role] += 1

        # Default: no stats
        stats = None
        top_categories = []

        # Numeric stats
        if role == "numeric" and s.notna().any():
            stats = {
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "sum": float(s.sum()),
            }

        # Datetime stats (min/max as strings)
        if role == "datetime" and s.notna().any():
            s_dt = pd.to_datetime(s, errors="coerce")
            s_dt = s_dt.dropna()
            if not s_dt.empty:
                stats = {
                    "min": s_dt.min().isoformat(),
                    "max": s_dt.max().isoformat(),
                    "mean": None,
                    "std": None,
                    "sum": None,
                }

        # Categorical stats: top 3 categories
        if role == "categorical":
            value_counts = s.value_counts(dropna=True).head(3)
            top_categories = [
                {"value": str(idx), "count": int(cnt)}
                for idx, cnt in value_counts.items()
            ]

        columns.append({
            "name": col,
            "dtype": str(s.dtype),
            "role": role,
            "missing_count": int(s.isna().sum()),
            "unique_count": int(s.nunique()),
            "stats": stats,
            "top_categories": top_categories,
        })

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "role_counts": role_counts,
        "columns": columns,
    }
