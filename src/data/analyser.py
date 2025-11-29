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
    # 1) Numeric (with special handling for year-like integers)
    if is_numeric_dtype(series):
        s_nonnull = series.dropna()
        if not s_nonnull.empty:
            try:
                col_min = float(s_nonnull.min())
                col_max = float(s_nonnull.max())
            except Exception:
                col_min = None
                col_max = None

            name_lower = (series.name or "").lower()

            # Heuristic: treat as datetime if it looks like a year column
            if (
                col_min is not None and col_max is not None
                and 1900 <= col_min <= 2100
                and 1900 <= col_max <= 2100
                and any(keyword in name_lower for keyword in ["year", "yr"])
            ):
                return "datetime"

        # Otherwise, just numeric
        return "numeric"

    # 2) Native datetime dtype
    if is_datetime64_any_dtype(series):
        return "datetime"

    # 3) Try to detect datetime even if stored as object (e.g. "InvoiceDate")
    if series.dtype == "object":
        sample = series.dropna().astype(str).head(50)
        if not sample.empty:
            parsed = pd.to_datetime(
                sample,
                errors="coerce",
                dayfirst=False,
                infer_datetime_format=True,
            )
            non_null_ratio = parsed.notna().mean()
            if non_null_ratio > 0.7:
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
    
    n_rows = int(len(df))
    n_cols = int(df.shape[1])

    columns = []

    for i, col in enumerate(df.columns):
        if i >= max_cols:
            break

        s = df[col]
        role = _infer_role(s)

        # Default: no stats
        stats = None
        top_categories = []

        # Only compute stats for numeric columns with at least one non-NaN
        if role == "numeric" and s.notna().any():
            stats = {
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "sum": float(s.sum()),
            }
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
        "columns": columns,
    }
