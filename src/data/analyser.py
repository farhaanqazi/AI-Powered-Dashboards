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
    Simple role detection matching our project’s idea of roles.
    """
    if is_numeric_dtype(series):
        return "numeric"
    if is_datetime64_any_dtype(series):
        return "datetime"
   

# 3) Try to detect datetime even if stored as object (e.g. "InvoiceDate")
    if series.dtype == "object":
        # Take a small sample of non-null values
        sample = series.dropna().astype(str).head(50)

        if not sample.empty:
            parsed = pd.to_datetime(sample, errors="coerce", dayfirst=False, infer_datetime_format=True)
            # If a good proportion of the sample parses as dates, treat as datetime
            non_null_ratio = parsed.notna().mean()
            if non_null_ratio > 0.7:
                return "datetime"

    # 4) Fallback: treat everything else as categorical for now
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
