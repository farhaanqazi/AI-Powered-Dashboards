from pandas.api.types import is_numeric_dtype


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
