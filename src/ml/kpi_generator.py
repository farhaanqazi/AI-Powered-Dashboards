def generate_basic_kpis(df, dataset_profile):
    """
    Returns a list of very simple KPIs.

    Dataset summary (Total Rows / Columns) is shown separately
    using dataset_profile, so we only add extra KPIs here.
    """
    kpis = []

    columns = dataset_profile["columns"]

    # Count by role
    numeric_cols = [col for col in columns if col["role"] == "numeric"]
    datetime_cols = [col for col in columns if col["role"] == "datetime"]
    categorical_cols = [col for col in columns if col["role"] == "categorical"]
    text_cols = [col for col in columns if col["role"] == "text"]

    # 1. Numeric columns count
    kpis.append({
        "label": "Numeric Columns",
        "value": len(numeric_cols),
        "format": "integer",
    })

    # 2. Datetime columns count
    kpis.append({
        "label": "Datetime Columns",
        "value": len(datetime_cols),
        "format": "integer",
    })

    # 3. Categorical columns count
    kpis.append({
        "label": "Categorical Columns",
        "value": len(categorical_cols),
        "format": "integer",
    })

    # 4. Text columns count
    kpis.append({
        "label": "Text Columns",
        "value": len(text_cols),
        "format": "integer",
    })

    return kpis
