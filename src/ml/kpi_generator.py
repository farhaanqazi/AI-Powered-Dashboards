def generate_basic_kpis(df, dataset_profile):
    """
    Returns a list of very simple KPIs.

    Dataset summary (Total Rows / Columns) is shown separately
    using dataset_profile, so we only add *extra* KPIs here.
    """
    kpis = []

    columns = dataset_profile["columns"]

    # 1. Numeric columns count
    numeric_cols = [col for col in columns if col["role"] == "numeric"]
    kpis.append({
        "label": "Numeric Columns",
        "value": len(numeric_cols),
        "format": "integer",
    })

    # 2. Non-numeric columns count
    non_numeric_cols = [col for col in columns if col["role"] != "numeric"]
    kpis.append({
        "label": "Non-Numeric Columns",
        "value": len(non_numeric_cols),
        "format": "integer",
    })

    return kpis
