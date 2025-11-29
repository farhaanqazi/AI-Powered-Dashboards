from src.data.parser import load_csv
from src.data.analyser import basic_profile, build_dataset_profile
from src.ml.kpi_generator import generate_basic_kpis


def build_dashboard_from_file(file_storage, max_cols=None):
    """
    Orchestrates the full dashboard build from an uploaded file.

    Returns a dict with:
    - df               : pandas DataFrame
    - dataset_profile  : DatasetProfile-shaped dict
    - profile          : basic column profile (legacy/simple)
    - kpis             : list of KPI dicts
    """
    # 1) Load CSV
    df = load_csv(file_storage)
    if df is None:
        return None

    # 2) Decide max_cols for profiling
    if max_cols is None:
        max_cols = df.shape[1]

    # 3) Build dataset-level profile
    dataset_profile = build_dataset_profile(df, max_cols=max_cols)

    # 4) Build legacy/basic profile (still available if needed)
    profile = basic_profile(df)

    # 5) Generate KPIs based on DatasetProfile
    kpis = generate_basic_kpis(df, dataset_profile)

    # 6) Return a simple DashboardState-like dict
    return {
        "df": df,
        "dataset_profile": dataset_profile,
        "profile": profile,
        "kpis": kpis,
    }
