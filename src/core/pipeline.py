# src/core/pipeline.py

from src.data.parser import load_csv
from src.data.analyser import basic_profile, build_dataset_profile
from src.ml.kpi_generator import generate_basic_kpis
from src.ml.chart_selector import suggest_charts
from src.viz.plotly_renderer import build_category_count_charts


def build_dashboard_from_df(df, max_cols=None):
    """
    Core dashboard builder that works from an in-memory DataFrame.
    All data sources (upload, URL, Kaggle, etc.) should end up here.
    """
    if df is None:
        return None

    # 1) Determine max columns
    if max_cols is None:
        max_cols = df.shape[1]

    # 2) Build dataset profile
    dataset_profile = build_dataset_profile(df, max_cols=max_cols)

    # 3) Legacy/simple profile (optional)
    profile = basic_profile(df)

    # 4) KPIs
    kpis = generate_basic_kpis(df, dataset_profile)

    # 5) Chart suggestions (generic ChartSpec-like dicts)
    charts = suggest_charts(df, dataset_profile, kpis)

    # 6) Build multiple category_count charts and pick a primary one
    category_charts = build_category_count_charts(df, charts)
    primary_chart = next(iter(category_charts.values()), None)

    return {
        "df": df,
        "dataset_profile": dataset_profile,
        "profile": profile,
        "kpis": kpis,
        "charts": charts,
        "primary_chart": primary_chart,
        "category_charts": category_charts,
    }


def build_dashboard_from_file(file_storage, max_cols=None):
    """
    Orchestrates the full dashboard build from an uploaded file.
    Keeps the old interface for the upload flow.
    """
    df = load_csv(file_storage)
    if df is None:
        return None

    return build_dashboard_from_df(df, max_cols=max_cols)
