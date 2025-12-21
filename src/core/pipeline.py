import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import pandas as pd
from datetime import datetime
import time
# Corrected import paths based on the modular structure
from src.data.analyser import build_dataset_profile, basic_profile
from src.ml.kpi_generator import generate_kpis
from src.ml.chart_selector import suggest_charts
from src.viz.plotly_renderer import build_category_count_charts, build_charts_from_specs
from src.viz.simple_renderer import generate_all_chart_data # Potentially used as fallback or for specific needs
from src.eda.insights_generator import generate_eda_summary
from src.ml.correlation_engine import analyze_correlations, generate_correlation_insights

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Structured result with warnings and errors"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    errors: List[str] = None
    warnings: List[str] = None
    timing: Dict[str, float] = None

@dataclass
class DashboardState:
    """Structured return type for dashboard state"""
    df: pd.DataFrame
    dataset_profile: Dict[str, Any]
    profile: List[Dict[str, Any]]
    kpis: List[Dict[str, Any]]
    charts: List[Dict[str, Any]]
    primary_chart: Optional[Dict[str, Any]]
    category_charts: Dict[str, Any]
    all_charts: List[Dict[str, Any]]
    eda_summary: Optional[Dict[str, Any]] = None
    correlation_analysis: Optional[Dict[str, Any]] = None
    original_filename: Optional[str] = None

def build_dashboard_from_df(df: pd.DataFrame, max_cols: Optional[int] = None,
                           max_categories: int = 10, max_charts: int = 20,
                           kpi_thresholds: Optional[Dict[str, float]] = None) -> Optional[DashboardState]:
    """
    Core dashboard builder that works from an-in-memory DataFrame.
    All data sources (upload, URL, Kaggle, etc.) should end up here.
    """
    if df is None:
        logger.error("Input DataFrame is None")
        return None

    # Ensure df is a pandas DataFrame
    if df is not None and not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df) if df else pd.DataFrame()

    if df.empty:
        logger.warning("Input DataFrame is empty")
        # Return a minimal state for empty data
        return DashboardState(
            df=df,
            dataset_profile={"n_rows": 0, "n_cols": 0, "role_counts": {}, "columns": []},
            profile=[],
            kpis=[],
            charts=[],
            primary_chart=None,
            category_charts={},
            all_charts=[],
            eda_summary={"summary_statistics": {"total_rows": 0, "total_columns": 0}},
            correlation_analysis={"meaningful_correlations": [], "cross_type_relationships": [], "spurious_correlations": [], "summary_stats": {}},
            original_filename=None
        )

    # Validate DataFrame structure
    if not isinstance(df, pd.DataFrame):
        logger.error("Input is not a valid DataFrame after conversion")
        return None

    # Additional validation: make sure columns are valid
    # Remove any columns with invalid names (None, NaN, etc.)
    valid_columns = []
    for col in df.columns:
        if pd.isna(col) or col is None:
            logger.warning(f"Found invalid column name: {col}, removing column")
            continue
        valid_columns.append(col)

    df = df[valid_columns] if valid_columns else df[[]]  # Handle case where all columns are invalid

    # Handle potentially problematic column names that could cause issues downstream
    original_columns = df.columns.tolist()
    try:
        # Clean up column names to ensure they are valid identifiers
        clean_columns = []
        for col in df.columns:
            # Convert to string and remove problematic characters
            clean_col = str(col).strip()
            # Replace spaces, hyphens, and other problematic characters
            clean_col = clean_col.replace(' ', '_').replace('-', '_').replace('.', '_').replace('(', '').replace(')', '')
            # If it starts with a number, add a prefix
            if clean_col and clean_col[0].isdigit():
                clean_col = f"col_{clean_col}"
            clean_columns.append(clean_col)

        # Check for duplicates after cleaning and resolve them
        seen = set()
        final_columns = []
        for col in clean_columns:
            counter = 1
            unique_col = col
            while unique_col in seen:
                unique_col = f"{col}_{counter}"
                counter += 1
            seen.add(unique_col)
            final_columns.append(unique_col)

        df.columns = final_columns

        logger.info(f"Cleaned up column names from {original_columns} to {df.columns.tolist()}")
    except Exception as e:
        logger.warning(f"Error cleaning column names: {e}, using original names")
        pass  # Continue with original names if cleaning fails

    # Cap rows and columns to prevent expensive processing
    MAX_ROWS = 100000
    if len(df) > MAX_ROWS:
        logger.warning(f"DataFrame has {len(df)} rows, sampling to {MAX_ROWS} for performance")
        df = df.sample(n=min(MAX_ROWS, len(df)), random_state=42)

    start_time = time.time()
    timing = {}

    # 1) Determine max columns
    if max_cols is None:
        MAX_COLS = 50
        max_cols = min(df.shape[1], MAX_COLS)

    logger.info(f"Building dashboard for DataFrame with {df.shape[0]} rows and {df.shape[1]} columns (using up to {max_cols})")

    # 2) Build dataset profile
    profile_start = time.time()
    try:
        dataset_profile = build_dataset_profile(df, max_cols=max_cols)
        if dataset_profile is None:
            logger.error("Dataset profile generation failed")
            return None
        logger.info(f"Dataset profile built with {dataset_profile['n_cols']} columns")
    except Exception as e:
        logger.exception("Error building dataset profile")
        return None
    timing['profile'] = time.time() - profile_start

    # 3) Legacy/simple profile (optional)
    profile_start = time.time()
    try:
        profile = basic_profile(df, max_cols=min(10, max_cols)) # Limit basic profile columns for speed
        logger.info(f"Basic profile built for {len(profile)} columns")
    except Exception as e:
        logger.exception("Error building basic profile")
        profile = []
    timing['basic_profile'] = time.time() - profile_start

    # 4) Generate correlation insights using the new engine
    correlation_start = time.time()
    try:
        correlation_analysis = analyze_correlations(df, dataset_profile)
        # Get correlation insights from the analysis
        correlation_insights = generate_correlation_insights(correlation_analysis)
        logger.info(f"Correlation analysis completed with {len(correlation_insights)} insights generated")
    except Exception as e:
        logger.exception("Error generating correlation insights")
        correlation_analysis = None
        correlation_insights = []
    timing['correlation_analysis'] = time.time() - correlation_start

    # 5) Generate EDA summary (now incorporating correlation insights)
    eda_start = time.time()
    try:
        eda_summary = generate_eda_summary(df, dataset_profile, correlation_insights=correlation_insights)
        logger.info(f"EDA summary generated with {len(eda_summary.get('use_cases', []))} use cases and {len(eda_summary.get('key_indicators', []))} key indicators")
    except Exception as e:
        logger.exception("Error generating EDA summary")
        eda_summary = {}
    timing['eda_summary'] = time.time() - eda_start

    # 6) KPIs (now using EDA insights for better identification)
    kpi_start = time.time()
    try:
        kpis = generate_kpis(df, dataset_profile, eda_summary=eda_summary, top_k=10) # Limit KPIs for speed
        logger.info(f"Generated {len(kpis)} KPIs")
    except Exception as e:
        logger.exception("Error generating KPIs")
        kpis = []
    timing['kpis'] = time.time() - kpi_start

    # 7) Chart suggestions (generic ChartSpec-like dicts)
    chart_start = time.time()
    try:
        charts = suggest_charts(df, dataset_profile, kpis)
        logger.info(f"Suggested {len(charts)} charts")
    except Exception as e:
        logger.exception("Error suggesting charts")
        charts = []
    timing['charts'] = time.time() - chart_start

    # 8) Build multiple category_count charts with semantic awareness and pick a primary one
    category_start = time.time()
    try:
        category_charts = build_category_count_charts(
            df,
            charts,  # chart_specs are the suggestions from suggest_charts
            dataset_profile=dataset_profile,
            max_categories=max_categories,
            max_charts=max_charts
        )
        logger.info(f"Built {len(category_charts)} category count charts")
        primary_chart = next(iter(category_charts.values()), None) # Use .values() to get chart data
    except Exception as e:
        logger.exception("Error building category charts")
        category_charts = {}
        primary_chart = None
    timing['category_charts'] = time.time() - category_start

    # 9) Build all charts using the new intelligent renderer with semantic awareness
    all_charts_start = time.time()
    try:
        all_charts = build_charts_from_specs(
            df,
            charts,
            dataset_profile=dataset_profile,
            eda_summary=eda_summary,
            max_categories=max_categories,
            max_charts=max_charts
        )
        logger.info(f"Generated {len(all_charts)} all charts")
    except Exception as e:
        logger.exception("Error generating all charts")
        all_charts = []
    timing['all_charts'] = time.time() - all_charts_start

    total_time = time.time() - start_time
    timing['total'] = total_time
    logger.info(f"Dashboard build completed in {total_time:.2f}s")
    logger.info(f"Timing breakdown: {timing}")

    # Validate the final state before returning
    # (Basic check - could be expanded)
    if not isinstance(dataset_profile, dict) or 'columns' not in dataset_profile:
         logger.error("Generated dataset_profile is invalid.")
         return None

    return DashboardState(
        df=df,
        dataset_profile=dataset_profile,
        profile=profile,
        kpis=kpis,
        charts=charts,
        primary_chart=primary_chart,
        category_charts=category_charts,
        all_charts=all_charts,
        eda_summary=eda_summary,
        correlation_analysis=correlation_analysis,
        original_filename=None
    )


def build_dashboard_from_file(file_storage, max_cols: Optional[int] = None,
                             max_categories: int = 10, max_charts: int = 20,
                             kpi_thresholds: Optional[Dict[str, float]] = None,
                             original_filename: Optional[str] = None) -> Optional[DashboardState]:
    """
    Orchestrates the full dashboard build from an uploaded file.
    Keeps the old interface for the upload flow.
    """
    from src.data.parser import load_csv_from_file # Import here to avoid circular dependency if needed elsewhere
    start_time = time.time()
    try:
        load_result = load_csv_from_file(file_storage)
        if not load_result.success:
            logger.error(f"Failed to load CSV from file storage: {load_result.error_code} - {load_result.detail}")
            return None

        df = load_result.df
        if df is None:
            logger.error("Failed to load CSV from file storage (result was None)")
            return None

        state = build_dashboard_from_df(df, max_cols=max_cols,
                                      max_categories=max_categories,
                                      max_charts=max_charts,
                                      kpi_thresholds=kpi_thresholds)

        if state:
            state.original_filename = original_filename

        total_time = time.time() - start_time
        logger.info(f"Dashboard built from file in {total_time:.2f}s")
        return state

    except Exception as e:
        logger.exception("Error in build_dashboard_from_file")
        return None