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
from src import config
from src.diagnostics import tracer

logger = logging.getLogger(__name__)

def validate_pipeline_contract(df: pd.DataFrame, dataset_profile: dict) -> bool:
    """
    Validate the pipeline contract between DataFrame and dataset_profile.
    Returns True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("df is not a pandas DataFrame")
        return False

    if not isinstance(dataset_profile, dict):
        logger.error("dataset_profile is not a dictionary")
        return False

    if 'columns' not in dataset_profile:
        logger.error("dataset_profile missing 'columns' key")
        return False

    if 'n_rows' not in dataset_profile or 'n_cols' not in dataset_profile:
        logger.error("dataset_profile missing 'n_rows' or 'n_cols' keys")
        return False

    # Check that the column names in profile match the DataFrame columns
    profile_col_names = {col.get('name') for col in dataset_profile.get('columns', []) if col.get('name')}
    df_col_names = set(df.columns)

    if profile_col_names != df_col_names:
        logger.error(f"Mismatch between profile columns {profile_col_names} and DataFrame columns {df_col_names}")
        return False

    # Check that row counts match
    if len(df) != dataset_profile.get('n_rows', -1):
        logger.error(f"Mismatch between DataFrame rows ({len(df)}) and profile rows ({dataset_profile.get('n_rows')})")
        return False

    # Check that column counts match
    if len(df.columns) != dataset_profile.get('n_cols', -1):
        logger.error(f"Mismatch between DataFrame columns ({len(df.columns)}) and profile columns ({dataset_profile.get('n_cols')})")
        return False

    return True


def validate_dashboard_state(state: 'DashboardState') -> List[str]:
    """
    Validate the DashboardState for consistency and potential issues.
    Returns a list of warning messages if any issues are found.
    """
    warnings = []

    # Check that KPIs reference columns that exist in the dataset profile
    if state.kpis:
        profile_col_names = {col['name'] for col in state.dataset_profile.get('columns', [])}
        for kpi in state.kpis:
            kpi_label = kpi.get('label', '')
            if kpi_label and kpi_label not in profile_col_names and kpi_label != 'correlation':
                warnings.append(f"KPI '{kpi_label}' references non-existent column")

    # Check that charts reference columns that exist in the dataset profile
    if state.charts:
        profile_col_names = {col['name'] for col in state.dataset_profile.get('columns', [])}
        for chart in state.charts:
            x_field = chart.get('x_field')
            y_field = chart.get('y_field')

            if x_field and x_field not in profile_col_names:
                warnings.append(f"Chart references non-existent x_field '{x_field}'")
            if y_field and y_field not in profile_col_names:
                warnings.append(f"Chart references non-existent y_field '{y_field}'")

    return warnings


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
    critical_aggregates: Optional[Dict[str, float]] = None
    critical_totals: Optional[Dict[str, float]] = None
    critical_full_dataset_aggregates: Optional[Dict[str, float]] = None
    errors: Optional[List[str]] = None

def build_dashboard_from_df(df: pd.DataFrame, max_cols: Optional[int] = None,
                           max_categories: int = 10, max_charts: int = 20,
                           kpi_thresholds: Optional[Dict[str, float]] = None,
                           original_filename: Optional[str] = "dataframe_input") -> Optional[DashboardState]:
    """
    Core dashboard builder that works from an-in-memory DataFrame.
    All data sources (upload, URL, Kaggle, etc.) should end up here.
    """
    trace_id = tracer.record_initial_state(df, source_name=original_filename)
    state = None
    try:
        if df is None:
            logger.error("Input DataFrame is None")
            raise ValueError("Input DataFrame cannot be None")

        # Ensure df is a pandas DataFrame
        if df is not None and not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df) if df else pd.DataFrame()

        if df.empty:
            logger.warning("Input DataFrame is empty")
            # Return a minimal state for empty data
            state = DashboardState(
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
                original_filename=original_filename,
                critical_aggregates={}
            )
            return state

        # Validate DataFrame structure
        if not isinstance(df, pd.DataFrame):
            logger.error("Input is not a valid DataFrame after conversion")
            raise ValueError("Input could not be converted to a valid DataFrame")

        # Memory usage monitoring
        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # Memory in MB
            logger.info(f"Initial memory usage: {initial_memory:.2f} MB")

            # Calculate estimated memory usage of DataFrame
            df_memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # Memory in MB
            logger.info(f"Estimated DataFrame memory usage: {df_memory_usage:.2f} MB")

            # Memory limit check (e.g., 500 MB)
            if df_memory_usage > config.MEMORY_LIMIT_MB:
                logger.warning(f"DataFrame memory usage ({df_memory_usage:.2f} MB) exceeds limit ({config.MEMORY_LIMIT_MB} MB), sampling to reduce memory usage")
                # Sample to reduce memory usage - sample about 10% or 50,000 rows, whichever is smaller
                target_rows = min(int(len(df) * 0.1), 50000, len(df))
                df = df.sample(n=target_rows, random_state=42).reset_index(drop=True)
                logger.info(f"Sampled DataFrame to {len(df)} rows to reduce memory usage")
        except ImportError:
            logger.warning("psutil not available for memory monitoring")
        except Exception as e:
            logger.warning(f"Memory monitoring error: {e}")

        # Additional validation: make sure columns are valid
        # Remove any columns with invalid names (None, NaN, etc.)
        valid_columns = []
        for col in df.columns:
            if pd.isna(col) or col is None:
                logger.warning(f"Found invalid column name: {col}, removing column")
                continue
            valid_columns.append(col)

        df = df[valid_columns] if valid_columns else df[[]]

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

        # --- Pre-sampling aggregation ---
        critical_aggregates = {}
        if 'revenue' in df.columns and pd.api.types.is_numeric_dtype(df['revenue']):
            critical_aggregates['total_revenue'] = df['revenue'].sum()
        if 'sales' in df.columns and pd.api.types.is_numeric_dtype(df['sales']):
            critical_aggregates['total_sales'] = df['sales'].sum()
        # ---------------------------------

        # --- Calculate critical totals for financial/quantitative columns ---
        critical_totals = {}
        financial_keywords = ['amount', 'revenue', 'price', 'total', 'cost', 'expense', 'profit', 'fee', 'charge', 'payment', 'income', 'value']
        quantity_keywords = ['qty', 'quantity', 'count', 'volume', 'size', 'number']

        for col in df.columns:
            col_lower = col.lower()

            # Check if column is numeric and matches financial keywords
            if pd.api.types.is_numeric_dtype(df[col]) and any(keyword in col_lower for keyword in financial_keywords):
                critical_totals[f"Total {col.replace('_', ' ').title()}"] = df[col].sum()

            # Check if column is numeric and matches quantity keywords
            if pd.api.types.is_numeric_dtype(df[col]) and any(keyword in col_lower for keyword in quantity_keywords):
                critical_totals[f"Total {col.replace('_', ' ').title()}"] = df[col].sum()
        # ---------------------------------

        # --- Calculate critical full-dataset aggregates before sampling ---
        critical_full_dataset_aggregates = {}
        # Identify numeric columns likely to represent totals based on semantic_tags in dataset_profile
        # This would typically be done after profile generation, but since we need it before sampling,
        # we'll use a heuristic approach based on column names and data types
        for col_name in df.columns:
            if pd.api.types.is_numeric_dtype(df[col_name]):
                col_lower = col_name.lower()
                # Check for monetary-related keywords
                if any(keyword in col_lower for keyword in ['amount', 'revenue', 'price', 'cost', 'expense', 'profit', 'fee', 'charge', 'payment', 'income', 'value', 'total']):
                    critical_full_dataset_aggregates[f"total_{col_name}"] = df[col_name].sum()
                # Check for quantity-related keywords
                elif any(keyword in col_lower for keyword in ['qty', 'quantity', 'count', 'volume', 'size', 'number']):
                    critical_full_dataset_aggregates[f"total_{col_name}"] = df[col_name].sum()
        # ---------------------------------

        # Cap rows and columns to prevent expensive processing
        if len(df) > config.MAX_ROWS:
            logger.warning(f"DataFrame has {len(df)} rows, sampling to {config.MAX_ROWS} for performance")
            df = df.sample(n=min(config.MAX_ROWS, len(df)), random_state=42).reset_index(drop=True)

        start_time = time.time()
        timing = {}
        errors = []

        # 1) Determine max columns
        if max_cols is None:
            max_cols = min(df.shape[1], config.MAX_COLS)

        logger.info(f"Building dashboard for DataFrame with {df.shape[0]} rows and {df.shape[1]} columns (using up to {max_cols})")

        # 2) Build dataset profile
        profile_start = time.time()
        try:
            dataset_profile = build_dataset_profile(df, max_cols=max_cols)
            if dataset_profile is None:
                raise ValueError("Dataset profile generation returned None")
            tracer.record_profiling_decision(trace_id, dataset_profile)
            logger.info(f"Dataset profile built with {dataset_profile['n_cols']} columns")
        except Exception as e:
            logger.exception("Error building dataset profile")
            raise
        timing['profile'] = time.time() - profile_start

        # 3) Validate pipeline contract immediately after profile generation
        try:
            if not validate_pipeline_contract(df, dataset_profile):
                raise ValueError("Pipeline contract validation failed after profile generation")
        except Exception as e:
            logger.error(f"Error during pipeline contract validation: {e}")
            raise

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
            kpis = generate_kpis(dataset_profile, eda_summary=eda_summary, top_k=10) # Limit KPIs for speed
            tracer.record_kpi_generation(trace_id, kpis)
            logger.info(f"Generated {len(kpis)} KPIs")
        except Exception as e:
            logger.exception("Error generating KPIs")
            kpis = []
            errors.append(f"KPI Generation Failed: {type(e).__name__}")
        timing['kpis'] = time.time() - kpi_start

        # 7) Chart suggestions (generic ChartSpec-like dicts)
        chart_start = time.time()
        try:
            charts = suggest_charts(df, dataset_profile, kpis)
            tracer.record_chart_selection(trace_id, charts)
            logger.info(f"Suggested {len(charts)} charts")
        except Exception as e:
            logger.exception("Error suggesting charts")
            charts = []
            errors.append(f"Chart Suggestion Failed: {type(e).__name__}")
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
            errors.append(f"Category Chart Building Failed: {type(e).__name__}")
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
            errors.append(f"Chart Generation Failed: {type(e).__name__}")
        timing['all_charts'] = time.time() - all_charts_start

        total_time = time.time() - start_time
        timing['total'] = total_time
        logger.info(f"Dashboard build completed in {total_time:.2f}s")
        logger.info(f"Timing breakdown: {timing}")

        # Validate the final state before returning
        # (Basic check - could be expanded)
        if not isinstance(dataset_profile, dict) or 'columns' not in dataset_profile:
             logger.error("Generated dataset_profile is invalid.")
             raise ValueError("Generated dataset_profile is invalid.")

        # Create the DashboardState
        state = DashboardState(
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
            original_filename=original_filename,
            critical_aggregates=critical_aggregates,
            critical_totals=critical_totals,
            critical_full_dataset_aggregates=critical_full_dataset_aggregates,
            errors=errors
        )

        # Validate DashboardState for inconsistencies
        try:
            state_warnings = validate_dashboard_state(state)
            if state_warnings:
                for warning in state_warnings:
                    logger.warning(f"DashboardState validation warning: {warning}")
        except Exception as e:
            logger.error(f"Error during DashboardState validation: {e}")

        return state
    finally:
        if trace_id:
            status = "SUCCESS" if state and not state.errors else "FAILURE"
            errors_to_log = state.errors if state and state.errors else []
            if not state:
                status = "CRITICAL_FAILURE"
                errors_to_log.append("Pipeline failed to return a state object.")
            tracer.record_pipeline_end(trace_id, status=status, errors=errors_to_log)


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
                                      kpi_thresholds=kpi_thresholds,
                                      original_filename=original_filename)

        if state:
            state.original_filename = original_filename

        total_time = time.time() - start_time
        logger.info(f"Dashboard built from file in {total_time:.2f}s")
        return state

    except Exception as e:
        logger.exception("Error in build_dashboard_from_file")
        return None