import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import pandas as pd
import re
import time

from src import config
from src.diagnostics import tracer

# --- New 4-Layer Analysis Engine ---
from src.analysis.layer_1_profiler import run_syntactic_profiling
from src.analysis.layer_2_classifier import run_semantic_classification
from src.analysis.layer_3_relational import run_relational_analysis
from src.analysis.layer_4_interpreter import determine_kpis, select_charts
from src.analysis.eda_analyzer import run_eda_analysis

# --- Existing Visualization and Data Structures ---
from src.viz.plotly_renderer import build_charts_from_specs
from src.analysis.data_structures import DashboardState, EnrichedProfile

logger = logging.getLogger(__name__)

def _reshape_if_wide_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects if a DataFrame is in a wide time-series format (e.g., columns are years)
    and, if so, reshapes it into a long format.
    """
    year_pattern = re.compile(r'^(col_)?(19\d{2}|20\d{2})$')
    year_cols = [col for col in df.columns if year_pattern.fullmatch(str(col))]

    if len(year_cols) > 5 and len(year_cols) / len(df.columns) > 0.3:
        logger.info(f"Wide time-series format detected with {len(year_cols)} year-like columns. Reshaping data.")
        id_vars = [col for col in df.columns if col not in year_cols]
        if not id_vars:
            logger.warning("No identifying columns found for reshaping wide time-series. Aborting.")
            return df

        try:
            df_long = pd.melt(df, id_vars=id_vars, value_vars=year_cols, var_name='Year', value_name='Value')
            df_long['Year'] = df_long['Year'].astype(str).str.replace(r'^(col_)?', '', regex=True).astype(int)
            logger.info(f"Reshaped DataFrame from {df.shape} to {df_long.shape}")
            return df_long
        except Exception as e:
            logger.error(f"Failed to reshape wide time-series data: {e}. Proceeding with original format.")
            return df
    return df


def _detect_dataset_grain(enriched_profiles: Dict[str, EnrichedProfile], df: pd.DataFrame) -> str:
    """
    Detects whether the dataset represents event-level (transactional) or entity-level (descriptive) records.

    Event-level: Each row represents an event/transaction (e.g., sales, orders, clicks)
    Entity-level: Each row represents an entity (e.g., customers, products, cars)
    """
    # Count identifier and entity-like columns
    identifier_count = 0
    numeric_count = 0

    for col_name, profile in enriched_profiles.items():
        if profile.role == 'identifier':
            identifier_count += 1
        elif profile.role == 'numeric':
            numeric_count += 1

    # If we have many identifiers relative to other columns, it's likely entity-level
    # If we have many numeric transactional metrics, it's likely event-level
    total_cols = len(enriched_profiles)

    if total_cols == 0:
        return "unknown"

    identifier_ratio = identifier_count / total_cols
    numeric_ratio = numeric_count / total_cols

    # Heuristic: if more than 30% of columns are identifiers, likely entity-level
    # If more than 40% are numeric and we have transactional indicators, likely event-level
    if identifier_ratio > 0.3:
        return "entity-level"  # Each row describes a unique entity

    # Look for transactional indicators (datetime + numeric combinations)
    datetime_count = sum(1 for profile in enriched_profiles.values() if profile.role == 'datetime')
    if datetime_count > 0 and numeric_count > 0:
        return "event-level"  # Likely transactional data with timestamps

    # Default heuristic based on column composition
    if numeric_ratio > 0.4:
        return "event-level"  # Likely transactional/quantitative data
    else:
        return "entity-level"  # Likely descriptive entity data


def build_dashboard_from_df(df: pd.DataFrame, max_cols: Optional[int] = 50,
                           original_filename: Optional[str] = "dataframe_input") -> Optional[DashboardState]:
    """
    Core dashboard builder using the new 4-layer analysis engine.
    """
    trace_id = tracer.record_initial_state(df, source_name=original_filename)
    state = None
    errors: List[str] = []
    
    try:
        if df is None:
            msg = "Input DataFrame cannot be None"
            errors.append(msg)
            raise ValueError(msg)

        df = _reshape_if_wide_timeseries(df)
        if df.empty:
            logger.warning("DataFrame is empty after initial prep.")
            return DashboardState(
                dataset_profile={},
                profile=[],
                kpis=[],
                charts=[],
                primary_chart=None,
                category_charts={},
                all_charts=[],
                critical_totals={},
                critical_full_dataset_aggregates={},
                eda_summary={}
            )

        # --- 4-LAYER ANALYSIS PIPELINE ---
        # Layer 1: Get raw facts
        try:
            syntactic_profiles = run_syntactic_profiling(df, max_cols=max_cols)
            tracer.record_custom_event(trace_id, "layer_1_complete", {"profiled_columns": list(syntactic_profiles.keys())})
        except Exception as e:
            msg = f"Layer 1 failed: {e}"
            errors.append(msg)
            logger.exception(msg)
            raise

        # Layer 2: Assign semantic meaning
        try:
            enriched_profiles = run_semantic_classification(syntactic_profiles, df)
            tracer.record_custom_event(trace_id, "layer_2_complete", {"roles": {n: p.role for n, p in enriched_profiles.items()}})
        except Exception as e:
            msg = f"Layer 2 failed: {e}"
            errors.append(msg)
            logger.exception(msg)
            raise
        
        # Layer 3: Find relationships
        try:
            relational_insights = run_relational_analysis(df, enriched_profiles)
            tracer.record_custom_event(trace_id, "layer_3_complete", {"insights_found": len(relational_insights)})
        except Exception as e:
            msg = f"Layer 3 failed: {e}"
            errors.append(msg)
            logger.exception(msg)
            raise

        # Detect dataset grain (event-level vs entity-level)
        dataset_grain = _detect_dataset_grain(enriched_profiles, df)

        # Run EDA analysis to generate insights
        try:
            eda_summary = run_eda_analysis(df, enriched_profiles, relational_insights)
            eda_errors = eda_summary.get("errors") if isinstance(eda_summary, dict) else None
            if eda_errors:
                errors.extend([f"EDA: {err}" for err in eda_errors])
        except Exception as e:
            msg = f"EDA failed: {e}"
            errors.append(msg)
            logger.exception(msg)
            eda_summary = {}

        # Layer 4: Decide what to show
        try:
            kpis = determine_kpis(enriched_profiles, relational_insights)
            tracer.record_kpi_generation(trace_id, kpis)
        except Exception as e:
            msg = f"KPI generation failed: {e}"
            errors.append(msg)
            logger.exception(msg)
            raise

        try:
            chart_specs = select_charts(enriched_profiles, relational_insights)
            tracer.record_chart_selection(trace_id, chart_specs)
        except Exception as e:
            msg = f"Chart selection failed: {e}"
            errors.append(msg)
            logger.exception(msg)
            raise

        # --- Visualization Stage ---
        # The analysis output is now used to drive rendering.
        # Calculate role counts
        role_counts = {}
        for profile in enriched_profiles.values():
            role = profile.role
            role_counts[role] = role_counts.get(role, 0) + 1

        dataset_profile_for_viz = {
            "n_rows": len(df), "n_cols": len(enriched_profiles),
            "role_counts": role_counts,
            "dataset_grain": dataset_grain,  # Add dataset grain information
            "columns": [p.__dict__ for p in enriched_profiles.values()]
        }

        try:
            all_charts = build_charts_from_specs(df, chart_specs, dataset_profile=dataset_profile_for_viz)
        except Exception as e:
            msg = f"Chart rendering failed: {e}"
            errors.append(msg)
            logger.exception(msg)
            raise

        primary_chart = next((c for c in all_charts if c.get('type') == 'bar'), None)

        state = DashboardState(
            dataset_profile=dataset_profile_for_viz,
            profile=[], # Old basic profile is deprecated
            kpis=kpis,
            charts=chart_specs,
            primary_chart=primary_chart,
            category_charts={},
            all_charts=all_charts,
            original_filename=original_filename,
            errors=errors,
            critical_totals={},
            critical_full_dataset_aggregates={},
            eda_summary=eda_summary
        )
        return state

    except Exception as e:
        logger.exception("Critical error during dashboard generation pipeline")
        # Re-raise the exception to be caught by the API layer in main.py
        # This ensures that the FastAPI endpoints can return a proper 500 Internal Server Error
        # instead of a 200 OK with embedded errors.
        raise RuntimeError(f"Dashboard pipeline failed: {e}") from e

    finally:
        if trace_id:
            status = "SUCCESS" if state and not state.errors else "FAILURE"
            errors_to_log: List[str] = []
            if state and state.errors:
                errors_to_log = list(state.errors)
            if not state:
                status = "CRITICAL_FAILURE"
                if errors:
                    errors_to_log.extend(errors)
                errors_to_log.append("Pipeline failed to return a state object.")
            tracer.record_pipeline_end(trace_id, status=status, errors=errors_to_log)


def build_dashboard_from_file(
    file_storage,
    max_cols: Optional[int] = 50,
    original_filename: Optional[str] = None,
    encoding: Optional[str] = None,
) -> Optional[DashboardState]:
    """
    Orchestrates the full dashboard build from an uploaded file.
    """
    from src.data.parser import load_csv_from_file
    
    load_result = load_csv_from_file(file_storage, encoding=encoding)
    if not load_result.success or load_result.df is None:
        logger.error(f"Failed to load CSV from file: {load_result.detail}")
        return None

    state = build_dashboard_from_df(
        df=load_result.df,
        max_cols=max_cols,
        original_filename=original_filename
    )
    if state is not None:
        state.warnings = load_result.warnings or []
    return state
