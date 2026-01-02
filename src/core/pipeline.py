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

# --- Existing Visualization and Data Structures ---
from src.viz.plotly_renderer import build_charts_from_specs
from src.analysis.data_structures import DashboardState

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


def build_dashboard_from_df(df: pd.DataFrame, max_cols: Optional[int] = 50,
                           original_filename: Optional[str] = "dataframe_input") -> Optional[DashboardState]:
    """
    Core dashboard builder using the new 4-layer analysis engine.
    """
    trace_id = tracer.record_initial_state(df, source_name=original_filename)
    state = None
    
    try:
        if df is None: raise ValueError("Input DataFrame cannot be None")

        df = _reshape_if_wide_timeseries(df)
        if df.empty:
            logger.warning("DataFrame is empty after initial prep.")
            return DashboardState(
                df=df,
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
        syntactic_profiles = run_syntactic_profiling(df, max_cols=max_cols)
        tracer.record_custom_event(trace_id, "layer_1_complete", {"profiled_columns": list(syntactic_profiles.keys())})

        # Layer 2: Assign semantic meaning
        enriched_profiles = run_semantic_classification(syntactic_profiles, df)
        tracer.record_custom_event(trace_id, "layer_2_complete", {"roles": {n: p.role for n, p in enriched_profiles.items()}})
        
        # Layer 3: Find relationships
        relational_insights = run_relational_analysis(df, enriched_profiles)
        tracer.record_custom_event(trace_id, "layer_3_complete", {"insights_found": len(relational_insights)})

        # Layer 4: Decide what to show
        kpis = determine_kpis(enriched_profiles, relational_insights)
        tracer.record_kpi_generation(trace_id, kpis)
        
        chart_specs = select_charts(enriched_profiles, relational_insights)
        tracer.record_chart_selection(trace_id, chart_specs)
        
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
            "columns": [p.__dict__ for p in enriched_profiles.values()]
        }

        all_charts = build_charts_from_specs(df, chart_specs, dataset_profile=dataset_profile_for_viz)
        primary_chart = next((c for c in all_charts if c.get('type') == 'bar'), None)
        
        state = DashboardState(
            df=df,
            dataset_profile=dataset_profile_for_viz,
            profile=[], # Old basic profile is deprecated
            kpis=kpis,
            charts=chart_specs,
            primary_chart=primary_chart,
            category_charts={},
            all_charts=all_charts,
            original_filename=original_filename,
            errors=[],
            critical_totals={},
            critical_full_dataset_aggregates={},
            eda_summary={}
        )
        return state

    except Exception as e:
        logger.exception("Error during dashboard generation pipeline")
        errors = [f"{type(e).__name__}: {e}"]
        # Create a minimal state for error reporting
        if not state:
            state = DashboardState(
                df=df,
                dataset_profile={},
                profile=[],
                kpis=[],
                charts=[],
                primary_chart=None,
                category_charts={},
                all_charts=[],
                errors=errors,
                critical_totals={},
                critical_full_dataset_aggregates={},
                eda_summary={}
            )
        else:
            state.errors = errors
        return state

    finally:
        if trace_id:
            status = "SUCCESS" if state and not state.errors else "FAILURE"
            errors_to_log = state.errors if state and state.errors else []
            if not state:
                status = "CRITICAL_FAILURE"
                errors_to_log.append("Pipeline failed to return a state object.")
            tracer.record_pipeline_end(trace_id, status=status, errors=errors_to_log)


def build_dashboard_from_file(file_storage, max_cols: Optional[int] = 50,
                             original_filename: Optional[str] = None) -> Optional[DashboardState]:
    """
    Orchestrates the full dashboard build from an uploaded file.
    """
    from src.data.parser import load_csv_from_file
    
    load_result = load_csv_from_file(file_storage)
    if not load_result.success or load_result.df is None:
        logger.error(f"Failed to load CSV from file: {load_result.detail}")
        return None

    return build_dashboard_from_df(
        df=load_result.df,
        max_cols=max_cols,
        original_filename=original_filename
    )
