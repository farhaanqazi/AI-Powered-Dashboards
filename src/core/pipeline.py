import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import pandas as pd
import re
import time

from src import config
from src.diagnostics import tracer
from src.observability.metrics import pipeline_layer_seconds


@contextmanager
def _time_layer(layer_name: str):
    with pipeline_layer_seconds.labels(layer=layer_name).time():
        yield

# --- New 4-Layer Analysis Engine ---
from src.analysis.layer_1_profiler import run_syntactic_profiling
from src.analysis.layer_2_classifier import run_semantic_classification
from src.analysis.layer_3_relational import run_relational_analysis
from src.analysis.layer_4_interpreter import determine_kpis, select_charts
from src.analysis.eda_analyzer import run_eda_analysis
from src.analysis.llm_analyst import run_ai_analyst, arbitrate_column_roles

# --- Existing Visualization and Data Structures ---
from src.viz.plotly_renderer import build_charts_from_specs
from src.analysis.data_structures import DashboardState, EnrichedProfile

# --- Semantic Contract Layer (Phases 1-4) ---
from src.contract import (
    run_ingest_gate,
    critique,
    apply_vetoes,
    compile_contract,
    get_contract_cache,
)
from src.contract.dq_report import build_dq_report, evaluate_acceptance
from src.contract.df_cache import get_df_cache

logger = logging.getLogger(__name__)


def _empty_dashboard_state(original_filename, *, errors=None, warnings=None,
                           dataset_profile=None, eda_summary=None) -> DashboardState:
    """A no-charts DashboardState used for rejected / review-gated datasets."""
    return DashboardState(
        dataset_profile=dataset_profile or {},
        profile=[], kpis=[], charts=[], primary_chart=None,
        category_charts={}, all_charts=[],
        original_filename=original_filename,
        errors=errors or [], warnings=warnings or [],
        critical_totals={}, critical_full_dataset_aggregates={},
        eda_summary=eda_summary or {},
    )


def _contract_compile_stage(df, enriched_profiles, ingest):
    """Critique → apply vetoes → compile → cache (locked-hit aware).

    Returns (enriched_profiles, contract). A locked cache hit short-circuits
    recompilation (the caller then also skips the LLM)."""
    crit = critique(df, enriched_profiles)
    enriched_profiles = apply_vetoes(enriched_profiles, crit)
    contract = compile_contract(df, enriched_profiles, ingest)
    cache = get_contract_cache()
    if cache.is_locked_hit(contract.schema_fingerprint):
        cached = cache.get(contract.schema_fingerprint)
        if cached is not None:
            contract = cached
    else:
        # S6.3 auto-accept: high-confidence, non-PII, grain-bearing datasets
        # auto-lock (no human review needed). Sub-threshold/PII stay unlocked
        # → DataQualityReport.status drives the Phase 7 HITL flow.
        accepted, _ = evaluate_acceptance(contract)
        if accepted and not contract.locked:
            contract = contract.with_lock(bump_version=False)
        cache.put(contract)
    dq = build_dq_report(contract, ingest, crit)
    # Stash the cleaned frame (transient, fingerprint-keyed) so a later HITL
    # override can truly re-run L3→render. No-op when disabled.
    get_df_cache().put(contract.schema_fingerprint, df)
    return enriched_profiles, contract, crit, dq


def _contract_into_profile(viz_profile: dict, contract, ingest, crit, dq=None) -> None:
    """Thread the contract + data-quality verdict into the dataset profile so
    it persists in the existing DashboardRecord (no new storage backend)."""
    viz_profile["contract"] = contract.model_dump(mode="json")
    viz_profile["sensitivity"] = contract.sensitivity
    viz_profile["pii_blocked"] = contract.pii_blocked
    viz_profile["data_quality"] = {
        "rejected": ingest.rejected,
        "reject_reason": ingest.reject_reason,
        "pii_columns": ingest.pii_columns,
        "pii_scan_engine": ingest.pii_scan_engine,
        "cleaning": ingest.manifest.model_dump(mode="json"),
        "vetoes": [v.model_dump() for v in crit.vetoes],
        "flags": [f.model_dump() for f in crit.flags],
        # Mutable runtime state (NOT in the frozen contract): whether the user
        # has consented to AI on this PII-bearing dataset. The consent endpoint
        # flips this and re-runs the AI layer.
        "ai_consent": bool(getattr(dq, "ai_consent", False)) if dq is not None else False,
        "report": dq.model_dump(mode="json") if dq is not None else None,
    }

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

        # --- INGEST CONTRACT GATE (Phase 1) ---
        with _time_layer("ingest_gate"):
            ingest = run_ingest_gate(df)
        if ingest.rejected:
            logger.warning(f"Ingest gate rejected dataset: {ingest.reject_reason}")
            return _empty_dashboard_state(
                original_filename,
                errors=[ingest.reject_reason or "Dataset rejected by ingest gate."],
                warnings=ingest.warnings,
                dataset_profile={"data_quality": {
                    "rejected": True, "reject_reason": ingest.reject_reason}},
            )
        df = ingest.df
        ingest_warnings = list(ingest.warnings)

        # --- 4-LAYER ANALYSIS PIPELINE ---
        # Layer 1: Get raw facts
        try:
            with _time_layer("profiling"):
                syntactic_profiles = run_syntactic_profiling(df, max_cols=max_cols)
            tracer.record_custom_event(trace_id, "layer_1_complete", {"profiled_columns": list(syntactic_profiles.keys())})
        except Exception as e:
            msg = f"Layer 1 failed: {e}"
            errors.append(msg)
            logger.exception(msg)
            raise

        # Layer 2: Assign semantic meaning
        try:
            with _time_layer("classifying"):
                enriched_profiles = run_semantic_classification(syntactic_profiles, df)
                arbitrate_column_roles(enriched_profiles, df)
            tracer.record_custom_event(trace_id, "layer_2_complete", {"roles": {n: p.role for n, p in enriched_profiles.items()}})
        except Exception as e:
            msg = f"Layer 2 failed: {e}"
            errors.append(msg)
            logger.exception(msg)
            raise
        
        # --- CONTRACT COMPILE + CRITIC + CACHE (Phases 2-4) ---
        with _time_layer("contract"):
            enriched_profiles, contract, crit, dq = _contract_compile_stage(
                df, enriched_profiles, ingest
            )
        # schema_review gate (Phase 5 mechanism; criterion = Phase 6 S6.3
        # auto-accept, UI = Phase 7). Inert unless SCHEMA_REVIEW_ENABLED.
        if config.SCHEMA_REVIEW_ENABLED and dq.status != "ok":
            logger.info(f"Schema review gate engaged ({dq.status}); halting before L3.")
            viz = {"n_rows": len(df), "n_cols": len(enriched_profiles),
                   "columns": [p.__dict__ for p in enriched_profiles.values()]}
            _contract_into_profile(viz, contract, ingest, crit, dq)
            viz["status"] = "schema_review"
            return _empty_dashboard_state(
                original_filename, warnings=ingest_warnings,
                dataset_profile=viz,
                eda_summary={"status": "schema_review",
                             "data_quality_report": dq.model_dump(mode="json")},
            )

        # Layer 3: Find relationships
        try:
            with _time_layer("relating"):
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
            with _time_layer("eda"):
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
        with _time_layer("interpreting"):
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

            # PII model: the deterministic dashboard always builds (nothing
            # leaves the server). The AI layer is the only egress, so when PII
            # is present it is gated behind explicit user consent. No consent
            # exists at first build → skip AI, flag the dataset so the UI can
            # ask. The consent endpoint later re-runs AI on the cached frame.
            if ingest.pii_blocked:
                logger.info(
                    "PII detected; AI Insights gated until the user consents."
                )
                if isinstance(eda_summary, dict):
                    eda_summary["ai_consent_required"] = True
            else:
                ai = run_ai_analyst(
                    enriched_profiles, relational_insights, eda_summary,
                    fallback_kpis=kpis, fallback_specs=chart_specs,
                    contract=contract,
                )
                kpis = ai["kpis"]
                chart_specs = ai["chart_specs"]
                if isinstance(eda_summary, dict):
                    if ai["narrative"]:
                        eda_summary["ai_narrative"] = ai["narrative"]
                    if ai["key_indicators"]:
                        eda_summary["key_indicators"] = ai["key_indicators"]
                    if ai["use_cases"]:
                        eda_summary["use_cases"] = ai["use_cases"]
                    if ai["recommendations"]:
                        eda_summary["recommendations"] = ai["recommendations"]

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
        _contract_into_profile(dataset_profile_for_viz, contract, ingest, crit, dq)
        if isinstance(eda_summary, dict):
            eda_summary["data_quality_report"] = dq.model_dump(mode="json")

        try:
            with _time_layer("rendering"):
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
            warnings=ingest_warnings,
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


def build_dashboard_from_file_generator(
    file_storage,
    max_cols: Optional[int] = 50,
    original_filename: Optional[str] = None,
    encoding: Optional[str] = None,
):
    """
    Generator variant of build_dashboard_from_file that yields phase events
    for SSE-style streaming. Final event has phase='done' with the
    DashboardState attached.
    """
    from src.data.parser import load_csv_from_file

    yield {"phase": "reading", "message": "Reading CSV file...", "percent": 5}
    load_result = load_csv_from_file(file_storage, encoding=encoding)
    if not load_result.success or load_result.df is None:
        yield {
            "phase": "error",
            "message": f"Failed to load CSV: {load_result.detail}",
            "percent": 100,
        }
        return

    state = None
    for event in build_dashboard_from_df_generator(
        df=load_result.df, max_cols=max_cols, original_filename=original_filename
    ):
        if event.get("phase") == "done":
            state = event.get("state")
        else:
            yield event

    if state is None:
        yield {"phase": "error", "message": "Pipeline returned no state.", "percent": 100}
        return

    state.warnings = load_result.warnings or []
    yield {"phase": "done", "message": "Complete", "percent": 100, "state": state}


def build_dashboard_from_df_generator(
    df: pd.DataFrame,
    max_cols: Optional[int] = 50,
    original_filename: Optional[str] = "dataframe_input",
):
    """
    Generator variant of build_dashboard_from_df that yields phase events
    between each of the 4 analysis layers. Final event has phase='done'
    with the constructed DashboardState in event['state'].

    Each event: {"phase": str, "message": str, "percent": int}
    Final event also includes: {"state": DashboardState}
    Error event: {"phase": "error", "message": str, "percent": 100}
    """
    trace_id = tracer.record_initial_state(df, source_name=original_filename)
    state: Optional[DashboardState] = None
    errors: List[str] = []

    try:
        if df is None:
            yield {"phase": "error", "message": "Input DataFrame is None.", "percent": 100}
            return

        yield {"phase": "preparing", "message": "Reshaping data if needed...", "percent": 10}
        df = _reshape_if_wide_timeseries(df)
        if df.empty:
            empty_state = DashboardState(
                dataset_profile={}, profile=[], kpis=[], charts=[], primary_chart=None,
                category_charts={}, all_charts=[], critical_totals={},
                critical_full_dataset_aggregates={}, eda_summary={},
            )
            yield {"phase": "done", "message": "Empty dataset", "percent": 100, "state": empty_state}
            return

        yield {"phase": "ingest_gate", "message": "Screening & cleaning data...", "percent": 15}
        with _time_layer("ingest_gate"):
            ingest = run_ingest_gate(df)
        if ingest.rejected:
            rej = _empty_dashboard_state(
                original_filename,
                errors=[ingest.reject_reason or "Dataset rejected by ingest gate."],
                warnings=ingest.warnings,
                dataset_profile={"data_quality": {
                    "rejected": True, "reject_reason": ingest.reject_reason}},
            )
            yield {"phase": "done", "message": "Dataset rejected", "percent": 100, "state": rej}
            return
        df = ingest.df
        ingest_warnings = list(ingest.warnings)

        yield {"phase": "profiling", "message": "Profiling columns...", "percent": 20}
        with _time_layer("profiling"):
            syntactic_profiles = run_syntactic_profiling(df, max_cols=max_cols)
        tracer.record_custom_event(trace_id, "layer_1_complete",
                                   {"profiled_columns": list(syntactic_profiles.keys())})

        yield {"phase": "classifying", "message": "Classifying column roles...", "percent": 35}
        with _time_layer("classifying"):
            enriched_profiles = run_semantic_classification(syntactic_profiles, df)
            arbitrate_column_roles(enriched_profiles, df)
        tracer.record_custom_event(trace_id, "layer_2_complete",
                                   {"roles": {n: p.role for n, p in enriched_profiles.items()}})

        with _time_layer("contract"):
            enriched_profiles, contract, crit, dq = _contract_compile_stage(
                df, enriched_profiles, ingest
            )
        if config.SCHEMA_REVIEW_ENABLED and dq.status != "ok":
            viz = {"n_rows": len(df), "n_cols": len(enriched_profiles),
                   "columns": [p.__dict__ for p in enriched_profiles.values()]}
            _contract_into_profile(viz, contract, ingest, crit, dq)
            viz["status"] = "schema_review"
            sr = _empty_dashboard_state(
                original_filename, warnings=ingest_warnings,
                dataset_profile=viz,
                eda_summary={"status": "schema_review",
                             "data_quality_report": dq.model_dump(mode="json")},
            )
            yield {"phase": "done", "message": "Schema review required",
                   "percent": 100, "state": sr}
            return

        yield {"phase": "relating", "message": "Finding correlations and relationships...", "percent": 50}
        with _time_layer("relating"):
            relational_insights = run_relational_analysis(df, enriched_profiles)
        tracer.record_custom_event(trace_id, "layer_3_complete",
                                   {"insights_found": len(relational_insights)})

        dataset_grain = _detect_dataset_grain(enriched_profiles, df)

        yield {"phase": "eda", "message": "Running exploratory data analysis...", "percent": 65}
        try:
            with _time_layer("eda"):
                eda_summary = run_eda_analysis(df, enriched_profiles, relational_insights)
            eda_errors = eda_summary.get("errors") if isinstance(eda_summary, dict) else None
            if eda_errors:
                errors.extend([f"EDA: {err}" for err in eda_errors])
        except Exception as e:
            errors.append(f"EDA failed: {e}")
            logger.exception("EDA failure")
            eda_summary = {}

        yield {"phase": "kpis", "message": "Computing KPIs and selecting charts...", "percent": 80}
        with _time_layer("interpreting"):
            kpis = determine_kpis(enriched_profiles, relational_insights)
            tracer.record_kpi_generation(trace_id, kpis)
            chart_specs = select_charts(enriched_profiles, relational_insights)
            tracer.record_chart_selection(trace_id, chart_specs)

            if ingest.pii_blocked:
                logger.info(
                    "PII detected; AI Insights gated until the user consents."
                )
                if isinstance(eda_summary, dict):
                    eda_summary["ai_consent_required"] = True
            else:
                ai = run_ai_analyst(
                    enriched_profiles, relational_insights, eda_summary,
                    fallback_kpis=kpis, fallback_specs=chart_specs,
                    contract=contract,
                )
                kpis = ai["kpis"]
                chart_specs = ai["chart_specs"]
                if isinstance(eda_summary, dict):
                    if ai["narrative"]:
                        eda_summary["ai_narrative"] = ai["narrative"]
                    if ai["key_indicators"]:
                        eda_summary["key_indicators"] = ai["key_indicators"]
                    if ai["use_cases"]:
                        eda_summary["use_cases"] = ai["use_cases"]
                    if ai["recommendations"]:
                        eda_summary["recommendations"] = ai["recommendations"]

        yield {"phase": "rendering", "message": "Building charts...", "percent": 92}
        role_counts: Dict[str, int] = {}
        for profile in enriched_profiles.values():
            role_counts[profile.role] = role_counts.get(profile.role, 0) + 1

        dataset_profile_for_viz = {
            "n_rows": len(df),
            "n_cols": len(enriched_profiles),
            "role_counts": role_counts,
            "dataset_grain": dataset_grain,
            "columns": [p.__dict__ for p in enriched_profiles.values()],
        }
        _contract_into_profile(dataset_profile_for_viz, contract, ingest, crit, dq)
        if isinstance(eda_summary, dict):
            eda_summary["data_quality_report"] = dq.model_dump(mode="json")
        with _time_layer("rendering"):
            all_charts = build_charts_from_specs(df, chart_specs, dataset_profile=dataset_profile_for_viz)
        primary_chart = next((c for c in all_charts if c.get("type") == "bar"), None)

        state = DashboardState(
            dataset_profile=dataset_profile_for_viz,
            profile=[],
            kpis=kpis,
            charts=chart_specs,
            primary_chart=primary_chart,
            category_charts={},
            all_charts=all_charts,
            original_filename=original_filename,
            errors=errors,
            warnings=ingest_warnings,
            critical_totals={},
            critical_full_dataset_aggregates={},
            eda_summary=eda_summary,
        )
        yield {"phase": "done", "message": "Dashboard ready", "percent": 100, "state": state}

    except Exception as e:
        logger.exception("Pipeline (generator) failed")
        yield {"phase": "error", "message": f"Pipeline failed: {e}", "percent": 100}

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
                errors_to_log.append("Pipeline (generator) failed to return a state object.")
            tracer.record_pipeline_end(trace_id, status=status, errors=errors_to_log)
