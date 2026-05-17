"""Phase 7 — deterministic dashboard re-render after a HITL override.

Given the cached cleaned DataFrame and the (now locked, post-override)
contract, re-run the deterministic tail: L1 stats → roles taken FROM THE
CONTRACT (not re-classified) → L3 relations → L4 KPIs/charts → render. The
LLM is never called and semantic classification is not re-run — the human's
corrected roles are authoritative.

Returns the payload fragments to overwrite; callers merge these onto the
existing dashboard payload. On any failure the caller keeps the contract-only
result (graceful degradation).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from src.analysis.data_structures import EnrichedProfile
from src.analysis.layer_1_profiler import run_syntactic_profiling
from src.analysis.layer_3_relational import run_relational_analysis
from src.analysis.layer_4_interpreter import determine_kpis, select_charts
from src.analysis.eda_analyzer import run_eda_analysis
from src.viz.plotly_renderer import build_charts_from_specs
from src.contract.models import DatasetContract

logger = logging.getLogger(__name__)


def _tags_for(fc) -> List[str]:
    tags: List[str] = []
    if fc.aggregation == "additive":
        tags.append("additive")
    elif fc.aggregation == "rate":
        tags.append("rate")
    if fc.domain == "monetary":
        tags.append("monetary")
    return tags


def _profiles_from_contract(
    df: pd.DataFrame, contract: DatasetContract
) -> Dict[str, EnrichedProfile]:
    syntactic = run_syntactic_profiling(df)
    enriched: Dict[str, EnrichedProfile] = {}
    for name, sp in syntactic.items():
        fc = contract.fields.get(name)
        role = fc.role if fc is not None else "unknown"
        enriched[name] = EnrichedProfile(
            role=role,
            confidence=fc.confidence if fc is not None else 0.0,
            alternatives=[],
            semantic_tags=_tags_for(fc) if fc is not None else [],
            **sp.__dict__,
        )
    return enriched


def rebuild_dashboard(df: pd.DataFrame, contract: DatasetContract) -> Dict[str, Any]:
    """Recompute KPIs/charts/EDA from the cleaned df + corrected contract.

    Raises on failure so the caller can fall back; never calls the LLM.
    """
    profiles = _profiles_from_contract(df, contract)
    relational = run_relational_analysis(df, profiles)
    kpis = determine_kpis(profiles, relational)
    specs = select_charts(profiles, relational)

    role_counts: Dict[str, int] = {}
    for p in profiles.values():
        role_counts[p.role] = role_counts.get(p.role, 0) + 1
    viz_profile = {
        "n_rows": len(df),
        "n_cols": len(profiles),
        "role_counts": role_counts,
        "columns": [p.__dict__ for p in profiles.values()],
    }
    all_charts = build_charts_from_specs(df, specs, dataset_profile=viz_profile)
    primary = next((c for c in all_charts if c.get("type") == "bar"), None)

    try:
        eda = run_eda_analysis(df, profiles, relational)
    except Exception:  # EDA is decorative here; never fail the override on it
        eda = {}

    return {
        "kpis": kpis,
        "charts": specs,
        "all_charts": all_charts,
        "primary_chart": primary,
        "eda_summary": eda,
        "columns": viz_profile["columns"],
        "role_counts": role_counts,
    }
