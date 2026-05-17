"""Single source of truth for turning a pipeline ``DashboardState`` into the
wire payload persisted + returned by every upload path (sync, SSE, job).

Previously this dict was duplicated in three endpoints in ``main.py``; the
job-queue path adds a fourth caller, so it is centralised here.
"""
from __future__ import annotations

from typing import Any, Dict


def state_to_payload(state: Any, original_filename: str) -> Dict[str, Any]:
    """Assemble the canonical dashboard payload from a pipeline state."""
    return {
        "dataset_profile": state.dataset_profile,
        "kpis": state.kpis,
        "charts": state.charts,
        "primary_chart": state.primary_chart,
        "category_charts": getattr(state, "category_charts", {}),
        "all_charts": state.all_charts,
        "original_filename": original_filename,
        "errors": getattr(state, "errors", []),
        "warnings": getattr(state, "warnings", []),
        "critical_totals": getattr(state, "critical_totals", {}),
        "critical_full_dataset_aggregates": getattr(
            state, "critical_full_dataset_aggregates", {}
        ),
        "eda_summary": getattr(state, "eda_summary", {}),
    }
