"""Phase 16 — chart builder behaviour (S16.3 low-signal suppression + tagging)."""
from __future__ import annotations

import pandas as pd

from src.viz.utils import _build_category_count_data
from src.viz.plotly_renderer import build_charts_from_specs, _section_for_intent


def test_single_category_chart_is_suppressed():
    # One distinct value → a single bar conveys no comparison (S16.3).
    df = pd.DataFrame({"region": ["North"] * 50})
    assert _build_category_count_data(df, "region") is None


def test_multi_category_chart_is_built():
    df = pd.DataFrame({"region": ["North"] * 30 + ["South"] * 20 + ["East"] * 10})
    out = _build_category_count_data(df, "region")
    assert out is not None
    assert len(out["data"]) >= 2
    assert out["type"] == "category_count"


def test_charts_are_tagged_with_intent_and_section():
    df = pd.DataFrame({"region": ["N", "S", "E", "W", "N", "S", "E", "N"]})
    specs = [{"id": "c1", "intent": "category_count", "x_field": "region"}]
    charts = build_charts_from_specs(df, specs)
    assert charts, "expected a category_count chart"
    assert charts[0]["intent"] == "category_count"
    assert charts[0]["section"] == "Breakdowns"


def test_section_taxonomy():
    assert _section_for_intent("time_series") == "Trends"
    assert _section_for_intent("scatter") == "Relationships"
    assert _section_for_intent("histogram") == "Distributions"
    assert _section_for_intent("nonexistent") == "Other"
