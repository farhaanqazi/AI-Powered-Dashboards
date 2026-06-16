"""Document B authority knob — additive, flag-gated.

Asserts the dial behavior at the acceptance point and, critically, that the
flag-off path is IDENTICAL to level 2 (== production default), so adding the
knob does not move the frozen path. The LLM-proposes / engine-owns-values
invariant is checked via the provenance tokens on every emitted figure.
"""
from __future__ import annotations

import types

import pytest

from src import config
from src.analysis import llm_analyst
from src.analysis.llm_analyst import run_ai_analyst


def _profile(role, **stats):
    return types.SimpleNamespace(role=role, stats=stats, top_categories=[], name="x")


PROFILES = {
    "spend": _profile("numeric", mean=10.0, std=2.0, sum=100.0),
    "day": _profile("datetime"),
    "revenue": _profile("numeric", mean=50.0, std=5.0, sum=500.0),
    "headcount": _profile("numeric", mean=5.0, std=1.0, sum=50.0),
}

PARSED = {
    "kpis": [{"label": "Avg spend", "column": "spend", "metric": "mean"}],
    "charts": [{"intent": "time_series", "x_field": "day", "y_field": "spend",
                "agg_func": "sum", "title": "Spend over time"}],
    "narrative": "n", "key_indicators": [], "use_cases": [], "recommendations": [],
    "derived": [
        {"op": "ratio", "numerator": "revenue", "denominator": "headcount",
         "label": "Revenue per headcount"},
        {"op": "growth_rate", "numerator": "revenue", "label": "Revenue growth"},
    ],
}

FALLBACK_KPIS = [{"label": "Corr: spend & revenue", "value": "0.50", "type": "correlation", "score": 0.5}]
FALLBACK_SPECS = [{"intent": "category_count", "x_field": "spend"}]


class _FakeProvider:
    name = "fake"
    def available(self):
        return True
    def complete_json(self, **kw):
        return dict(PARSED)


@pytest.fixture(autouse=True)
def _wire(monkeypatch):
    monkeypatch.setattr(config, "AI_ANALYST_ENABLED", True)
    monkeypatch.setattr(llm_analyst, "_ground_truth", lambda *a, **k: {})
    monkeypatch.setattr("src.analysis.llm.get_llm_provider", lambda: _FakeProvider())
    monkeypatch.delenv("BENCHMARK_AUTHORITY_LEVEL", raising=False)
    yield


def _run():
    return run_ai_analyst(PROFILES, [], {}, fallback_kpis=FALLBACK_KPIS,
                          fallback_specs=FALLBACK_SPECS)


def test_flag_off_equals_level_2(monkeypatch):
    off = _run()
    monkeypatch.setenv("BENCHMARK_AUTHORITY_LEVEL", "2")
    lvl2 = _run()
    assert off == lvl2  # adding the knob does not move the default path


def test_off_produces_ai_selection_with_tokens():
    out = _run()
    # The LLM-proposed KPI is present and the engine computed + tokened its value.
    kpi = next(k for k in out["kpis"] if k["label"] == "Avg spend")
    assert kpi["value"] == "10.00"
    assert kpi["provenance"]["token"] == "L1.spend.mean"


def test_level_0_is_deterministic_only(monkeypatch):
    monkeypatch.setenv("BENCHMARK_AUTHORITY_LEVEL", "0")
    out = _run()
    assert out["kpis"] == FALLBACK_KPIS
    assert out["chart_specs"] == FALLBACK_SPECS


def test_level_1_llm_chart_types_det_kpis_no_agg(monkeypatch):
    monkeypatch.setenv("BENCHMARK_AUTHORITY_LEVEL", "1")
    out = _run()
    assert out["kpis"] == FALLBACK_KPIS                 # KPIs deterministic
    assert len(out["chart_specs"]) == 1                 # LLM selection, no backfill
    assert out["chart_specs"][0]["intent"] == "time_series"
    assert out["chart_specs"][0]["agg_func"] == "mean"  # LLM agg ("sum") stripped


def test_level_3_adds_engine_computed_derived(monkeypatch):
    monkeypatch.setenv("BENCHMARK_AUTHORITY_LEVEL", "3")
    out = _run()
    derived = next(k for k in out["kpis"] if k["label"] == "Revenue per headcount")
    # Engine owns the value: sum(revenue)/sum(headcount) = 500/50 = 10.00, tokened.
    assert derived["value"] == "10.00"
    assert derived["provenance"]["token"] == "L4.ratio.revenue|headcount"


def test_level_3_logs_unreducible_proposal_as_result(monkeypatch):
    """A derived proposal the engine can't reduce to a recipe is recorded as the
    authority/fence boundary — never displayed as an (untraceable) value."""
    monkeypatch.setenv("BENCHMARK_AUTHORITY_LEVEL", "3")
    out = _run()
    # The growth_rate proposal is not an executable recipe -> not a KPI...
    assert all(k["label"] != "Revenue growth" for k in out["kpis"])
    # ...but it is on record as a boundary result.
    boundary = out["authority_unreducible"]
    assert any(u["op"] == "growth_rate" and u["label"] == "Revenue growth" for u in boundary)
