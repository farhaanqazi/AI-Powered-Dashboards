"""Phase 14 S14.2 — structured interaction engine.

Invariants under test: no LLM in the path; contract guards hold for filters
and calculations; provenance is chained over the filter set; the engine never
raises; the LRU result cache is deterministic and bounded.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.analysis.ask import apply_filters, run_interaction
from src.analysis.ask.interact import (
    reset_interaction_cache_for_tests,
    run_interaction_cached,
)
from src.analysis.ask.tools import ToolError
from src.analysis.layer_1_profiler import run_syntactic_profiling
from src.analysis.layer_2_classifier import run_semantic_classification
from src.contract import compile_contract, run_ingest_gate


@pytest.fixture
def df_and_contract():
    df = pd.DataFrame({
        "region": ["N", "S", "E", "W"] * 9,
        "revenue": [f"${v:,.2f}" for v in [100.0, 250.0, 320.0, 90.0] * 9],
        "margin_pct": [round(0.1 * (i % 5), 3) for i in range(36)],
        "units": [3, 7, 12, 5] * 9,
    })
    res = run_ingest_gate(df)
    profiles = run_semantic_classification(
        run_syntactic_profiling(res.df), res.df
    )
    contract = compile_contract(res.df, profiles, res)
    return res.df, contract


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_interaction_cache_for_tests()
    yield
    reset_interaction_cache_for_tests()


# --- apply_filters --------------------------------------------------------

def test_filter_eq_narrows_rows(df_and_contract):
    df, c = df_and_contract
    out = apply_filters(df, c, [{"column": "region", "op": "eq",
                                 "value": "N"}])
    assert len(out) == 9 and set(out["region"]) == {"N"}


def test_filter_unknown_column_rejected(df_and_contract):
    df, c = df_and_contract
    with pytest.raises(ToolError):
        apply_filters(df, c, [{"column": "nope", "op": "eq", "value": 1}])


def test_filter_between_numeric(df_and_contract):
    df, c = df_and_contract
    out = apply_filters(df, c, [{"column": "units", "op": "between",
                                 "value": [6, 13]}])
    assert set(out["units"]) <= {7, 12}


def test_too_many_filters_rejected(df_and_contract, monkeypatch):
    df, c = df_and_contract
    monkeypatch.setattr(config, "INTERACT_MAX_FILTERS", 1)
    with pytest.raises(ToolError):
        apply_filters(df, c, [
            {"column": "region", "op": "eq", "value": "N"},
            {"column": "units", "op": "gt", "value": 0},
        ])


# --- run_interaction ------------------------------------------------------

def test_interaction_ok_with_filter_and_chained_provenance(df_and_contract):
    df, c = df_and_contract
    out = run_interaction(df, c, {
        "calculation": "aggregate",
        "params": {"group_by": "region", "value": "revenue", "agg": "sum"},
        "filters": [{"column": "units", "op": "gte", "value": 5}],
    })
    assert out["status"] == "ok"
    assert out["numbers_traceable"] is True
    assert out["provenance"]["verified"] is True
    assert out["provenance"]["filtered"] is True
    assert out["provenance"]["derived_token"].startswith("derived:")
    assert out["rows_after"] < len(df)


def test_interaction_unknown_calculation_is_error_not_raise(df_and_contract):
    df, c = df_and_contract
    out = run_interaction(df, c, {"calculation": "rm -rf", "filters": []})
    assert out["status"] == "error" and "available" in out


def test_interaction_contract_guard_does_not_raise(df_and_contract):
    df, c = df_and_contract
    out = run_interaction(df, c, {
        "calculation": "column_stat",
        "params": {"column": "margin_pct", "metric": "sum"},
    })
    assert out["status"] == "error"  # ratio cannot be summed — guarded


def test_interaction_empty_selection(df_and_contract):
    df, c = df_and_contract
    out = run_interaction(df, c, {
        "calculation": "top_categories",
        "params": {"column": "region", "n": 5},
        "filters": [{"column": "units", "op": "gt", "value": 9999}],
    })
    assert out["status"] == "empty" and out["rows_after"] == 0


# --- LRU cache ------------------------------------------------------------

def test_cache_hit_is_identical_and_flagged(df_and_contract):
    df, c = df_and_contract
    spec = {"calculation": "top_categories",
            "params": {"column": "region", "n": 4}, "filters": []}
    first = run_interaction_cached("fp1", df, c, spec)
    second = run_interaction_cached("fp1", df, c, spec)
    assert first["cached"] is False and second["cached"] is True
    assert first["result"] == second["result"]


def test_cache_key_is_filter_order_independent(df_and_contract):
    df, c = df_and_contract
    fa = [{"column": "region", "op": "eq", "value": "N"},
          {"column": "units", "op": "gte", "value": 3}]
    s1 = {"calculation": "column_stat",
          "params": {"column": "revenue", "metric": "mean"}, "filters": fa}
    s2 = {"calculation": "column_stat",
          "params": {"column": "revenue", "metric": "mean"},
          "filters": list(reversed(fa))}
    run_interaction_cached("fp", df, c, s1)
    assert run_interaction_cached("fp", df, c, s2)["cached"] is True


def test_cache_lru_eviction(df_and_contract, monkeypatch):
    df, c = df_and_contract
    monkeypatch.setattr(config, "INTERACT_RESULT_CACHE_MAX_ENTRIES", 1)
    run_interaction_cached("fp", df, c, {
        "calculation": "top_categories",
        "params": {"column": "region", "n": 2}, "filters": []})
    run_interaction_cached("fp", df, c, {
        "calculation": "top_categories",
        "params": {"column": "region", "n": 3}, "filters": []})
    # First spec evicted → recompute, so cached must be False again.
    again = run_interaction_cached("fp", df, c, {
        "calculation": "top_categories",
        "params": {"column": "region", "n": 2}, "filters": []})
    assert again["cached"] is False
