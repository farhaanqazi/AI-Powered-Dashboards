"""Phase 8 (S8.1) — snapshot tests for compiled contract mappings.

The contract is the spine: a regression in role/aggregation/allow-list
classification silently corrupts every downstream chart and KPI. These tests
freeze the compiled (role, aggregation, allowed_aggregations, allowed_charts)
mapping for controlled golden datasets and assert it is *deterministic* and
*order-independent*.
"""
from __future__ import annotations

import pandas as pd

from src.analysis.layer_1_profiler import run_syntactic_profiling
from src.analysis.layer_2_classifier import run_semantic_classification
from src.contract import compile_contract, run_ingest_gate


def _compile(df: pd.DataFrame):
    res = run_ingest_gate(df)
    profiles = run_semantic_classification(run_syntactic_profiling(res.df), res.df)
    return compile_contract(res.df, profiles, res)


def _mapping(contract):
    # The meaningful contract surface: what charts/KPIs are allowed to do with
    # each field. The internal `aggregation` enum is intentionally excluded —
    # the allow-lists are what downstream rendering actually consumes.
    return {
        name: (
            fc.role,
            tuple(fc.allowed_aggregations),
            tuple(fc.allowed_charts),
        )
        for name, fc in contract.fields.items()
    }


def _sales_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [f"O{i}" for i in range(1, 13)],
            "region": ["N", "S"] * 6,
            "year": [2020, 2021] * 6,
            "revenue": [f"${v:,.2f}" for v in ([100.5, 250.0, 320.0] * 4)],
            "margin_pct": [round(0.1 * (i % 5), 3) for i in range(1, 13)],
        }
    )


# Frozen snapshot — a change here is a deliberate contract change, not a fix.
_SALES_SNAPSHOT = {
    "order_id": ("identifier", ("count", "nunique"), ()),
    "region": ("categorical", ("count", "nunique"), ("bar", "pie")),
    "year": ("year", ("min", "max", "range", "nunique"), ("line", "bar")),
    "revenue": (
        "numeric",
        ("sum", "mean", "min", "max", "median"),
        ("bar", "line", "histogram", "box", "scatter"),
    ),
    "margin_pct": (
        "ratio",
        ("mean", "median", "min", "max"),
        ("line", "bar", "box"),
    ),
}


def test_sales_contract_matches_frozen_snapshot():
    assert _mapping(_compile(_sales_df())) == _SALES_SNAPSHOT


def test_contract_mapping_is_order_independent():
    df = _sales_df()
    reversed_df = df[df.columns[::-1]].copy()
    a = _compile(df)
    b = _compile(reversed_df)
    assert _mapping(a) == _mapping(b)
    assert a.schema_fingerprint == b.schema_fingerprint


def test_contract_compile_is_deterministic():
    df = _sales_df()
    first = _compile(df)
    second = _compile(df)
    assert _mapping(first) == _mapping(second)
    assert first.schema_fingerprint == second.schema_fingerprint
    assert first.grain == second.grain


def test_aggregate_row_dataset_snapshot():
    df = pd.DataFrame(
        {"region": ["N", "S", "Total"], "revenue": [10.0, 20.0, 30.0]}
    )
    c = _compile(df)
    assert c.has_aggregate_rows and c.aggregate_row_count == 1
    assert c.fields["revenue"].aggregation == "additive"
