"""Phase 9 (S9.3) — AI eval harness: golden datasets + classification cases.

This is a deterministic regression eval over the classification spine
(ingest gate → L1/L2 profiles → contract). The LLM is non-deterministic and
key-gated (absent under test), so the *gradeable* surface is the deterministic
role/aggregation/grain mapping that every AI narration is constrained by.

Each golden dataset declares the column roles we are confident about; the
harness grades the compiled contract against them and asserts a perfect score.
A drift (regression *or* improvement) fails loudly and must be re-baselined
deliberately — exactly the contract-stability guarantee Phase 8 established,
extended to whole datasets.

Known classifier quirks are intentionally NOT graded here (they are tracked as
xfails elsewhere): ISO-date columns of unique values classify as `identifier`
(not `datetime`), and an email column whose values strip to digits is coerced
numeric. Grading only the confident columns keeps the eval honest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.layer_1_profiler import run_syntactic_profiling
from src.analysis.layer_2_classifier import run_semantic_classification
from src.contract import compile_contract, run_ingest_gate


def _compile(df):
    res = run_ingest_gate(df)
    profiles = run_semantic_classification(run_syntactic_profiling(res.df), res.df)
    return res, compile_contract(res.df, profiles, res)


# --- golden datasets ------------------------------------------------------

def _sales():
    return pd.DataFrame({
        "order_id": [f"O{i}" for i in range(1, 25)],
        "region": ["N", "S", "E", "W"] * 6,
        "year": [2019, 2020, 2021, 2022] * 6,
        "revenue": [f"${v:,.2f}" for v in [100.0, 250.5, 320.0, 90.0] * 6],
        "discount_rate": [round(0.05 * (i % 5), 3) for i in range(24)],
    })


def _timeseries():
    return pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=30, freq="D").astype(str),
        "sales": np.arange(30) * 1.0 + 5,
        "units": np.arange(30) % 7,
    })


def _survey():
    return pd.DataFrame({
        "respondent_id": list(range(1, 21)),
        "satisfied": [True, False] * 10,
        "segment": ["A", "B", "C", "D"] * 5,
        "nps": [1, 2, 3, 4, 5] * 4,
    })


def _financial():
    return pd.DataFrame({
        "account": [f"AC{i:04d}" for i in range(1, 16)],
        "balance": [f"{v:,.2f}" for v in [1000.0 * i for i in range(1, 16)]],
        "currency": ["USD"] * 15,
        "fy": [2021, 2022, 2023] * 5,
    })


def _mixed_nulls():
    return pd.DataFrame({
        "city": ["NYC", "LA", "NA", "SF", "-", "CHI"] * 4,
        "pop": [8.0, 4.0, None, 1.0, None, 3.0] * 4,
        "grade": ["A", "B", "C", "A", "B", "C"] * 4,
    })


def _ratios():
    # headcount is intentionally non-unique/non-sequential so it reads as a
    # numeric measure, not an identifier (sequential unique ints classify as
    # identifier — correct behaviour, just not what this case grades).
    return pd.DataFrame({
        "team": ["alpha", "beta", "gamma"] * 8,
        "conversion_pct": [round(0.1 * (i % 9), 3) for i in range(24)],
        "headcount": [12, 7, 25, 12, 7, 25, 18, 7] * 3,
    })


# name -> (builder, {column: expected_role})
GOLDEN = {
    "sales": (_sales, {
        "order_id": "identifier",
        "region": "categorical",
        "year": "year",
        "revenue": "numeric",
        "discount_rate": "ratio",
    }),
    "timeseries": (_timeseries, {
        "sales": "numeric",
        "units": "numeric",
    }),
    "survey": (_survey, {
        "respondent_id": "identifier",
        "satisfied": "boolean",
        "nps": "numeric",
    }),
    "financial": (_financial, {
        "account": "identifier",
        "balance": "numeric",
        "currency": "categorical",
        "fy": "year",
    }),
    "mixed_nulls": (_mixed_nulls, {
        "pop": "numeric",
        "grade": "categorical",
    }),
    "ratios": (_ratios, {
        "team": "categorical",
        "conversion_pct": "ratio",
        "headcount": "numeric",
    }),
}


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_classification_is_perfect(name):
    builder, expected = GOLDEN[name]
    _res, contract = _compile(builder())

    graded = {
        col: contract.fields[col].role
        for col in expected
        if col in contract.fields
    }
    assert graded == expected, (
        f"[{name}] classification drift: got {graded}, expected {expected}"
    )


def test_eval_aggregate_score_is_100pct():
    """One headline number for the whole harness — must be a clean sweep."""
    total = correct = 0
    for builder, expected in GOLDEN.values():
        _res, contract = _compile(builder())
        for col, want in expected.items():
            total += 1
            if col in contract.fields and contract.fields[col].role == want:
                correct += 1
    assert total >= 18  # 6 golden datasets, 3+ graded columns each
    assert correct == total, f"eval score {correct}/{total} (expected perfect)"


def test_keyed_datasets_expose_an_identifier_grain():
    """Contract invariant: a row-keyed dataset resolves a grain on its id."""
    for name in ("sales", "survey", "financial"):
        builder, _ = GOLDEN[name]
        _res, contract = _compile(builder())
        ids = [n for n, f in contract.fields.items() if f.is_identifier]
        assert ids, f"[{name}] expected an identifier column"
        assert any(i in contract.grain for i in ids), (
            f"[{name}] grain {contract.grain} excludes identifier {ids}"
        )


def test_ratio_columns_cannot_be_summed():
    """Semantic guarantee: a rate/ratio never enters an additive total."""
    for name in ("sales", "ratios"):
        builder, expected = GOLDEN[name]
        _res, contract = _compile(builder())
        for col, role in expected.items():
            if role == "ratio":
                assert "sum" not in contract.fields[col].allowed_aggregations
