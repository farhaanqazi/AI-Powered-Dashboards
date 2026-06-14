"""Phase 15 S15.1 — supervised driver analysis + prediction quality.

Synthetic datasets with a *known* driver are fed through the real Layer-1/2
profilers so the assertions exercise the same path the pipeline uses. Numbers
are deterministic (seeded) — every run must agree. The module must never raise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.analysis.layer_1_profiler import run_syntactic_profiling
from src.analysis.layer_2_classifier import run_semantic_classification
from src.analysis.ml import compute_ml_insights


def _profiles(df):
    return run_semantic_classification(run_syntactic_profiling(df), df)


sklearn = pytest.importorskip("sklearn")


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """`sales` is driven mostly by `spend` (+) and a bit by `visits`; `noise`
    is irrelevant; `cust_id` is a high-cardinality identifier to be dropped."""
    rng = np.random.default_rng(0)
    n = 400
    spend = rng.uniform(0, 100, n)
    visits = rng.uniform(0, 50, n)
    noise = rng.normal(0, 1, n)
    sales = 5.0 * spend + 2.0 * visits + rng.normal(0, 5, n)
    return pd.DataFrame({
        "cust_id": np.arange(100000, 100000 + n),
        "spend": spend,
        "visits": visits,
        "noise": noise,
        "sales": sales,
    })


@pytest.fixture
def classification_df() -> pd.DataFrame:
    """`churn` (yes/no) is driven by `tenure` (lower → churn)."""
    rng = np.random.default_rng(1)
    n = 400
    tenure = rng.uniform(0, 60, n)
    logit = 3.0 - 0.1 * tenure
    prob = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.uniform(0, 1, n) < prob, "yes", "no")
    return pd.DataFrame({
        "tenure": tenure,
        "extra": rng.normal(0, 1, n),
        "churn": churn,
    })


def test_regression_picks_target_and_ranks_real_driver(regression_df):
    rep = compute_ml_insights(regression_df, _profiles(regression_df))

    assert rep["available"] is True
    assert rep["task"] == "regression"
    assert rep["target"] == "sales"
    # The identifier is never a feature.
    assert "cust_id" not in rep["numeric_features"]
    # Strong, learnable signal.
    assert rep["metrics"]["r2_mean"] > 0.8
    # The strongest driver is `spend`, and its effect is positive.
    assert rep["top_driver"] == "spend"
    top = next(i for i in rep["importances"] if i["feature"] == "spend")
    assert top.get("direction") == "positive"
    assert "sales" in rep["verdict"]


def test_regression_chart_is_renderable_bar(regression_df):
    chart = compute_ml_insights(regression_df, _profiles(regression_df))["chart"]
    assert chart["type"] == "bar"
    assert chart["data"]
    # Frontend ChartRenderer picks `category`/`value`.
    item = chart["data"][0]
    assert {"feature", "importance", "category", "value"} <= set(item)


def test_classification_target_and_baseline(classification_df):
    rep = compute_ml_insights(classification_df, _profiles(classification_df))
    assert rep["available"] is True
    assert rep["task"] == "classification"
    assert rep["target"] == "churn"
    m = rep["metrics"]
    assert 0.0 <= m["f1_mean"] <= 1.0
    assert "majority_baseline_accuracy" in m
    assert rep["top_driver"] == "tenure"


def test_determinism(regression_df):
    a = compute_ml_insights(regression_df, _profiles(regression_df))
    b = compute_ml_insights(regression_df, _profiles(regression_df))
    assert a["metrics"] == b["metrics"]
    assert a["importances"] == b["importances"]


def test_disabled_via_config(monkeypatch, regression_df):
    monkeypatch.setattr(config, "ML_INSIGHTS_ENABLED", False)
    rep = compute_ml_insights(regression_df, _profiles(regression_df))
    assert rep == {"available": False, "reason": "disabled"}


def test_empty_and_no_target_degrade_gracefully():
    assert compute_ml_insights(pd.DataFrame(), {})["available"] is False
    # Only an identifier column → no modellable target, no raise.
    ids = pd.DataFrame({"id": np.arange(100000, 100100)})
    rep = compute_ml_insights(ids, _profiles(ids))
    assert rep["available"] is False


def test_leaky_feature_is_dropped(regression_df):
    df = regression_df.copy()
    df["sales_copy"] = df["sales"] * 1.0  # perfect leak
    rep = compute_ml_insights(df, _profiles(df))
    assert rep["available"] is True
    assert "sales_copy" not in rep["numeric_features"]
    assert any("leakage" in n.lower() for n in rep["notes"])
