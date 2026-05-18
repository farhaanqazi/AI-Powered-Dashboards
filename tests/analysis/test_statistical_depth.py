"""Phase 9 (S9.2) — deterministic statistical depth.

A synthetic dataset with *known* structure (a strong monotone pair, a
category that drives a numeric, a clear upward time trend, injected outliers)
is fed through the real profilers so the assertions exercise the same path the
pipeline uses. The numbers are deterministic — every run must agree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.analysis.layer_1_profiler import run_syntactic_profiling
from src.analysis.layer_2_classifier import run_semantic_classification
from src.analysis.statistical_depth import compute_statistical_depth


def _profiles(df):
    return run_semantic_classification(run_syntactic_profiling(df), df)


@pytest.fixture
def structured_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 240
    x = rng.normal(0, 1, n)
    y = 3.0 * x + rng.normal(0, 0.2, n)  # strong monotone dependence on x
    region = np.array(["north", "south", "east", "west"])[rng.integers(0, 4, n)]
    base = {"north": 10.0, "south": 50.0, "east": 90.0, "west": 130.0}
    score = np.array([base[r] for r in region]) + rng.normal(0, 1.0, n)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=n, freq="D"),
            "trend_val": np.arange(n) + rng.normal(0, 0.5, n),
            "x": x,
            "y": y,
            "region": region,
            "score": score,
        }
    )
    # Inject a handful of gross outliers for the anomaly detectors.
    df.loc[: 4, ["x", "y"]] = 999.0
    return df


def test_report_populated_and_structured(structured_df):
    rep = compute_statistical_depth(structured_df, _profiles(structured_df))

    assert rep["available"] is True
    assert rep["n_rows_sampled"] == len(structured_df)

    # Distributions: skew/kurtosis/normality present for numeric columns.
    dist = rep["distributions"]
    assert "x" in dist and {"skew", "kurtosis", "is_normal"} <= set(dist["x"])

    # Spearman: x~y is a near-perfect monotone pair.
    sp = rep["associations"]["spearman"]
    xy = next(d for d in sp if {d["a"], d["b"]} == {"x", "y"})
    assert abs(xy["rho"]) > 0.9

    # Correlation ratio η: region strongly explains score.
    eta = rep["associations"]["correlation_ratio_eta"]
    rs = next(d for d in eta if d["category"] == "region" and d["numeric"] == "score")
    assert rs["eta"] > 0.9

    # Trend: trend_val is monotonically increasing.
    assert rep["trend"]["trend_val"]["mann_kendall"]["trend"] == "increasing"

    # Anomalies: injected gross outliers are caught.
    assert rep["anomalies"]["isolation_forest"]["n_outliers"] >= 1

    # Clustering + drivers produced.
    assert "kmeans" in rep["clustering"]
    assert rep["drivers"]["target"] in rep["numeric_columns"]
    assert rep["drivers"]["importances"]


def test_is_deterministic(structured_df):
    p = _profiles(structured_df)
    a = compute_statistical_depth(structured_df, p)
    b = compute_statistical_depth(structured_df, p)
    assert a == b


def test_disabled_short_circuits(structured_df, monkeypatch):
    monkeypatch.setattr(config, "STATISTICAL_DEPTH_ENABLED", False)
    rep = compute_statistical_depth(structured_df, _profiles(structured_df))
    assert rep == {"available": False, "reason": "disabled"}


def test_empty_frame_is_safe():
    rep = compute_statistical_depth(pd.DataFrame(), {})
    assert rep["available"] is False


def test_degenerate_single_column_does_not_raise():
    df = pd.DataFrame({"only": [1, 1, 1, 1, 1, 1, 1, 1]})
    rep = compute_statistical_depth(df, _profiles(df))
    # No pairs / no matrix → empty sections, never an exception.
    assert rep["available"] is True
    assert rep["associations"]["spearman"] == []
    assert rep["anomalies"] == {}
