"""Phase 15 S15.2 — unsupervised segmentation + anomaly detection.

Synthetic data with *known* structure (three well-separated numeric blobs;
injected gross outliers) fed through the real profilers. Deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.analysis.layer_1_profiler import run_syntactic_profiling
from src.analysis.layer_2_classifier import run_semantic_classification
from src.analysis.ml import compute_anomalies, compute_segments


def _profiles(df):
    return run_semantic_classification(run_syntactic_profiling(df), df)


pytest.importorskip("sklearn")


@pytest.fixture
def clustered_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    blobs = []
    for cx, cy in [(0, 0), (20, 20), (40, 0)]:
        blobs.append(np.column_stack([
            rng.normal(cx, 1.0, 120), rng.normal(cy, 1.0, 120),
            rng.normal(cx / 2, 1.0, 120),
        ]))
    pts = np.vstack(blobs)
    return pd.DataFrame({"feat_a": pts[:, 0], "feat_b": pts[:, 1], "feat_c": pts[:, 2]})


def test_segments_found_and_separated(clustered_df):
    rep = compute_segments(clustered_df, _profiles(clustered_df))
    assert rep["available"] is True
    assert rep["method"] == "KMeans"
    assert rep["k"] >= 2
    assert rep["silhouette"] > 0.4               # well-separated blobs
    assert len(rep["segments"]) == rep["k"]
    assert sum(s["size"] for s in rep["segments"]) == rep["n_rows_used"]
    # Each segment is described by distinguishing features.
    assert any(s["distinguishing"] for s in rep["segments"])


def test_segments_scatter_is_grouped(clustered_df):
    chart = compute_segments(clustered_df, _profiles(clustered_df)).get("chart")
    assert chart and chart["type"] == "scatter"
    item = chart["data"][0]
    assert {"x", "y", "group"} <= set(item)


def test_segments_determinism(clustered_df):
    a = compute_segments(clustered_df, _profiles(clustered_df))
    b = compute_segments(clustered_df, _profiles(clustered_df))
    assert a["k"] == b["k"] and a["silhouette"] == b["silhouette"]


def test_anomalies_flag_injected_outliers():
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "y": rng.normal(0, 1, n),
        "z": rng.normal(0, 1, n),
    })
    df.loc[:4, ["x", "y", "z"]] = 50.0  # gross outliers
    rep = compute_anomalies(df, _profiles(df))
    assert rep["available"] is True
    assert rep["method"] == "IsolationForest"
    assert rep["n_outliers"] > 0
    assert 0.0 < rep["fraction"] < 1.0
    assert rep["top_features"]


def test_unsupervised_degrade_gracefully():
    assert compute_segments(pd.DataFrame(), {})["available"] is False
    # One numeric column → cannot segment in 2-D feature space.
    one = pd.DataFrame({"v": np.arange(200.0)})
    assert compute_segments(one, _profiles(one))["available"] is False


def test_disabled_via_config(monkeypatch, clustered_df):
    monkeypatch.setattr(config, "ML_INSIGHTS_ENABLED", False)
    assert compute_segments(clustered_df, _profiles(clustered_df))["available"] is False
    assert compute_anomalies(clustered_df, _profiles(clustered_df))["available"] is False
