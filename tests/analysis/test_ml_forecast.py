"""Phase 15 S15.3 — Holt-Winters forecasting (statsmodels, never Prophet)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.analysis.layer_1_profiler import run_syntactic_profiling
from src.analysis.layer_2_classifier import run_semantic_classification
from src.analysis.ml import compute_forecast


def _profiles(df):
    return run_semantic_classification(run_syntactic_profiling(df), df)


pytest.importorskip("statsmodels")


@pytest.fixture
def timeseries_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 90
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    revenue = 100 + np.arange(n) * 2.0 + rng.normal(0, 3, n)  # clear upward trend
    return pd.DataFrame({"day": dates, "revenue": revenue})


def test_forecast_extends_history_with_band(timeseries_df):
    rep = compute_forecast(timeseries_df, _profiles(timeseries_df))
    assert rep["available"] is True
    assert rep["target"] == "revenue"
    assert rep["date_column"] == "day"
    assert len(rep["forecast"]) == config.ML_FORECAST_HORIZON
    for f in rep["forecast"]:
        assert f["lower"] <= f["yhat"] <= f["upper"]
    # Band widens with the horizon.
    width0 = rep["forecast"][0]["upper"] - rep["forecast"][0]["lower"]
    widthN = rep["forecast"][-1]["upper"] - rep["forecast"][-1]["lower"]
    assert widthN >= width0
    assert rep["chart"]["type"] == "time_series"


def test_forecast_requires_datetime_and_measure():
    df = pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0) * 2})
    assert compute_forecast(df, _profiles(df))["available"] is False


def test_forecast_too_short_history():
    dates = pd.date_range("2022-01-01", periods=10, freq="D")
    df = pd.DataFrame({"day": dates, "revenue": np.arange(10.0)})
    rep = compute_forecast(df, _profiles(df))
    assert rep["available"] is False
    assert rep["reason"] == "too-short-history"


def test_disabled_via_config(monkeypatch, timeseries_df):
    monkeypatch.setattr(config, "ML_INSIGHTS_ENABLED", False)
    assert compute_forecast(timeseries_df, _profiles(timeseries_df))["available"] is False
