"""Phase 15 S15.4 — what-if predict + re-segment (deterministic, no retrain)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.analysis.layer_1_profiler import run_syntactic_profiling
from src.analysis.layer_2_classifier import run_semantic_classification
from src.analysis.ml import compute_ml_insights
from src.analysis.ml import model_cache
from src.analysis.ml.predict import run_predict, run_resegment


def _profiles(df):
    return run_semantic_classification(run_syntactic_profiling(df), df)


pytest.importorskip("sklearn")
FP = "fp-test-123"


@pytest.fixture(autouse=True)
def _clear_cache():
    model_cache.reset_for_tests()
    yield
    model_cache.reset_for_tests()


@pytest.fixture
def regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    spend = rng.uniform(0, 100, n)
    visits = rng.uniform(0, 50, n)
    sales = 5.0 * spend + 2.0 * visits + rng.normal(0, 5, n)
    return pd.DataFrame({"spend": spend, "visits": visits, "sales": sales})


def _fit(df):
    return compute_ml_insights(df, _profiles(df), fingerprint=FP)


def test_predict_uses_cached_model_no_retrain(regression_df):
    rep = _fit(regression_df)
    assert rep["available"] is True
    out = run_predict(FP, {"spend": 80, "visits": 40})
    assert out["status"] == "ok"
    assert out["task"] == "regression"
    assert out["target"] == "sales"
    r = out["result"]
    assert r["lower"] <= r["prediction"] <= r["upper"]
    # High spend should predict high sales (driver is strongly positive).
    assert r["prediction"] > 300
    assert out["provenance"]["retrained"] is False


def test_predict_is_deterministic(regression_df):
    _fit(regression_df)
    a = run_predict(FP, {"spend": 50, "visits": 25})
    b = run_predict(FP, {"spend": 50, "visits": 25})
    assert a["result"] == b["result"]


def test_predict_higher_input_higher_output(regression_df):
    _fit(regression_df)
    low = run_predict(FP, {"spend": 10, "visits": 5})["result"]["prediction"]
    high = run_predict(FP, {"spend": 90, "visits": 45})["result"]["prediction"]
    assert high > low


def test_predict_missing_feature_uses_median(regression_df):
    _fit(regression_df)
    out = run_predict(FP, {"spend": 50})  # visits omitted → median
    assert out["status"] == "ok"
    assert "visits" not in out["inputs_used"]


def test_predict_cache_miss_is_unavailable(regression_df):
    _fit(regression_df)
    out = run_predict("no-such-fingerprint", {"spend": 50})
    assert out["status"] == "unavailable"
    assert "re-run" in out["error"].lower()


def test_predict_classification_returns_class(regression_df):
    rng = np.random.default_rng(1)
    n = 400
    tenure = rng.uniform(0, 60, n)
    churn = np.where(rng.uniform(0, 1, n) < 1 / (1 + np.exp(-(3 - 0.1 * tenure))), "yes", "no")
    df = pd.DataFrame({"tenure": tenure, "extra": rng.normal(0, 1, n), "churn": churn})
    rep = compute_ml_insights(df, _profiles(df), fingerprint=FP)
    assert rep["task"] == "classification"
    out = run_predict(FP, {"tenure": 5, "extra": 0})
    assert out["status"] == "ok"
    assert out["result"]["prediction"] in ("yes", "no")
    assert out["result"]["probability"] is None or 0.0 <= out["result"]["probability"] <= 1.0


def test_resegment_on_chosen_columns():
    rng = np.random.default_rng(0)
    blobs = [np.column_stack([rng.normal(c, 1, 120), rng.normal(c, 1, 120)]) for c in (0, 20, 40)]
    pts = np.vstack(blobs)
    df = pd.DataFrame({"a": pts[:, 0], "b": pts[:, 1], "c": rng.normal(0, 1, len(pts))})
    out = run_resegment(FP, df, ["a", "b"])
    assert out["status"] == "ok"
    assert out["segments"]["available"] is True
    assert out["segments"]["k"] >= 2
    assert out["provenance"]["retrained"] is False


def test_resegment_needs_two_columns():
    df = pd.DataFrame({"a": np.arange(100.0)})
    assert run_resegment(FP, df, ["a"])["status"] == "error"


def test_disabled_via_config(monkeypatch, regression_df):
    _fit(regression_df)
    monkeypatch.setattr(config, "ML_INSIGHTS_ENABLED", False)
    assert run_predict(FP, {"spend": 50})["status"] == "disabled"


# --- API routing through /api/interact -----------------------------------

def test_predict_api_requires_prior_analysis(client):
    r = client.post("/api/interact", json={"calculation": "predict", "params": {"features": {}}})
    assert r.status_code == 404


def test_predict_api_routes_after_upload(client):
    import io
    import time
    csv = b"region,revenue\nN,100\nS,250\nE,320\nW,90\nN,110\nS,260\n"
    sub = client.post("/api/jobs/upload",
                      files={"dataset": ("s.csv", io.BytesIO(csv), "text/csv")})
    job_id = sub.json()["job_id"]
    deadline = time.time() + 60
    while time.time() < deadline:
        if client.get(f"/api/jobs/{job_id}").json()["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(0.5)
    r = client.post("/api/interact",
                    json={"calculation": "predict", "params": {"features": {"revenue": 100}}})
    # Endpoint is wired: tiny data trains no model → a structured status, not 500.
    assert r.status_code == 200, r.text
    assert r.json()["status"] in ("ok", "unavailable", "disabled")
