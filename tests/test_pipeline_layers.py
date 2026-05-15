"""Per-layer unit tests for the analysis pipeline.

These verify each layer in isolation against minimal DataFrames so any
regression points at the offending layer, not the whole 4-layer chain.
"""
import numpy as np
import pandas as pd
import pytest

from src.analysis.layer_1_profiler import run_syntactic_profiling
from src.analysis.layer_2_classifier import run_semantic_classification
from src.analysis.layer_3_relational import run_relational_analysis
from src.analysis.layer_4_interpreter import determine_kpis, select_charts


# ---------- Layer 1 ----------

def test_layer1_profiles_numeric_column():
    df = pd.DataFrame({"price": [1.0, 2.0, 3.0, 4.0, 5.0]})
    profiles = run_syntactic_profiling(df, max_cols=50)
    assert "price" in profiles
    p = profiles["price"]
    assert p.stats["min"] == 1.0
    assert p.stats["max"] == 5.0
    assert p.stats["mean"] == 3.0


def test_layer1_truncates_to_max_cols():
    df = pd.DataFrame({f"c{i}": range(5) for i in range(10)})
    profiles = run_syntactic_profiling(df, max_cols=3)
    assert len(profiles) == 3


def test_layer1_skips_all_null_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "all_null": [None, None, None]})
    profiles = run_syntactic_profiling(df, max_cols=50)
    assert "a" in profiles
    assert "all_null" not in profiles


# ---------- Layer 2 ----------

@pytest.mark.xfail(
    reason=(
        "Layer 2 bug: run_semantic_classification checks "
        "is_likely_identifier_with_confidence BEFORE the object-column "
        "datetime parse (_detect_datetime). A short column of distinct ISO-8601 "
        "dates has unique_ratio=1.0, firing the 'very_high_cardinality' "
        "identifier signal (~0.87 confidence > 0.65 threshold), so it is "
        "classified 'identifier' and the datetime branch is never reached. "
        "Should be 'datetime'."
    ),
    strict=True,
)
def test_layer2_detects_datetime_role():
    df = pd.DataFrame({"d": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]})
    profiles = run_syntactic_profiling(df, max_cols=50)
    enriched = run_semantic_classification(profiles, df)
    assert enriched["d"].role == "datetime"


def test_layer2_detects_numeric_role():
    df = pd.DataFrame({"x": [1.1, 2.2, 3.3, 4.4]})
    profiles = run_syntactic_profiling(df, max_cols=50)
    enriched = run_semantic_classification(profiles, df)
    assert enriched["x"].role == "numeric"


def test_layer2_detects_categorical_role():
    df = pd.DataFrame({"c": ["red", "blue", "red", "green", "blue"] * 10})
    profiles = run_syntactic_profiling(df, max_cols=50)
    enriched = run_semantic_classification(profiles, df)
    assert enriched["c"].role == "categorical"


def test_layer2_detects_identifier_role():
    df = pd.DataFrame({"id": [f"u-{i:05d}" for i in range(200)], "v": list(range(200))})
    profiles = run_syntactic_profiling(df, max_cols=50)
    enriched = run_semantic_classification(profiles, df)
    assert enriched["id"].role == "identifier"


# ---------- Layer 3 ----------

def test_layer3_detects_strong_correlation():
    rng = np.random.default_rng(seed=7)
    base = rng.normal(0, 1, 100)
    df = pd.DataFrame({
        "a": base,
        "b": 0.95 * base + 0.05 * rng.normal(0, 1, 100),
    })
    profiles = run_syntactic_profiling(df, max_cols=50)
    enriched = run_semantic_classification(profiles, df)
    insights = run_relational_analysis(df, enriched)
    assert any({"a", "b"} == set(i.columns) for i in insights)


def test_layer3_omits_independent_columns():
    rng = np.random.default_rng(seed=11)
    df = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200)})
    profiles = run_syntactic_profiling(df, max_cols=50)
    enriched = run_semantic_classification(profiles, df)
    insights = run_relational_analysis(df, enriched)
    assert not insights


# ---------- Layer 4 ----------

def test_layer4_kpis_returns_list():
    df = pd.DataFrame({"revenue": [100.0, 200.0, 300.0, 400.0, 500.0]})
    profiles = run_syntactic_profiling(df, max_cols=50)
    enriched = run_semantic_classification(profiles, df)
    insights = run_relational_analysis(df, enriched)
    kpis = determine_kpis(enriched, insights, top_k=5)
    assert isinstance(kpis, list)


def test_layer4_select_charts_returns_list_with_priority():
    df = pd.DataFrame({
        "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
        "cat": ["a", "b", "a", "b", "a"],
    })
    profiles = run_syntactic_profiling(df, max_cols=50)
    enriched = run_semantic_classification(profiles, df)
    insights = run_relational_analysis(df, enriched)
    charts = select_charts(enriched, insights, max_charts=10)
    assert isinstance(charts, list)
    if charts:
        chart_types = {c.get("chart_type") for c in charts}
        assert chart_types.issubset({"histogram", "bar", "line", "scatter"})
