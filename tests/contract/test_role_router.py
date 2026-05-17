"""Phase 3 — role-aware router + L3/L4/EDA wiring."""
import pandas as pd

from src.analysis.data_structures import EnrichedProfile
from src.analysis.layer_1_profiler import run_syntactic_profiling
from src.analysis.layer_2_classifier import run_semantic_classification
from src.analysis.layer_3_relational import run_relational_analysis
from src.analysis.eda_analyzer import run_eda_analysis
from src.contract.models import FieldContract
from src.contract.role_router import (
    can_sum,
    collapse_to_grain,
    get_allowed_aggregations,
    is_correlatable,
    recompute_ratio,
)


def _prof(name, role, tags=None, std=1.0):
    return EnrichedProfile(
        name=name, dtype="float64", role=role, semantic_tags=tags or [],
        null_count=0, unique_count=50, stats={"std": std, "count": 100},
        top_categories=[],
    )


def test_identifier_and_year_not_correlatable():
    assert is_correlatable(_prof("sales", "numeric")) is True
    assert is_correlatable(_prof("customer_id", "identifier")) is False
    assert is_correlatable(_prof("fiscal_year", "numeric")) is False  # name=year
    assert is_correlatable(_prof("region", "categorical")) is False


def test_ratio_cannot_be_summed():
    ratio = _prof("conversion_rate", "numeric", tags=["rate"])
    assert can_sum(ratio) is False
    assert "sum" not in get_allowed_aggregations(ratio)
    additive = _prof("revenue", "numeric", tags=["additive"])
    assert can_sum(additive) is True
    assert "sum" in get_allowed_aggregations(additive)


def test_fieldcontract_allowed_aggs_passthrough():
    fc = FieldContract(
        name="r", dtype="float64", role="ratio",
        allowed_aggregations=("mean", "median"),
    )
    assert get_allowed_aggregations(fc) == ("mean", "median")


def test_recompute_ratio_uses_component_sums_not_mean_of_ratios():
    # Per-row ratios: 1/10, 9/10  -> mean would be 0.5 (wrong).
    # Correct total ratio = (1+9) / (10+10) = 0.5 here; use asymmetric data:
    num = pd.Series([1, 99])
    den = pd.Series([10, 100])
    # mean of ratios = (0.1 + 0.99)/2 = 0.545 ; correct = 100/110 = 0.909...
    assert abs(recompute_ratio(num, den) - (100 / 110)) < 1e-9
    assert recompute_ratio(5, 0) == 0.0


def test_collapse_to_grain_sums_additive_means_rate():
    df = pd.DataFrame({
        "region": ["N", "N", "S"],
        "revenue": [100.0, 50.0, 30.0],
        "win_rate": [0.4, 0.6, 0.5],
    })
    profiles = {
        "region": _prof("region", "categorical"),
        "revenue": _prof("revenue", "numeric", tags=["additive"]),
        "win_rate": _prof("win_rate", "numeric", tags=["rate"]),
    }
    out = collapse_to_grain(df, ["region"], profiles).set_index("region")
    assert out.loc["N", "revenue"] == 150.0          # summed
    assert abs(out.loc["N", "win_rate"] - 0.5) < 1e-9  # averaged


def test_layer3_excludes_identifier_numeric_from_correlation():
    # account_no is perfectly correlated with itself-derived col but is an ID.
    df = pd.DataFrame({
        "account_no": list(range(1000, 1100)),
        "spend": [x * 2.0 for x in range(100)],
        "visits": [x * 2.0 + 1 for x in range(100)],
    })
    enriched = run_semantic_classification(run_syntactic_profiling(df), df)
    # Force account_no to identifier (name + uniqueness already do this).
    insights = run_relational_analysis(df, enriched)
    for ins in insights:
        assert "account_no" not in ins.columns


def test_eda_year_column_reports_range_not_average():
    df = pd.DataFrame({
        "year": [2018, 2019, 2020, 2021, 2022] * 4,
        "revenue": [float(i) for i in range(20)],
    })
    enriched = run_semantic_classification(run_syntactic_profiling(df), df)
    eda = run_eda_analysis(df, enriched, [])
    inds = eda["key_indicators"]
    year_inds = [i for i in inds if "year" in i["indicator"].lower()]
    assert any(i["type"] == "range" for i in year_inds)
    assert not any(i["indicator"] == "Average year" for i in inds)
