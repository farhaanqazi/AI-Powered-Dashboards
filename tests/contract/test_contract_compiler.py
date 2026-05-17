"""Phase 2 — contract compiler + frozen models + cache."""
import pandas as pd
import pytest
from pydantic import ValidationError

from src.analysis.layer_1_profiler import run_syntactic_profiling
from src.analysis.layer_2_classifier import run_semantic_classification
from src.contract import (
    DatasetContract,
    FieldContract,
    compile_contract,
    run_ingest_gate,
    schema_fingerprint,
)
from src.contract.cache import ContractCache


def _profiles(df):
    return run_semantic_classification(run_syntactic_profiling(df), df)


def _sales_df():
    return pd.DataFrame(
        {
            "order_id": [f"O{i}" for i in range(1, 13)],
            "region": ["N", "S"] * 6,
            "year": [2020, 2021] * 6,
            # Non-unique + currency-formatted so it coerces to a numeric
            # *measure* (a perfectly-unique column trips identifier detection —
            # that's the Phase 4 invariant-critic case, out of scope here).
            "revenue": [f"${v:,.2f}" for v in ([100.5, 250.0, 320.0] * 4)],
            "margin_pct": [round(0.1 * (i % 5), 3) for i in range(1, 13)],
        }
    )


def test_compile_produces_frozen_contract():
    df = _sales_df()
    res = run_ingest_gate(df)
    contract = compile_contract(res.df, _profiles(res.df), res)
    assert isinstance(contract, DatasetContract)
    with pytest.raises(ValidationError):
        contract.fields = {}  # frozen


def test_field_contract_is_frozen():
    fc = FieldContract(name="x", dtype="int64", role="numeric")
    with pytest.raises(ValidationError):
        fc.role = "text"


def test_fingerprint_is_stable_and_order_independent():
    df = _sales_df()
    p = _profiles(df)
    fp1 = schema_fingerprint(df, p)
    fp2 = schema_fingerprint(df[df.columns[::-1]], _profiles(df[df.columns[::-1]]))
    assert fp1 == fp2 and len(fp1) == 64


def test_additive_revenue_can_sum_ratio_cannot():
    df = _sales_df()
    res = run_ingest_gate(df)
    c = compile_contract(res.df, _profiles(res.df), res)
    rev = c.fields["revenue"]
    assert rev.aggregation == "additive"
    assert "sum" in rev.allowed_aggregations
    margin = c.fields["margin_pct"]
    assert "sum" not in margin.allowed_aggregations


def test_identifier_has_no_charts_and_grain_detected():
    df = _sales_df()
    res = run_ingest_gate(df)
    c = compile_contract(res.df, _profiles(res.df), res)
    assert c.fields["order_id"].is_identifier
    assert c.fields["order_id"].allowed_charts == ()
    # order_id is unique per row -> it is the grain
    assert "order_id" in c.grain


def test_aggregate_rows_flagged():
    df = pd.DataFrame(
        {"region": ["N", "S", "Total"], "revenue": [10.0, 20.0, 30.0]}
    )
    res = run_ingest_gate(df)
    c = compile_contract(res.df, _profiles(res.df), res)
    assert c.has_aggregate_rows
    assert c.aggregate_row_count == 1


def test_lock_bumps_version_without_mutating_original():
    c = DatasetContract(schema_fingerprint="abc")
    locked = c.with_lock()
    assert c.locked is False and c.version == 1
    assert locked.locked is True and locked.version == 2


def test_contract_cache_roundtrip_and_locked_hit():
    cache = ContractCache(client=None)
    c = DatasetContract(schema_fingerprint="fp1", locked=False)
    cache.put(c)
    assert cache.get("fp1").schema_fingerprint == "fp1"
    assert cache.is_locked_hit("fp1") is False
    cache.put(c.with_lock())
    assert cache.is_locked_hit("fp1") is True
    assert cache.get("missing") is None


def test_classifier_emits_confidence_and_alternatives():
    df = _sales_df()
    profs = _profiles(df)
    rev = profs["revenue"]
    assert 0.0 < rev.confidence <= 1.0
    assert len(rev.alternatives) <= 2
    assert all("role" in a and "confidence" in a for a in rev.alternatives)
