"""Phase 8 (S8.1) — backward-compatibility load tests.

Records persisted *before* the Semantic Contract Layer have no ``contract`` /
``data_quality`` keys. The architectural invariant is that the contract
serializes into the existing ``DashboardRecord`` with **no new storage
backend**, so a legacy payload must still load, and a contract must survive a
JSON round-trip unchanged.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.analysis.layer_1_profiler import run_syntactic_profiling
from src.analysis.layer_2_classifier import run_semantic_classification
from src.contract import DatasetContract, compile_contract, run_ingest_gate
from src.persistence import db
from src.persistence.repository import DashboardRepository


@pytest.fixture
def repo(tmp_path):
    # Own isolated SQLite file + disposed engine so the Windows session
    # teardown can unlink the shared test DB (no leaked file handle).
    engine = db.make_engine(f"sqlite:///{tmp_path / 'compat.db'}")
    db.init_db(engine)
    yield DashboardRepository(db.make_session_factory(engine))
    engine.dispose()

# A payload shaped the way the API stored it before Phases 1–8: no contract,
# no data_quality, no provenance tokens.
_LEGACY_PAYLOAD = {
    "trace_id": "legacy-trace-001",
    "kpis": [{"label": "Total Revenue", "value": 12345.0}],
    "charts": [{"intent": "category_count", "x_field": "region"}],
    "dataset_profile": {"n_rows": 100, "n_cols": 4},
    "eda_summary": {"notes": "pre-contract analysis"},
}


def test_legacy_payload_round_trips_through_repository(repo):
    repo.save("sess-legacy", trace_id="legacy-trace-001", payload=_LEGACY_PAYLOAD)
    got = repo.get("sess-legacy")
    assert got == _LEGACY_PAYLOAD
    # Absence of the contract keys must not raise anywhere.
    assert "contract" not in got
    assert "data_quality" not in got


def test_contract_survives_json_round_trip():
    df = pd.DataFrame(
        {
            "order_id": [f"O{i}" for i in range(1, 9)],
            "region": ["N", "S"] * 4,
            "revenue": [f"${v:,.2f}" for v in ([100.0, 250.0] * 4)],
        }
    )
    res = run_ingest_gate(df)
    profiles = run_semantic_classification(run_syntactic_profiling(res.df), res.df)
    contract = compile_contract(res.df, profiles, res)

    # Serialize into a DashboardRecord-style payload and back.
    serialized = json.loads(json.dumps(contract.model_dump()))
    restored = DatasetContract.model_validate(serialized)

    assert restored == contract
    assert restored.schema_fingerprint == contract.schema_fingerprint
    assert restored.fields.keys() == contract.fields.keys()


def test_contract_forward_compat_ignores_unknown_top_level_payload_keys(repo):
    """A future writer may add sibling keys to the payload; older readers that
    only pull the contract sub-dict must keep working."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    res = run_ingest_gate(df)
    profiles = run_semantic_classification(run_syntactic_profiling(res.df), res.df)
    contract = compile_contract(res.df, profiles, res)

    payload = {
        "contract": contract.model_dump(),
        "kpis": [],
        "charts": [],
        "some_future_key_v2": {"unrelated": True},
    }
    repo.save("sess-fwd", trace_id="fwd-1", payload=json.loads(json.dumps(payload)))
    got = repo.get("sess-fwd")

    rebuilt = DatasetContract.model_validate(got["contract"])
    assert rebuilt == contract
    assert got["some_future_key_v2"] == {"unrelated": True}
