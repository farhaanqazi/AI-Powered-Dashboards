"""DashboardRepository save/get behaviour."""
import pytest

from src.persistence import db
from src.persistence.repository import DashboardRepository


@pytest.fixture
def repo(tmp_path):
    engine = db.make_engine(f"sqlite:///{tmp_path / 'repo.db'}")
    db.init_db(engine)
    session_factory = db.make_session_factory(engine)
    yield DashboardRepository(session_factory)
    engine.dispose()


def _payload(name="a.csv"):
    return {
        "dataset_profile": {"n_rows": 3, "n_cols": 2},
        "kpis": [{"name": "x", "score": 0.9}],
        "charts": [],
        "primary_chart": None,
        "category_charts": {},
        "all_charts": [],
        "original_filename": name,
        "errors": [],
        "warnings": [],
        "critical_totals": {},
        "critical_full_dataset_aggregates": {},
        "eda_summary": {},
    }


def test_get_missing_returns_none(repo):
    assert repo.get("guest:nope") is None


def test_save_then_get_round_trips(repo):
    repo.save("guest:s1", trace_id="t1", payload=_payload("f1.csv"))
    got = repo.get("guest:s1")
    assert got is not None
    assert got["original_filename"] == "f1.csv"
    assert got["kpis"][0]["name"] == "x"


def test_save_is_upsert_keyed_by_session(repo):
    repo.save("guest:s1", trace_id="t1", payload=_payload("first.csv"))
    repo.save("guest:s1", trace_id="t2", payload=_payload("second.csv"))
    got = repo.get("guest:s1")
    assert got["original_filename"] == "second.csv"
    assert repo.count() == 1


def test_sessions_are_isolated(repo):
    repo.save("user:alice", trace_id="ta", payload=_payload("alice.csv"))
    repo.save("guest:bob", trace_id="tb", payload=_payload("bob.csv"))
    assert repo.get("user:alice")["original_filename"] == "alice.csv"
    assert repo.get("guest:bob")["original_filename"] == "bob.csv"
    assert repo.count() == 2
