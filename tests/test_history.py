"""Phase 10 S10.4 — per-owner analysis history (persist / list / reopen).

Covers the ownership scope (org > user > guest — Clerk multi-tenancy), the
append-only repository contract, and the list/reopen API including
cross-owner isolation.
"""
from __future__ import annotations

import io
import time

import pytest

from src.auth import owner_key, sign_guest_session_id as _sgn
from src.persistence import db
from src.persistence.repository import DashboardRepository


# --- ownership scope ------------------------------------------------------

def test_owner_key_precedence():
    assert owner_key({"guest": True, "session_key": "guest:g1"}) == "guest:g1"
    assert owner_key({"sub": "user_123"}) == "user:user_123"
    # Org token: members of an org share history.
    assert owner_key({"sub": "user_123", "org_id": "org_abc"}) == "org:org_abc"


# --- repository contract --------------------------------------------------

@pytest.fixture
def repo(tmp_path):
    engine = db.make_engine(f"sqlite:///{tmp_path / 'hist.db'}")
    db.init_db(engine)
    yield DashboardRepository(db.make_session_factory(engine))
    engine.dispose()


def _payload(name, **extra):
    return {"original_filename": name, "kpis": [], "charts": [], **extra}


def test_history_is_append_only_and_owner_scoped(repo):
    repo.record_history("org:A", session_key="user:1", trace_id="t1",
                        payload=_payload("a.csv"))
    time.sleep(0.01)
    repo.record_history("org:A", session_key="user:2", trace_id="t2",
                        payload=_payload("b.csv"))
    repo.record_history("user:other", session_key="user:9", trace_id="t9",
                        payload=_payload("z.csv"))

    items = repo.list_history("org:A")
    assert [i["trace_id"] for i in items] == ["t2", "t1"]  # newest first
    assert all("payload" not in i for i in items)  # summaries only

    # Cross-owner isolation.
    other = repo.list_history("user:other")
    assert [i["trace_id"] for i in other] == ["t9"]
    assert other[0]["original_filename"] == "z.csv"

    # Reopen by trace_id, owner-checked.
    assert repo.get_history("org:A", "t1")["original_filename"] == "a.csv"
    assert repo.get_history("org:A", "t9") is None  # not this owner
    assert repo.get_history("user:other", "t9")["original_filename"] == "z.csv"


def test_expired_history_is_purged(repo):
    repo._ttl = -1  # everything written is already expired
    repo.record_history("u", session_key="s", trace_id="t",
                        payload=_payload("x.csv"))
    assert repo.list_history("u") == []
    assert repo.get_history("u", "t") is None
    repo.purge_expired()


# --- API: list + reopen ---------------------------------------------------

def _wait_done(client, job_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.5)
    raise AssertionError("job did not finish")


def _csv_files():
    return {"dataset": ("hist.csv",
                        io.BytesIO(b"region,revenue\nN,10\nS,20\nE,30\n"),
                        "text/csv")}


def test_history_api_list_and_reopen(client):
    sub = client.post("/api/jobs/upload", files=_csv_files())
    job = sub.json()
    final = _wait_done(client, job["job_id"])
    assert final["status"] == "done"

    listed = client.get("/api/history")
    assert listed.status_code == 200
    body = listed.json()
    assert body["count"] >= 1
    traces = [it["trace_id"] for it in body["items"]]
    assert job["trace_id"] in traces

    reopen = client.get(f"/api/history/{job['trace_id']}")
    assert reopen.status_code == 200
    assert reopen.json()["status"] == "reopened"

    # Reopen made it the session's current dashboard.
    dash = client.get("/api/dashboard")
    assert dash.status_code == 200
    assert dash.json().get("dataset_profile") is not None


def test_history_unknown_trace_is_404(client):
    assert client.get("/api/history/nope-nope").status_code == 404


def test_history_is_not_visible_to_other_guest_session(client):
    sub = client.post("/api/jobs/upload", files=_csv_files())
    _wait_done(client, sub.json()["job_id"])

    other = client.get(
        "/api/history",
        headers={"X-Guest-Mode": "1", "X-Guest-Session-Id": _sgn("stranger")},
    )
    assert other.status_code == 200
    assert other.json()["count"] == 0
