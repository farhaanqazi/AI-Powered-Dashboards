"""Phase 10 S10.1 — idempotent jobs keyed on the data hash.

Re-submitting identical bytes from the same session must reuse the existing
job (no duplicate spool, no recompute); a different session is independent;
and a failed/cancelled prior run never blocks a fresh attempt.
"""
from __future__ import annotations

import io

from src.auth import sign_guest_session_id as _sgn
from src.jobs.store import _MemoryJobStore


def _files(content: bytes = b"a,b\n1,2\n3,4\n"):
    return {"dataset": ("d.csv", io.BytesIO(content), "text/csv")}


def test_same_session_same_bytes_reuses_job(client):
    payload = _files()
    first = client.post("/api/jobs/upload", files=payload)
    assert first.status_code == 202
    j1 = first.json()

    second = client.post("/api/jobs/upload", files=_files())
    assert second.status_code == 202
    j2 = second.json()

    assert j2["job_id"] == j1["job_id"]
    assert j2.get("idempotent") is True
    assert j2["backend"] == "idempotent-reuse"
    assert j2["trace_id"] == j1["trace_id"]


def test_different_bytes_get_distinct_jobs(client):
    a = client.post("/api/jobs/upload", files=_files(b"x,y\n1,2\n")).json()
    b = client.post("/api/jobs/upload", files=_files(b"x,y\n9,9\n8,8\n")).json()
    assert a["job_id"] != b["job_id"]
    assert not b.get("idempotent")


def test_different_session_is_independent(client):
    a = client.post("/api/jobs/upload", files=_files()).json()
    b = client.post(
        "/api/jobs/upload",
        files=_files(),
        headers={"X-Guest-Mode": "1", "X-Guest-Session-Id": _sgn("other-guest")},
    ).json()
    assert a["job_id"] != b["job_id"]


def test_failed_or_cancelled_prior_run_does_not_block_retry():
    store = _MemoryJobStore()
    store.create("j1", session_key="s", trace_id="t", filename="f.csv",
                 data_hash="H")
    assert store.find_active_by_hash("s", "H") == "j1"  # queued ⇒ active

    store.append_event("j1", {"phase": "running", "percent": 10})
    assert store.find_active_by_hash("s", "H") == "j1"

    store.set_error("j1", "boom")  # terminal-failed
    assert store.find_active_by_hash("s", "H") is None  # retry allowed

    # An unrelated hash never matches.
    assert store.find_active_by_hash("s", "OTHER") is None


def test_done_job_is_reused_not_recomputed():
    store = _MemoryJobStore()
    store.create("j2", session_key="s", trace_id="t", filename="f.csv",
                 data_hash="H")
    store.append_event("j2", {"phase": "done", "percent": 100,
                              "trace_id": "t"})
    # A completed job is still "active" for idempotency: identical bytes
    # return the finished result rather than triggering a recompute.
    assert store.find_active_by_hash("s", "H") == "j2"
