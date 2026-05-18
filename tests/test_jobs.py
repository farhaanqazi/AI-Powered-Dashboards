"""Phase 10 S10.1 — async analysis job flow (in-process backend).

Exercises the production-facing contract without Redis/Arq: submit → 202 +
job_id, poll status to terminal, dashboard persisted, ownership enforced.
"""
import time

import pytest

from src.auth import sign_guest_session_id as _sgn


def _wait_terminal(client, job_id, timeout=60):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        last = r.json()
        if last["status"] in ("done", "failed", "cancelled"):
            return last
        time.sleep(0.5)
    raise AssertionError(f"job {job_id} did not finish; last={last}")


def test_job_upload_returns_202_and_completes(client, upload_files):
    sub = client.post("/api/jobs/upload", files=upload_files)
    assert sub.status_code == 202, sub.text
    body = sub.json()
    assert body["status"] == "accepted"
    assert body["job_id"] and body["trace_id"]
    assert body["backend"] in ("inprocess", "arq")

    final = _wait_terminal(client, body["job_id"])
    assert final["status"] == "done", final
    assert final["trace_id"] == body["trace_id"]

    # The finished dashboard is persisted under the session, same as the
    # synchronous path — the frontend loads it normally.
    dash = client.get("/api/dashboard")
    assert dash.status_code == 200
    assert dash.json().get("dataset_profile") is not None


def test_job_status_is_session_scoped(client, upload_files):
    sub = client.post("/api/jobs/upload", files=upload_files)
    job_id = sub.json()["job_id"]

    # A different guest session must not see someone else's job.
    other = client.get(
        f"/api/jobs/{job_id}",
        headers={"X-Guest-Mode": "1", "X-Guest-Session-Id": _sgn("someone-else")},
    )
    assert other.status_code == 404


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/does-not-exist").status_code == 404


def test_job_cancel_is_accepted(client, upload_files):
    sub = client.post("/api/jobs/upload", files=upload_files)
    job_id = sub.json()["job_id"]
    c = client.post(f"/api/jobs/{job_id}/cancel")
    assert c.status_code == 200
    assert c.json()["status"] == "cancelling"
    # Either cancelled in time or finished first — both are valid terminals.
    final = _wait_terminal(client, job_id)
    assert final["status"] in ("cancelled", "done")


def test_legacy_sync_upload_still_works(client, upload_files):
    # Back-compat: the old endpoint must remain functional.
    r = client.post("/api/upload", files=upload_files)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "success"
