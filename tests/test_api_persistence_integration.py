"""End-to-end: the HTTP API persists through the repository, not a dict."""


def test_upload_persists_via_repository(client, upload_files):
    from src.persistence.repository import get_repository

    r = client.post("/api/upload", files=upload_files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success"

    repo = get_repository()
    stored = repo.get("guest:pytest-session")
    assert stored is not None
    assert stored["original_filename"] == "sample_data.csv"


def test_upload_does_not_create_trace_id_keyed_row(client, upload_files):
    """§11 issue 5: trace_id must never be a storage key."""
    from src.persistence.repository import get_repository

    r = client.post("/api/upload", files=upload_files)
    trace_id = r.json()["trace_id"]
    repo = get_repository()
    assert repo.get(trace_id) is None
    assert repo.get("guest:pytest-session") is not None
    assert repo.count() == 1


import json


def test_stream_persists_via_repository(client, upload_files):
    from src.persistence.repository import get_repository

    with client.stream("POST", "/api/upload/stream", files=upload_files) as resp:
        assert resp.status_code == 200
        body = resp.read().decode("utf-8")

    repo = get_repository()
    stored = repo.get("guest:pytest-session")
    assert stored is not None
    assert "done" in body
    assert repo.count() == 1


from unittest.mock import patch


def test_load_external_persists_via_repository(client):
    import pandas as pd
    from src.data.parser import LoadResult
    from src.persistence.repository import get_repository

    fake = LoadResult(
        df=pd.DataFrame({
            "x": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [2.0, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        }),
        success=True,
        warnings=[],
    )
    with patch("main.load_csv_from_url", return_value=fake):
        r = client.post(
            "/api/load_external",
            json={"external_source": "https://example.com/data.csv"},
        )
    assert r.status_code == 200, r.text
    repo = get_repository()
    assert repo.get("guest:pytest-session") is not None
    assert repo.count() == 1


def test_dashboard_reads_from_repository(client, upload_files):
    client.post("/api/upload", files=upload_files)
    r = client.get("/api/dashboard")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["original_filename"] == "sample_data.csv"


def test_dashboard_empty_when_repository_has_no_row(client):
    r = client.get("/api/dashboard")
    assert r.status_code == 200
    assert r.json()["status"] == "empty"


def test_no_dashboard_storage_attribute_remains():
    """The in-process dict must be fully removed (§11 issue 1)."""
    import main as main_module
    assert not hasattr(main_module, "dashboard_storage")
    assert not hasattr(main_module, "storage_lock")


def test_dashboard_survives_repository_singleton_reset(client, upload_files):
    """Simulate a process restart: drop the in-memory singleton, rebuild it
    from the same DB file, and confirm the dashboard is still there.
    This is the core proof of §11 issue 1 (no persistence) being fixed."""
    import src.persistence.repository as repo_mod

    client.post("/api/upload", files=upload_files)
    assert client.get("/api/dashboard").json()["status"] == "ready"

    # Keep the conftest-bound singleton so it can be restored afterwards.
    original = repo_mod._repository
    try:
        # "Restart": forget the singleton, rebuild from the same DATABASE_URL.
        repo_mod._repository = None
        rebuilt = repo_mod.get_repository()
        assert rebuilt.get("guest:pytest-session") is not None

        # And the HTTP layer still serves it after the reset.
        assert client.get("/api/dashboard").json()["status"] == "ready"
    finally:
        # A real restarted process would not coexist with the prior engine;
        # here both live in one process. Dispose the freshly-built engine
        # (it otherwise keeps the SQLite test DB locked on Windows, breaking
        # the session-teardown unlink) and restore the conftest singleton.
        if repo_mod._repository is not None and repo_mod._repository is not original:
            repo_mod._repository._sf.kw["bind"].dispose()
        repo_mod._repository = original
