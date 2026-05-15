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
