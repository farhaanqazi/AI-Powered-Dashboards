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
