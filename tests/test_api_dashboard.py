"""Tests for GET /api/dashboard (session-keyed retrieval)."""


def test_dashboard_returns_empty_when_no_upload_yet(client):
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "empty"
    assert body["kpis"] == []
    assert body["all_charts"] == []


def test_dashboard_returns_ready_after_upload(client, upload_files):
    upload = client.post("/api/upload", files=upload_files)
    assert upload.status_code == 200
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["original_filename"] == "sample_data.csv"
    assert isinstance(body["metadata"]["rows"], int)
    assert isinstance(body["metadata"]["columns"], int)


def test_dashboard_is_per_session(client, upload_files):
    """Different guest session ids must not see each other's dashboards."""
    upload = client.post(
        "/api/upload", files=upload_files,
        headers={"X-Guest-Mode": "1", "X-Guest-Session-Id": "alice"},
    )
    assert upload.status_code == 200

    bob = client.get(
        "/api/dashboard",
        headers={"X-Guest-Mode": "1", "X-Guest-Session-Id": "bob"},
    )
    assert bob.status_code == 200
    assert bob.json()["status"] == "empty"

    alice = client.get(
        "/api/dashboard",
        headers={"X-Guest-Mode": "1", "X-Guest-Session-Id": "alice"},
    )
    assert alice.status_code == 200
    assert alice.json()["status"] == "ready"
