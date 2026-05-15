"""Tests for POST /api/upload (sync upload endpoint)."""
import io

import pytest


def test_upload_happy_path_returns_200_and_dashboard_payload(client, upload_files):
    response = client.post("/api/upload", files=upload_files)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    assert "trace_id" in body
    data = body["data"]
    assert "dataset_profile" in data
    assert "kpis" in data
    assert "all_charts" in data


def test_upload_rejects_non_csv_extension(client):
    files = {"dataset": ("not_a_csv.txt", io.BytesIO(b"col\n1\n"), "text/plain")}
    response = client.post("/api/upload", files=files)
    assert response.status_code == 400
    assert "csv" in response.json()["detail"].lower()


def test_upload_rejects_path_traversal_filename(client):
    files = {"dataset": ("../escape.csv", io.BytesIO(b"col\n1\n"), "text/csv")}
    response = client.post("/api/upload", files=files)
    assert response.status_code == 400
    assert "invalid" in response.json()["detail"].lower()


@pytest.mark.xfail(
    reason="REAL ENDPOINT BUG: an empty multipart filename is not parsed as an "
    "UploadFile, so FastAPI request validation raises a ValueError that Starlette "
    "tries to place into a JSONResponse and fails with 'Object of type ValueError "
    "is not JSON serializable' -> HTTP 500 instead of a clean 400/422. Not fixing "
    "per Phase 0 task scope (tests only); xfail'd so the chain continues.",
    strict=True,
)
def test_upload_rejects_empty_filename(client):
    files = {"dataset": ("", io.BytesIO(b"col\n1\n"), "text/csv")}
    response = client.post("/api/upload", files=files)
    assert response.status_code in (400, 422)


def test_upload_persists_data_so_dashboard_endpoint_can_read_it(client, upload_files):
    upload = client.post("/api/upload", files=upload_files)
    assert upload.status_code == 200
    dashboard = client.get("/api/dashboard")
    assert dashboard.status_code == 200
    assert dashboard.json()["status"] == "ready"
