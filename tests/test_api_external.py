"""Tests for POST /api/validate_external and POST /api/load_external.

These tests stub `requests.get` so they don't perform real network I/O.
"""
from unittest.mock import patch, MagicMock


def test_validate_external_rejects_empty(client):
    response = client.post("/api/validate_external", json={"external_source": ""})
    assert response.status_code == 400


def test_validate_external_rejects_malformed_url(client):
    response = client.post(
        "/api/validate_external", json={"external_source": "http://"},
    )
    assert response.status_code == 400


def test_validate_external_rejects_kaggle_dataset_page_url(client):
    response = client.post(
        "/api/validate_external",
        json={"external_source": "https://www.kaggle.com/datasets/foo/bar"},
    )
    assert response.status_code == 400
    assert "kaggle" in response.json()["detail"].lower()


def test_validate_external_accepts_valid_slug(client):
    response = client.post(
        "/api/validate_external", json={"external_source": "owner/dataset"},
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_validate_external_rejects_bad_slug(client):
    response = client.post(
        "/api/validate_external", json={"external_source": "missing-slash"},
    )
    assert response.status_code == 400


def test_validate_external_accepts_raw_csv_url(client):
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {"Content-Type": "text/csv"}
    fake_response.iter_content = lambda chunk_size: iter([b"col1,col2\n1,2\n"])
    fake_response.close = lambda: None

    with patch("main.requests.get", return_value=fake_response):
        response = client.post(
            "/api/validate_external",
            json={"external_source": "https://example.com/data.csv"},
        )
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_validate_external_rejects_html_content_type(client):
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {"Content-Type": "text/html; charset=utf-8"}
    fake_response.iter_content = lambda chunk_size: iter([b"<html></html>"])
    fake_response.close = lambda: None

    with patch("main.requests.get", return_value=fake_response):
        response = client.post(
            "/api/validate_external",
            json={"external_source": "https://example.com/page"},
        )
    assert response.status_code == 400
    assert "html" in response.json()["detail"].lower() or "csv" in response.json()["detail"].lower()


def test_validate_external_rejects_html_body_with_csv_content_type(client):
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {"Content-Type": "text/csv"}
    fake_response.iter_content = lambda chunk_size: iter([b"<!DOCTYPE html><html></html>"])
    fake_response.close = lambda: None

    with patch("main.requests.get", return_value=fake_response):
        response = client.post(
            "/api/validate_external",
            json={"external_source": "https://example.com/sneaky"},
        )
    assert response.status_code == 400


def test_load_external_rejects_bad_slug_shape(client):
    response = client.post(
        "/api/load_external", json={"external_source": "no-slash-here"},
    )
    assert response.status_code == 400


def test_load_external_loads_url_and_returns_dashboard(client):
    import pandas as pd
    from src.data.parser import LoadResult

    fake_result = LoadResult(
        df=pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        }),
        success=True,
        warnings=[],
    )

    with patch("main.load_csv_from_url", return_value=fake_result):
        response = client.post(
            "/api/load_external",
            json={"external_source": "https://example.com/data.csv"},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    assert body["data"]["original_filename"] == "https://example.com/data.csv"
