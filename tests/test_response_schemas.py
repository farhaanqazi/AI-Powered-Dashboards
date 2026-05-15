"""Pydantic response-model contract tests."""
import pytest

from src.api import schemas


def test_upload_response_validates_minimal_payload():
    payload = {
        "status": "success",
        "trace_id": "abc-123",
        "data": {
            "dataset_profile": {},
            "kpis": [],
            "charts": [],
            "primary_chart": None,
            "category_charts": {},
            "all_charts": [],
            "original_filename": "x.csv",
            "errors": [],
            "warnings": [],
            "critical_totals": {},
            "critical_full_dataset_aggregates": {},
            "eda_summary": {},
        },
    }
    parsed = schemas.UploadResponse.model_validate(payload)
    assert parsed.status == "success"
    assert parsed.trace_id == "abc-123"


def test_dashboard_response_validates_empty_state():
    payload = {
        "status": "empty",
        "timestamp": "2026-01-01T00:00:00",
        "metadata": {"hint": "Upload a dataset to generate insights"},
        "kpis": [],
        "charts": [],
        "eda": {},
        "errors": [],
        "warnings": [],
        "message": "Dashboard initializing.",
        "dataset_profile": {},
        "primary_chart": None,
        "category_charts": {},
        "all_charts": [],
        "original_filename": "",
        "critical_totals": {},
        "critical_full_dataset_aggregates": {},
        "eda_summary": {},
    }
    parsed = schemas.DashboardResponse.model_validate(payload)
    assert parsed.status == "empty"


def test_dashboard_response_validates_ready_state():
    payload = {
        "status": "ready",
        "timestamp": "2026-01-01T00:00:00",
        "metadata": {"columns": 5, "rows": 10, "filename": "x.csv"},
        "kpis": [{"name": "Amount", "score": 0.9}],
        "charts": [],
        "eda": {},
        "errors": [],
        "warnings": [],
        "message": None,
        "dataset_profile": {"n_cols": 5, "n_rows": 10},
        "primary_chart": None,
        "category_charts": {},
        "all_charts": [],
        "original_filename": "x.csv",
        "critical_totals": {},
        "critical_full_dataset_aggregates": {},
        "eda_summary": {},
    }
    parsed = schemas.DashboardResponse.model_validate(payload)
    assert parsed.status == "ready"


def test_validate_external_response():
    parsed = schemas.ValidateExternalResponse.model_validate({"ok": True})
    assert parsed.ok is True


def test_load_external_response_validates():
    payload = {
        "status": "success",
        "trace_id": "xyz",
        "data": {
            "dataset_profile": {},
            "kpis": [],
            "charts": [],
            "primary_chart": None,
            "category_charts": {},
            "all_charts": [],
            "original_filename": "url",
            "errors": [],
            "warnings": [],
            "critical_totals": {},
            "critical_full_dataset_aggregates": {},
            "eda_summary": {},
        },
    }
    parsed = schemas.LoadExternalResponse.model_validate(payload)
    assert parsed.status == "success"
