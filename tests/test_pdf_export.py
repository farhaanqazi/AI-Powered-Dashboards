"""Server-side PDF export (GET /api/dashboard/export.pdf + build_dashboard_pdf).

Replaces the old in-browser screenshot path. The unit test renders a hand-built
payload directly; the endpoint test exercises the real upload→persist→export
round trip through the app. Skipped entirely if reportlab is absent (the export
is an additive feature; the rest of the app must still import without it)."""
import pytest

pytest.importorskip("reportlab")

from src.reporting import build_dashboard_pdf


def _sample_payload():
    return {
        "original_filename": "sales_q3.csv",
        "dataset_profile": {
            "n_rows": 1200,
            "n_cols": 3,
            "columns": [
                {"name": "region", "label": "Region", "role": "categorical",
                 "dtype": "object", "null_count": 0},
                {"name": "units", "label": "Units", "role": "numeric",
                 "dtype": "int64", "null_count": 12},
                {"name": "price", "label": "Price", "role": "numeric",
                 "dtype": "float64", "null_count": 0},
            ],
        },
        "kpis": [
            {"label": "Total units", "value": "48,300", "type": "numeric"},
            {"label": "Avg price", "value": "12.40", "type": "numeric"},
            {"label": "Corr: units & price", "value": "0.82", "type": "correlation"},
        ],
        "eda_summary": {
            "ai_narrative": "Sales concentrate in EMEA; price and units correlate.",
            "key_indicators": ["EMEA leads volume", {"title": "APAC fastest growth"}],
            "recommendations": ["Stock EMEA earlier"],
        },
        "all_charts": [
            {"title": "Units by region", "type": "bar", "column": "region",
             "data": [{"category": "EMEA", "value": 22000},
                      {"category": "APAC", "value": 15000},
                      {"category": "AMER", "value": 11300}]},
            {"title": "Share by region", "type": "pie", "column": "region",
             "data": [{"label": "EMEA", "value": 22000},
                      {"label": "APAC", "value": 15000},
                      {"label": "AMER", "value": 11300}]},
            {"title": "Price vs units", "type": "scatter", "column": "price",
             "data": [{"x": 10, "y": 100}, {"x": 12, "y": 140}, {"x": 14, "y": 180}]},
        ],
    }


def test_build_pdf_returns_valid_bytes():
    pdf = build_dashboard_pdf(_sample_payload())
    assert isinstance(pdf, (bytes, bytearray))
    assert pdf[:5] == b"%PDF-"          # valid PDF magic
    assert pdf.rstrip().endswith(b"%%EOF")
    assert len(pdf) > 2000              # real content, not an empty shell


def test_build_pdf_tolerates_empty_payload():
    # Must never raise on a thin/odd payload — a partial PDF beats a 500.
    pdf = build_dashboard_pdf({})
    assert pdf[:5] == b"%PDF-"


def test_build_pdf_skips_unrenderable_charts():
    payload = {
        "original_filename": "x.csv",
        "all_charts": [
            {"title": "Heat", "type": "heatmap", "data": {"z": [[1]]}},
            {"title": "Junk", "type": "bar", "data": [{"category": "a"}]},  # no value
        ],
    }
    pdf = build_dashboard_pdf(payload)
    assert pdf[:5] == b"%PDF-"


def test_export_endpoint_returns_pdf(client, upload_files):
    # Build a real dashboard for the session, then export it.
    up = client.post("/api/upload", files=upload_files)
    assert up.status_code == 200, up.text

    res = client.get("/api/dashboard/export.pdf")
    assert res.status_code == 200, res.text
    assert res.headers["content-type"] == "application/pdf"
    assert "attachment" in res.headers.get("content-disposition", "")
    assert res.content[:5] == b"%PDF-"


def test_export_404_without_dashboard(client):
    # A fresh session with nothing uploaded cannot export.
    res = client.get("/api/dashboard/export.pdf")
    assert res.status_code == 404
