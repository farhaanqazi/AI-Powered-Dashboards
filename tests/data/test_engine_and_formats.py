"""Phase 10 S10.2 + S10.3 — DataFrame engine seam & multi-format ingestion.

The seam must produce an identically-shaped pandas frame regardless of engine
or source format, so the rest of the (pandas-only) pipeline is untouched.
"""
from __future__ import annotations

import io

import pandas as pd
import pytest

from src import config
from src.data.engine import (
    DuckDBEngine,
    PandasEngine,
    PolarsEngine,
    get_engine,
    reset_engine_for_tests,
)
from src.data.formats import detect_format, read_dataframe
from src.data.parser import load_table_from_file

_DF = pd.DataFrame(
    {"region": ["N", "S", "E", "W"] * 3,
     "revenue": [10.0, 20.5, 30.0, 5.5] * 3,
     "qty": list(range(12))}
)


@pytest.fixture(autouse=True)
def _reset_engine():
    reset_engine_for_tests()
    yield
    reset_engine_for_tests()


def _csv() -> bytes:
    return _DF.to_csv(index=False).encode()


def _parquet() -> bytes:
    b = io.BytesIO()
    _DF.to_parquet(b)
    return b.getvalue()


def _xlsx() -> bytes:
    b = io.BytesIO()
    _DF.to_excel(b, index=False)
    return b.getvalue()


def _ndjson() -> bytes:
    return _DF.to_json(orient="records", lines=True).encode()


def _json() -> bytes:
    return _DF.to_json(orient="records").encode()


# --- engine seam ----------------------------------------------------------

@pytest.mark.parametrize("engine", [PandasEngine(), PolarsEngine(), DuckDBEngine()])
@pytest.mark.parametrize(
    "fmt,maker",
    [("csv", _csv), ("parquet", _parquet), ("ndjson", _ndjson)],
)
def test_every_engine_yields_equivalent_frame(engine, fmt, maker):
    df = engine.read(maker(), fmt)
    assert list(df.columns) == ["region", "revenue", "qty"]
    assert len(df) == 12
    assert df["qty"].tolist() == list(range(12))


def test_pandas_engine_reads_excel():
    df = PandasEngine().read(_xlsx(), "xlsx")
    assert len(df) == 12 and "revenue" in df.columns


def test_get_engine_is_config_driven_and_falls_back(monkeypatch):
    monkeypatch.setattr(config, "DATAFRAME_ENGINE", "polars")
    reset_engine_for_tests()
    assert get_engine().name == "polars"
    monkeypatch.setattr(config, "DATAFRAME_ENGINE", "nonsense")
    reset_engine_for_tests()
    assert get_engine().name == "pandas"  # unknown ⇒ pandas


# --- format detection -----------------------------------------------------

@pytest.mark.parametrize(
    "name,expected",
    [
        ("a.csv", "csv"),
        ("a.parquet", "parquet"),
        ("a.PQ", "parquet"),
        ("a.xlsx", "xlsx"),
        ("a.ndjson", "ndjson"),
        ("a.jsonl", "jsonl"),
        ("a.txt", None),
        ("noext", None),
        (None, None),
    ],
)
def test_detect_format(name, expected):
    assert detect_format(name) == expected


def test_detect_format_respects_allowlist(monkeypatch):
    monkeypatch.setattr(config, "INGEST_ALLOWED_FORMATS", ["csv"])
    assert detect_format("a.parquet") is None
    assert detect_format("a.csv") == "csv"


# --- read_dataframe normalization + cap -----------------------------------

def test_read_dataframe_normalizes_columns_and_caps_rows():
    b = io.BytesIO()
    pd.DataFrame({"My Col": list(range(50)), "x-y": list(range(50))}).to_parquet(b)
    df, warns = read_dataframe(b.getvalue(), "big.parquet", max_rows=10)
    assert list(df.columns) == ["My_Col", "x_y"]
    assert len(df) == 10
    assert any("sampled" in w for w in warns)


def test_read_dataframe_rejects_unsupported():
    with pytest.raises(ValueError):
        read_dataframe(b"irrelevant", "data.txt")


# --- parser interface -----------------------------------------------------

def test_load_table_from_file_csv_delegates():
    res = load_table_from_file(io.BytesIO(_csv()), filename="d.csv")
    assert res.success and res.df is not None
    assert len(res.df) == 12


def test_load_table_from_file_parquet():
    res = load_table_from_file(io.BytesIO(_parquet()), filename="d.parquet")
    assert res.success and list(res.df.columns) == ["region", "revenue", "qty"]


def test_load_table_from_file_unsupported():
    res = load_table_from_file(io.BytesIO(b"x"), filename="d.exe")
    assert not res.success and res.error_code == "UNSUPPORTED_FORMAT"


# --- end-to-end through the job pipeline -----------------------------------

def _wait_terminal(client, job_id, timeout=60):
    import time
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(f"/api/jobs/{job_id}").json()
        if last["status"] in ("done", "failed", "cancelled"):
            return last
        time.sleep(0.5)
    raise AssertionError(f"job did not finish: {last}")


def test_parquet_upload_runs_through_jobs_pipeline(client):
    files = {"dataset": ("sales.parquet", io.BytesIO(_parquet()),
                          "application/octet-stream")}
    sub = client.post("/api/jobs/upload", files=files)
    assert sub.status_code == 202, sub.text
    final = _wait_terminal(client, sub.json()["job_id"])
    assert final["status"] == "done", final


def test_unsupported_extension_rejected_at_api(client):
    files = {"dataset": ("bad.exe", io.BytesIO(b"nope"), "application/octet-stream")}
    r = client.post("/api/jobs/upload", files=files)
    assert r.status_code == 400
    assert "Unsupported file type" in r.text
