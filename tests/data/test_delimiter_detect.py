"""Robust CSV delimiter detection (DuckDB sniffer). A ;/tab/|-separated upload
must not be misread as one comma-delimited column. These exercise the real
upload entry point (`load_csv_from_file`). Skipped when duckdb is absent — the
feature is best-effort and falls back to pandas' comma default."""
import io

import pytest

pytest.importorskip("duckdb")

from src.data.parser import load_csv_from_file, _sniff_csv_delimiter


def test_semicolon_csv_is_split_into_columns():
    raw = b"region;units;product\nEMEA;12;A\nAPAC;20;B\n"
    res = load_csv_from_file(io.BytesIO(raw))
    assert res.success, res.detail
    assert list(res.df.columns) == ["region", "units", "product"]
    assert len(res.df) == 2
    assert any("semicolon" in w.lower() for w in (res.warnings or []))


def test_pipe_csv_is_split_into_columns():
    raw = b"a|b|c\n1|2|3\n4|5|6\n"
    res = load_csv_from_file(io.BytesIO(raw))
    assert res.success and list(res.df.columns) == ["a", "b", "c"]
    assert len(res.df) == 2


def test_comma_csv_is_unchanged_and_unwarned():
    raw = b"a,b,c\n1,2,3\n"
    res = load_csv_from_file(io.BytesIO(raw))
    assert res.success and list(res.df.columns) == ["a", "b", "c"]
    # No delimiter warning for ordinary comma files.
    assert not any("separated file" in w for w in (res.warnings or []))


def test_sniff_detects_and_rewinds():
    bio = io.BytesIO(b"x;y\n1;2\n")
    assert _sniff_csv_delimiter(bio) == ";"
    assert bio.tell() == 0  # caller's read position is preserved
