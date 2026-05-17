"""Phase 1 — ingest contract gate behaviour."""
import pandas as pd

from src.contract import run_ingest_gate
from src.contract.models import CleaningManifest, IngestResult


def test_currency_and_thousands_coerced_to_numeric():
    df = pd.DataFrame({"revenue": ["$1,234.50", "$2,000", "(1,200)", "3000"]})
    res = run_ingest_gate(df)
    assert res.ok and not res.rejected
    assert pd.api.types.is_numeric_dtype(res.df["revenue"])
    assert res.df["revenue"].tolist() == [1234.50, 2000.0, -1200.0, 3000.0]
    assert "revenue" in res.manifest.coerced_numeric


def test_sentinels_become_na_and_are_recorded():
    # Second column keeps the rows alive so the nulled sentinels are observable
    # (single-column rows would be fully-null and dropped instead).
    df = pd.DataFrame(
        {"city": ["Paris", "N/A", "-", "Berlin"], "pop": [1, 2, 3, 4]}
    )
    res = run_ingest_gate(df)
    assert res.df["city"].isna().sum() == 2
    assert res.manifest.sentinels_nulled["city"] == 2


def test_fully_null_rows_dropped():
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", None, "z"]})
    res = run_ingest_gate(df)
    assert res.manifest.dropped_null_rows == 1
    assert len(res.df) == 2


def test_empty_after_cleaning_is_rejected():
    df = pd.DataFrame({"a": ["NA", "null"], "b": ["-", "n/a"]})
    res = run_ingest_gate(df)
    assert res.rejected and not res.ok
    assert res.df is None
    assert "empty" in (res.reject_reason or "").lower()


def test_no_columns_rejected_fail_closed():
    res = run_ingest_gate(pd.DataFrame())
    assert res.rejected
    assert res.sensitivity == "sensitive"  # SENSITIVITY_FAIL_CLOSED default


def test_pii_email_blocks_egress():
    df = pd.DataFrame(
        {
            "email": ["alice@example.com", "bob@test.org", "c@d.co"],
            "amount": [10, 20, 30],
        }
    )
    res = run_ingest_gate(df)
    assert res.sensitivity == "sensitive"
    assert res.pii_blocked is True
    assert "EMAIL_ADDRESS" in res.pii_columns.get("email", [])


def test_gate_does_not_mutate_caller_frame():
    df = pd.DataFrame({"x": ["1,000", "2,000"]})
    before = df.copy()
    run_ingest_gate(df)
    pd.testing.assert_frame_equal(df, before)


def test_result_is_typed_models():
    res = run_ingest_gate(pd.DataFrame({"n": [1, 2, 3]}))
    assert isinstance(res, IngestResult)
    assert isinstance(res.manifest, CleaningManifest)
    assert res.manifest.original_shape == (3, 1)
