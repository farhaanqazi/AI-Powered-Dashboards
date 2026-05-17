"""Phase 5 — contract layer wired into the live pipeline."""
import pandas as pd

from src.core.pipeline import build_dashboard_from_df


def _df():
    return pd.DataFrame(
        {
            "region": ["N", "S", "E", "W"] * 8,
            "revenue": [f"${v:,.2f}" for v in ([100.0, 250.0, 320.0, 90.0] * 8)],
            "year": [2020, 2021, 2022, 2023] * 8,
        }
    )


def test_contract_and_data_quality_threaded_into_profile():
    state = build_dashboard_from_df(_df(), original_filename="t.csv")
    prof = state.dataset_profile
    assert "contract" in prof
    assert prof["contract"]["schema_fingerprint"]
    assert "data_quality" in prof
    assert "cleaning" in prof["data_quality"]
    assert prof["sensitivity"] in ("public", "sensitive")
    assert prof["pii_blocked"] is False


def test_currency_cleaned_by_ingest_gate_before_profiling():
    state = build_dashboard_from_df(_df(), original_filename="t.csv")
    cols = {c["name"]: c for c in state.dataset_profile["columns"]}
    # revenue arrived as "$100.00" strings; ingest gate coerced it numeric.
    assert cols["revenue"]["role"] == "numeric"


def test_pii_email_blocks_egress_and_marks_sensitive():
    df = pd.DataFrame(
        {
            "email": [f"user{i}@example.com" for i in range(20)],
            "spend": [float(i) for i in range(20)],
        }
    )
    state = build_dashboard_from_df(df, original_filename="p.csv")
    prof = state.dataset_profile
    assert prof["pii_blocked"] is True
    assert prof["sensitivity"] == "sensitive"
    assert "email" in prof["data_quality"]["pii_columns"]
    # Sensitive column must be redacted in the persisted contract field.
    assert prof["contract"]["fields"]["email"]["sensitivity"] == "sensitive"


def test_all_sentinel_dataset_rejected_with_no_charts():
    df = pd.DataFrame({"a": ["NA", "n/a", "-"], "b": ["null", "?", "--"]})
    state = build_dashboard_from_df(df, original_filename="bad.csv")
    assert state.all_charts == []
    assert state.errors
    assert state.dataset_profile["data_quality"]["rejected"] is True


def test_ground_truth_redacts_sensitive_columns():
    from src.analysis.llm_analyst import _ground_truth
    from src.analysis.data_structures import EnrichedProfile

    class _FC:
        def __init__(self, sensitivity, pii):
            self.sensitivity = sensitivity
            self.pii_entities = pii

    class _C:
        fields = {"email": _FC("sensitive", ("EMAIL_ADDRESS",)),
                  "spend": _FC("public", ())}

    profs = {
        "email": EnrichedProfile(name="email", dtype="object", role="text",
                                 null_count=0, unique_count=20,
                                 stats={"count": 20},
                                 top_categories=[{"value": "a@b.com", "count": 1}]),
        "spend": EnrichedProfile(name="spend", dtype="float64", role="numeric",
                                 null_count=0, unique_count=20,
                                 stats={"mean": 5.0}, top_categories=[]),
    }
    gt = _ground_truth(profs, [], {}, _C())
    cols = {c["name"]: c for c in gt["columns"]}
    assert cols["email"]["redacted"] is True
    assert cols["email"]["top_categories"] == []
    assert cols["email"]["stats"] == {}
    assert cols["spend"]["redacted"] is False
    assert cols["spend"]["stats"] == {"mean": 5.0}
