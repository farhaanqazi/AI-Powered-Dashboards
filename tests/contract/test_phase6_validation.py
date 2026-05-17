"""Phase 6 — LLM output contract, DQ report, auto-accept."""
import pandas as pd

from src.contract.models import DatasetContract, FieldContract, LLMOutputContract
from src.contract.dq_report import build_dq_report, evaluate_acceptance
from src.core.pipeline import build_dashboard_from_df


# ---- S6.1 LLMOutputContract ----

def test_llm_output_rejects_unknown_column_and_missing_provenance():
    kpis = [
        {"label": "Good", "provenance": {"source": "column:revenue"}},
        {"label": "Bad col", "provenance": {"source": "column:ghost"}},
        {"label": "No prov"},
    ]
    v = LLMOutputContract.validate_output(kpis, [], {"revenue"})
    assert v.ok is False
    assert v.validated_kpis == 1
    assert any("ghost" in r for r in v.reasons)
    assert any("no provenance" in r.lower() for r in v.reasons)


def test_llm_output_accepts_clean_payload():
    kpis = [{"label": "Rev", "provenance": {"source": "column:revenue"}}]
    charts = [{"intent": "distribution", "x_field": "revenue", "y_field": None}]
    v = LLMOutputContract.validate_output(kpis, charts, {"revenue"})
    assert v.ok and v.validated_kpis == 1 and v.validated_charts == 1


def test_llm_output_rejects_bad_intent():
    v = LLMOutputContract.validate_output(
        [], [{"intent": "pie_of_doom", "x_field": "a"}], {"a"}
    )
    assert v.ok is False


# ---- S6.3 auto-accept ----

def _contract(conf, *, grain=("id",), pii=False):
    return DatasetContract(
        schema_fingerprint="fp", grain=grain, pii_blocked=pii,
        sensitivity="sensitive" if pii else "public",
        fields={"id": FieldContract(name="id", dtype="int64", role="identifier",
                                    confidence=conf)},
    )


def test_high_confidence_non_pii_auto_accepts():
    ok, reasons = evaluate_acceptance(_contract(0.95))
    assert ok and reasons == []


def test_pii_never_auto_accepts():
    ok, reasons = evaluate_acceptance(_contract(0.99, pii=True))
    assert ok is False
    assert any("PII" in r for r in reasons)


def test_low_confidence_needs_review():
    ok, reasons = evaluate_acceptance(_contract(0.10))
    assert ok is False
    assert any("threshold" in r for r in reasons)


def test_missing_grain_needs_review():
    ok, reasons = evaluate_acceptance(_contract(0.99, grain=()))
    assert ok is False
    assert any("grain" in r.lower() for r in reasons)


# ---- S6.2 DQ report ----

def test_dq_report_blocked_on_pii():
    rep = build_dq_report(_contract(0.9, pii=True), None, type("C", (), {"vetoes": [], "flags": []})())
    assert rep.status == "blocked"
    assert rep.pii_blocked is True


def test_dq_report_ok_when_accepted():
    rep = build_dq_report(_contract(0.95), None, type("C", (), {"vetoes": [], "flags": []})())
    assert rep.status == "ok" and rep.auto_accepted is True


# ---- pipeline integration ----

def test_pipeline_emits_dq_report_and_autolocks_clean_dataset():
    df = pd.DataFrame({
        "region": ["N", "S", "E", "W"] * 8,
        "revenue": [10.0, 20.0, 30.0, 40.0] * 8,
        "txn_id": list(range(32)),
    })
    state = build_dashboard_from_df(df, original_filename="ok.csv")
    rep = state.eda_summary["data_quality_report"]
    assert rep["status"] in ("ok", "review")
    assert "schema_fingerprint" in rep
    assert state.dataset_profile["data_quality"]["report"] is not None
