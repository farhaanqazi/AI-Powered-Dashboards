"""Phase 6 — Data Quality Report + auto-accept rule.

``build_dq_report`` produces the verdict the frontend (Phase 7) and the
pipeline use to decide whether a dashboard can be shown as-is, needs human
schema review, or is PII-blocked. ``evaluate_acceptance`` is the S6.3
auto-accept rule — config-driven, no literals.
"""
from __future__ import annotations

from typing import Any, List, Tuple

from pydantic import BaseModel, ConfigDict, Field

from src import config
from src.contract.models import DatasetContract


def evaluate_acceptance(contract: DatasetContract) -> Tuple[bool, List[str]]:
    """S6.3 (calibrated 2026-05-18): a contract auto-accepts (auto-locks, no
    review) only when ALL hold — a grain was detected, the **mean** per-field
    confidence ≥ ``config.AUTO_ACCEPT_CONFIDENCE`` (overall quality bar), AND
    no single column is below ``config.CRITICAL_FIELD_CONFIDENCE_FLOOR`` (the
    catastrophe guard). Returns (accepted, reasons).

    Why two gates, not a raw minimum: a raw min sent datasets to review for a
    single legitimately-fuzzy text column (~0.6), causing review fatigue. A raw
    mean let one catastrophically mis-typed column (0.10 among 0.95s) auto-lock.
    Mean keeps the overall bar; the floor only trips on worse-than-coin-flip
    columns — catching the genuinely-bad case without the false interruptions.
    The displayed ``mean_confidence`` metric is unaffected.

    NOTE (PII model change, supersedes the original fail-closed invariant):
    the deterministic dashboard never sends data anywhere, so PII does NOT
    block dashboard acceptance. PII only gates the *AI Insights* layer behind
    explicit user consent — see ``ai_consent_required`` below. Sharing one's
    own sensitive data is the user's prerogative and responsibility.
    """
    reasons: List[str] = []
    if not contract.grain:
        reasons.append(
            "Couldn’t identify what each row represents (no clear unique "
            "record was found)."
        )
    confs = [f.confidence for f in contract.fields.values()] or [0.0]
    mean_conf = sum(confs) / len(confs)
    min_conf = min(confs)
    if mean_conf < config.AUTO_ACCEPT_CONFIDENCE:
        reasons.append(
            f"Overall column-type certainty is low "
            f"({mean_conf * 100:.0f}%), below the auto-approve level "
            f"({config.AUTO_ACCEPT_CONFIDENCE * 100:.0f}%) — a quick check is "
            f"recommended."
        )
    if min_conf < config.CRITICAL_FIELD_CONFIDENCE_FLOOR:
        reasons.append(
            f"At least one column's type is highly uncertain "
            f"({min_conf * 100:.0f}%) and was likely mis-detected — please "
            f"review that column before continuing."
        )
    accepted = not reasons
    return accepted, reasons


class DataQualityReport(BaseModel):
    """Emitted whenever a dataset cannot be silently auto-accepted: undetectable
    grain, pii_blocked, or a sub-threshold (unlockable-without-review)
    contract. ``status`` drives the Phase 7 HITL UX."""

    model_config = ConfigDict(extra="forbid")

    status: str  # "ok" | "review"  (PII no longer yields "blocked")
    reasons: List[str] = Field(default_factory=list)
    schema_fingerprint: str = ""
    grain: List[str] = Field(default_factory=list)
    sensitivity: str = "public"
    # ``pii_blocked`` now means "PII present → AI egress needs consent", NOT a
    # permanent wall. The dashboard still builds. Kept for back-compat.
    pii_blocked: bool = False
    pii_present: bool = False
    ai_consent: bool = False           # user explicitly allowed AI on this data
    ai_consent_required: bool = False  # PII present AND consent not yet given
    pii_columns: dict = Field(default_factory=dict)
    mean_confidence: float = 0.0
    cleaning: dict = Field(default_factory=dict)
    vetoes: list = Field(default_factory=list)
    flags: list = Field(default_factory=list)
    auto_accepted: bool = False


def build_dq_report(
    contract: DatasetContract, ingest: Any, crit: Any, ai_consent: bool = False
) -> DataQualityReport:
    accepted, reasons = evaluate_acceptance(contract)
    # PII never produces "blocked" anymore — the dashboard always builds.
    status = "ok" if accepted else "review"

    pii_present = bool(contract.pii_blocked)
    ai_consent_required = pii_present and not ai_consent

    confs = [f.confidence for f in contract.fields.values()] or [0.0]
    return DataQualityReport(
        status=status,
        reasons=reasons,
        schema_fingerprint=contract.schema_fingerprint,
        grain=list(contract.grain),
        sensitivity=contract.sensitivity,
        pii_blocked=contract.pii_blocked,
        pii_present=pii_present,
        ai_consent=ai_consent,
        ai_consent_required=ai_consent_required,
        pii_columns=dict(getattr(ingest, "pii_columns", {}) or {}),
        mean_confidence=round(sum(confs) / len(confs), 4),
        cleaning=ingest.manifest.model_dump(mode="json") if ingest is not None else {},
        vetoes=[v.model_dump() for v in getattr(crit, "vetoes", [])],
        flags=[f.model_dump() for f in getattr(crit, "flags", [])],
        auto_accepted=accepted,
    )
