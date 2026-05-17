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
    """S6.3: a contract auto-accepts (auto-locks, no review) only when mean
    per-field confidence ≥ ``config.AUTO_ACCEPT_CONFIDENCE`` AND it is not
    pii_blocked AND a grain was detected. Returns (accepted, reasons)."""
    reasons: List[str] = []
    if contract.pii_blocked:
        reasons.append(
            "This dataset contains sensitive personal data, so it can’t be "
            "shared with the AI. A review won’t change that."
        )
    if not contract.grain:
        reasons.append(
            "Couldn’t identify what each row represents (no clear unique "
            "record was found)."
        )
    confs = [f.confidence for f in contract.fields.values()] or [0.0]
    mean_conf = sum(confs) / len(confs)
    if mean_conf < config.AUTO_ACCEPT_CONFIDENCE:
        reasons.append(
            f"Column types were detected with low certainty "
            f"({mean_conf * 100:.0f}%), below the auto-approve level "
            f"({config.AUTO_ACCEPT_CONFIDENCE * 100:.0f}%) — a quick check is "
            f"recommended."
        )
    # PII never auto-accepts; everything else accepts iff no blocking reason.
    accepted = not reasons
    return accepted, reasons


class DataQualityReport(BaseModel):
    """Emitted whenever a dataset cannot be silently auto-accepted: undetectable
    grain, pii_blocked, or a sub-threshold (unlockable-without-review)
    contract. ``status`` drives the Phase 7 HITL UX."""

    model_config = ConfigDict(extra="forbid")

    status: str  # "ok" | "review" | "blocked"
    reasons: List[str] = Field(default_factory=list)
    schema_fingerprint: str = ""
    grain: List[str] = Field(default_factory=list)
    sensitivity: str = "public"
    pii_blocked: bool = False
    pii_columns: dict = Field(default_factory=dict)
    mean_confidence: float = 0.0
    cleaning: dict = Field(default_factory=dict)
    vetoes: list = Field(default_factory=list)
    flags: list = Field(default_factory=list)
    auto_accepted: bool = False


def build_dq_report(contract: DatasetContract, ingest: Any, crit: Any) -> DataQualityReport:
    accepted, reasons = evaluate_acceptance(contract)
    if contract.pii_blocked:
        status = "blocked"
    elif accepted:
        status = "ok"
    else:
        status = "review"

    confs = [f.confidence for f in contract.fields.values()] or [0.0]
    return DataQualityReport(
        status=status,
        reasons=reasons,
        schema_fingerprint=contract.schema_fingerprint,
        grain=list(contract.grain),
        sensitivity=contract.sensitivity,
        pii_blocked=contract.pii_blocked,
        pii_columns=dict(getattr(ingest, "pii_columns", {}) or {}),
        mean_confidence=round(sum(confs) / len(confs), 4),
        cleaning=ingest.manifest.model_dump(mode="json") if ingest is not None else {},
        vetoes=[v.model_dump() for v in getattr(crit, "vetoes", [])],
        flags=[f.model_dump() for f in getattr(crit, "flags", [])],
        auto_accepted=accepted and not contract.pii_blocked,
    )
