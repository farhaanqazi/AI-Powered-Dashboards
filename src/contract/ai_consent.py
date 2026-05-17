"""PII-consent AI re-run (supersedes the original PII fail-closed wall).

The deterministic dashboard always builds — nothing it computes ever leaves
the server. The *AI Insights* layer is the only egress, so on a PII-bearing
dataset it is withheld until the user explicitly consents. Consent is the
user's prerogative and responsibility.

When the user consents, this module re-runs ONLY the AI layer against the
transiently-cached cleaned frame + the persisted contract, with redaction
turned OFF (full send — that is what the user agreed to), and merges the AI
outputs back onto the existing dashboard payload. No re-profiling beyond what
the contract already fixes; no new storage backend.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict

from src.analysis.layer_3_relational import run_relational_analysis
from src.analysis.llm_analyst import run_ai_analyst
from src.viz.plotly_renderer import build_charts_from_specs
from src.contract.models import DatasetContract
from src.contract.rebuild import _profiles_from_contract

logger = logging.getLogger(__name__)


def apply_ai_consent(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a NEW payload with AI Insights computed under user consent.

    Raises ValueError with a user-facing message if there is no contract or
    the cleaned frame is no longer cached (the user must re-upload).
    """
    payload = copy.deepcopy(payload)
    profile = payload.get("dataset_profile") or {}
    raw_contract = profile.get("contract")
    if not raw_contract:
        raise ValueError("There’s nothing to enable AI for on this dashboard yet.")

    contract = DatasetContract.model_validate(raw_contract)

    # Record consent regardless of whether we can re-run right now.
    dq = profile.get("data_quality") or {}
    dq["ai_consent"] = True
    report = dq.get("report") or {}
    report["ai_consent"] = True
    report["ai_consent_required"] = False

    from src.contract.df_cache import get_df_cache

    df = get_df_cache().get(contract.schema_fingerprint)
    if df is None:
        # Consent is saved, but the transient cleaned frame expired — AI
        # cannot run without it (architectural invariant: no raw-data store).
        dq["report"] = report
        profile["data_quality"] = dq
        payload["dataset_profile"] = profile
        raise ValueError(
            "Your consent is saved, but this dataset’s data is no longer held "
            "in memory (it’s only kept briefly). Please re-upload the file to "
            "generate AI insights."
        )

    profiles = _profiles_from_contract(df, contract)
    relational = run_relational_analysis(df, profiles)
    eda_summary = payload.get("eda_summary") or {}

    ai = run_ai_analyst(
        profiles,
        relational,
        eda_summary,
        fallback_kpis=payload.get("kpis") or [],
        fallback_specs=payload.get("charts") or [],
        contract=contract,
        redact_sensitive=False,  # full consent — user opted in to send all
    )

    payload["kpis"] = ai["kpis"]
    payload["charts"] = ai["chart_specs"]

    role_counts: Dict[str, int] = {}
    for p in profiles.values():
        role_counts[p.role] = role_counts.get(p.role, 0) + 1
    viz_profile = {
        "n_rows": len(df),
        "n_cols": len(profiles),
        "role_counts": role_counts,
        "columns": [p.__dict__ for p in profiles.values()],
    }
    try:
        all_charts = build_charts_from_specs(
            df, ai["chart_specs"], dataset_profile=viz_profile
        )
        payload["all_charts"] = all_charts
        payload["primary_chart"] = next(
            (c for c in all_charts if c.get("type") == "bar"), None
        )
    except Exception as exc:  # charts are decorative; keep AI text on failure
        logger.warning("Consent re-render of charts failed (%s); keeping prior.", exc)

    eda = dict(eda_summary) if isinstance(eda_summary, dict) else {}
    eda["ai_consent_required"] = False
    if ai.get("narrative"):
        eda["ai_narrative"] = ai["narrative"]
    if ai.get("key_indicators"):
        eda["key_indicators"] = ai["key_indicators"]
    if ai.get("use_cases"):
        eda["use_cases"] = ai["use_cases"]
    if ai.get("recommendations"):
        eda["recommendations"] = ai["recommendations"]
    payload["eda_summary"] = eda

    dq["report"] = report
    profile["data_quality"] = dq
    profile["sensitivity"] = contract.sensitivity
    payload["dataset_profile"] = profile
    return payload
