"""Fenced-arm ingest: drive the real engine end-to-end and return its payload.

Uses the engine's own delimiter-sniffing loader (`build_dashboard_from_file_*`),
NOT `pd.read_csv`, so `;`-delimited and Excel files parse correctly — the system
is tested, not a pandas stand-in.

Two gate verdicts are recorded and kept SEPARATE:
  - `would_schema_halt`  — auto-acceptance failed (low confidence / no grain).
    This is the MEANINGFUL split for the Trust include/exclude reporting.
  - `pii_gate_tripped`    — the regex PII fallback flagged a column. On the
    curated (real-PII-removed) set these are FALSE POSITIVES; when `force_consent`
    is on we re-run the AI layer via the engine's own `apply_ai_consent` so the
    fenced (tokened) path actually produces output. Logged as noise, never mixed
    into the schema-review split.

`apply_ai_consent` reads the cleaned frame from the in-process df_cache, so this
must run in the same process that built the dashboard.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

_DATASETS = os.path.join("benchmark", "datasets")
_CAPPED = os.path.join(_DATASETS, "capped")


def resolve_path(filename: str) -> str:
    """Single source of truth for which file to read. Prefers the deterministic
    capped copy (benchmark/datasets/capped/<file>) when it exists, so both arms
    AND ground truth always run on the identical frame."""
    capped = os.path.join(_CAPPED, filename)
    return capped if os.path.exists(capped) else os.path.join(_DATASETS, filename)


def load_fenced(path: str, filename: Optional[str] = None,
                force_consent: bool = True) -> Dict[str, Any]:
    """Run one table through the fenced engine. Returns a result dict; never raises."""
    # Lazy imports: env must be configured before `src` is first imported.
    from src.core.pipeline import build_dashboard_from_file_generator
    from src.core.state_payload import state_to_payload

    filename = filename or os.path.basename(path)
    path = resolve_path(filename)  # prefer the capped copy when present
    try:
        with open(path, "rb") as fh:
            state = None
            for ev in build_dashboard_from_file_generator(fh, original_filename=filename):
                phase = ev.get("phase")
                if phase == "done":
                    state = ev.get("state")
                elif phase == "error":
                    return {"ok": False, "filename": filename, "error": ev.get("message")}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "filename": filename, "error": f"{type(e).__name__}: {e}"}

    if state is None:
        return {"ok": False, "filename": filename, "error": "pipeline returned no state"}

    payload = state_to_payload(state, filename)
    rep = (((payload.get("dataset_profile") or {}).get("data_quality") or {}).get("report") or {})
    eda = payload.get("eda_summary") or {}

    gate = {
        # Meaningful split: would the schema-review gate have halted this table?
        "would_schema_halt": rep.get("auto_accepted") is False,
        "auto_accepted": rep.get("auto_accepted"),
        "schema_status": eda.get("status"),
        # Noise: regex PII false positive (real PII already pruned from the set).
        "pii_gate_tripped": bool(rep.get("ai_consent_required")),
        "pii_columns": rep.get("pii_columns"),
    }

    consented = False
    if force_consent and rep.get("ai_consent_required"):
        try:
            from src.contract.ai_consent import apply_ai_consent
            payload = apply_ai_consent(payload)
            consented = True
        except Exception as e:  # noqa: BLE001 - consent is best-effort
            gate["consent_error"] = f"{type(e).__name__}: {e}"

    kpis = payload.get("kpis") or []
    return {
        "ok": True,
        "filename": filename,
        "payload": payload,
        "gate": gate,
        "consented": consented,
        "n_rows": (payload.get("dataset_profile") or {}).get("n_rows"),
        "n_cols": (payload.get("dataset_profile") or {}).get("n_cols"),
        "n_kpis": len(kpis),
        "n_kpis_tokened": sum(1 for k in kpis if k.get("provenance")),
    }
