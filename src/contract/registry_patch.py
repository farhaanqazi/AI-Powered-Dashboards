"""Phase 7 — HITL registry override (pure logic).

A human corrects column roles/domains/sensitivity on the schema-review screen.
This module applies those overrides to the *persisted* dashboard payload
deterministically — produce a new **locked** contract (version + 1), re-derive
each field's aggregation/chart allow-lists, and drop any chart/KPI the new
roles no longer permit.

It deliberately does NOT re-profile or call the LLM, and does not need the raw
DataFrame (which isn't persisted — architectural invariant: no new storage
backend). Full L1/L3 re-aggregation from raw rows is out of scope here; the
override re-derives everything obtainable from the stored contract + payload.

Hard invariant: a human override can relax or correct roles but can NEVER
clear ``pii_blocked`` — fail-closed PII is not human-overridable.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List

from src.contract.models import DatasetContract, FieldContract
from src.contract.compiler import _AGG_ALLOW, _CHART_ALLOW


def _allow_key(role: str, domain: str, aggregation: str) -> str:
    if role == "year":
        return "year"
    if role == "ratio":
        return "ratio"
    if role == "numeric" and aggregation == "additive":
        return "numeric_additive"
    return role if role in _AGG_ALLOW else "text"


def _rederive_field(fc: FieldContract, ov: Dict[str, Any]) -> FieldContract:
    role = ov.get("role", fc.role)
    domain = ov.get("domain", fc.domain)
    sensitivity = ov.get("sensitivity", fc.sensitivity)
    # An override to identifier forces non-aggregable; ratio→rate; else keep.
    if role == "identifier":
        aggregation = "none"
    elif role == "ratio":
        aggregation = "rate"
    else:
        aggregation = fc.aggregation
    key = _allow_key(role, domain, aggregation)
    return fc.model_copy(
        update={
            "role": role,
            "domain": domain,
            "sensitivity": sensitivity,
            "aggregation": aggregation,
            "is_identifier": role == "identifier",
            "is_year": role == "year",
            "is_ratio": role == "ratio",
            "allowed_aggregations": _AGG_ALLOW.get(key, ("count",)),
            "allowed_charts": _CHART_ALLOW.get(key, ()),
        }
    )


def apply_registry_overrides(
    payload: Dict[str, Any], overrides: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Return a NEW payload dict with overrides applied and the contract
    re-locked. Raises ValueError if there is no contract to patch."""
    payload = copy.deepcopy(payload)
    profile = payload.get("dataset_profile") or {}
    raw_contract = profile.get("contract")
    if not raw_contract:
        raise ValueError("No compiled contract on this dashboard to override.")

    contract = DatasetContract.model_validate(raw_contract)
    by_name = {o["name"]: o for o in overrides if o.get("name")}

    new_fields: Dict[str, FieldContract] = {}
    for name, fc in contract.fields.items():
        new_fields[name] = _rederive_field(fc, by_name.get(name, {})) if name in by_name else fc

    any_sensitive = any(f.sensitivity == "sensitive" for f in new_fields.values())
    locked = contract.model_copy(
        update={
            "fields": new_fields,
            "locked": True,
            "version": contract.version + 1,
            # PII fail-closed: a human override never clears pii_blocked.
            "pii_blocked": contract.pii_blocked,
            "sensitivity": "sensitive"
            if (any_sensitive or contract.pii_blocked)
            else "public",
        }
    )

    # Thread the new contract back + sync the human-visible column roles.
    profile["contract"] = locked.model_dump(mode="json")
    profile["sensitivity"] = locked.sensitivity
    cols = profile.get("columns") or []
    for c in cols:
        nm = c.get("name")
        if nm in by_name and nm in locked.fields:
            c["role"] = locked.fields[nm].role
    profile["columns"] = cols

    # Drop charts/KPIs the corrected roles no longer permit.
    def _chart_ok(ch: Dict[str, Any]) -> bool:
        for fld in (ch.get("x_field"), ch.get("y_field")):
            fc = locked.fields.get(fld) if fld else None
            if fc is not None and fc.role == "identifier":
                return False  # identifiers are never plotted as a measure
        return True

    for key in ("all_charts", "charts"):
        if isinstance(payload.get(key), list):
            payload[key] = [c for c in payload[key] if _chart_ok(c)]

    def _kpi_ok(k: Dict[str, Any]) -> bool:
        prov = (k.get("provenance") or {}).get("source", "")
        if prov.startswith("column:"):
            fc = locked.fields.get(prov.split(":", 1)[1])
            if fc is not None and fc.role == "identifier":
                return False
        return True

    if isinstance(payload.get("kpis"), list):
        payload["kpis"] = [k for k in payload["kpis"] if _kpi_ok(k)]

    # Refresh the data-quality verdict: human-approved & locked ⇒ ok, unless
    # still PII-blocked (which review cannot clear).
    dq = (profile.get("data_quality") or {})
    report = dq.get("report") or {}
    report["status"] = "blocked" if locked.pii_blocked else "ok"
    report["auto_accepted"] = False
    report["human_reviewed"] = True
    dq["report"] = report
    profile["data_quality"] = dq
    payload["dataset_profile"] = profile
    return payload
