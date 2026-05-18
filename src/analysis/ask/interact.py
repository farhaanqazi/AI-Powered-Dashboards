"""Phase 14 S14.2 — structured, non-LLM interaction engine.

A dashboard interaction (filter / narrow / drill / recompute) is a deterministic
*spec*, never a natural-language question — so there is **no LLM in this path at
all**. The UI maps a click straight to one entry of the Phase 11 tool catalogue
plus an optional filter set; the backend executes it deterministically over the
(now durable) cleaned frame and returns provenance-tagged numbers.

Trust rule preserved: every result keeps the tool's own provenance token AND a
*derived* token chained over the applied filters, so a filtered view stays
auditable. The AI never computes here.

Results are memoised in-process keyed on sha256(schema_fingerprint +
canonical(spec)) with LRU eviction, so repeated filter states are instant and
single-container memory stays bounded.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Optional

import pandas as pd

from src import config
from src.analysis.ask.tools import TOOLS, ToolError

_FILTER_OPS = {"eq", "neq", "in", "nin", "gt", "gte", "lt", "lte", "between"}
_NUMERIC_OPS = {"gt", "gte", "lt", "lte", "between"}


def _known_column(contract: Any, name: str):
    fields = getattr(contract, "fields", {}) or {}
    if name not in fields:
        raise ToolError(f"Unknown column '{name}'")
    return fields[name]


def _coerce_num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ToolError("Numeric filter needs a numeric value.")


def apply_filters(
    df: pd.DataFrame, contract: Any, filters: Optional[List[Dict[str, Any]]]
) -> pd.DataFrame:
    """Deterministically narrow ``df`` by ``filters``. Contract-guarded:
    every column must exist in the frozen contract; unknown columns / ops are
    rejected (raised as ToolError, never silently ignored)."""
    if not filters:
        return df
    if len(filters) > config.INTERACT_MAX_FILTERS:
        raise ToolError(
            f"Too many filters ({len(filters)} > "
            f"{config.INTERACT_MAX_FILTERS})."
        )
    out = df
    for f in filters:
        col = (f or {}).get("column")
        op = str((f or {}).get("op", "eq")).lower()
        val = (f or {}).get("value")
        _known_column(contract, col)
        if op not in _FILTER_OPS:
            raise ToolError(f"Unsupported filter op '{op}'")
        series = out[col]
        if op in _NUMERIC_OPS:
            series = pd.to_numeric(series, errors="coerce")
            if op == "between":
                if not isinstance(val, (list, tuple)) or len(val) != 2:
                    raise ToolError("'between' needs [low, high].")
                lo, hi = _coerce_num(val[0]), _coerce_num(val[1])
                mask = series.between(lo, hi)
            else:
                num = _coerce_num(val)
                mask = {
                    "gt": series > num, "gte": series >= num,
                    "lt": series < num, "lte": series <= num,
                }[op]
        elif op in ("in", "nin"):
            vals = val if isinstance(val, (list, tuple)) else [val]
            vals = [str(v) for v in vals]
            isin = out[col].astype(str).isin(vals)
            mask = isin if op == "in" else ~isin
        else:  # eq / neq
            s = out[col].astype(str)
            v = str(val)
            mask = (s == v) if op == "eq" else (s != v)
        out = out[mask]
    return out


def _canonical(spec: Dict[str, Any]) -> str:
    """Order-independent canonical JSON of an interaction spec (so logically
    identical filter states hash to the same cache key)."""
    calc = spec.get("calculation")
    params = spec.get("params") or {}
    filters = spec.get("filters") or []
    norm_filters = sorted(
        ({"column": f.get("column"),
          "op": str(f.get("op", "eq")).lower(),
          "value": f.get("value")} for f in filters),
        key=lambda d: json.dumps(d, sort_keys=True, default=str),
    )
    return json.dumps(
        {"calculation": calc, "params": params, "filters": norm_filters},
        sort_keys=True, default=str,
    )


def _derived_token(base_token: str, canonical: str) -> str:
    h = hashlib.sha256(f"{base_token}|{canonical}".encode()).hexdigest()[:16]
    return f"derived:{base_token}#{h}"


def run_interaction(
    df: pd.DataFrame, contract: Any, spec: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute one structured interaction. Never raises: an invalid spec or a
    contract-guard violation comes back as ``status='error'``."""
    if not config.INTERACT_ENABLED:
        return {"status": "disabled", "result": None}
    calc = (spec or {}).get("calculation")
    params = (spec or {}).get("params") or {}
    filters = (spec or {}).get("filters") or []
    fn = TOOLS.get(calc)
    if fn is None:
        return {"status": "error",
                "error": f"Unknown calculation '{calc}'",
                "available": sorted(TOOLS.keys())}
    try:
        filtered = apply_filters(df, contract, filters)
    except ToolError as e:
        return {"status": "error", "error": str(e)}
    rows_after = int(len(filtered))
    if rows_after == 0:
        return {"status": "empty", "error": "No rows match these filters.",
                "filters": filters, "rows_after": 0}
    try:
        res = fn(filtered, contract, params)
    except ToolError as e:
        return {"status": "error", "error": str(e), "filters": filters}
    except Exception:  # noqa: BLE001 - never break the request
        return {"status": "error",
                "error": "Calculation failed for this selection.",
                "filters": filters}

    base = (res.get("provenance") or {}).get("token", f"{calc}")
    canonical = _canonical({"calculation": calc, "params": params,
                            "filters": filters})
    res["provenance"] = {
        **(res.get("provenance") or {}),
        "verified": True,
        "filtered": bool(filters),
        "rows_after": rows_after,
        "derived_token": _derived_token(base, canonical),
    }
    return {
        "status": "ok",
        "calculation": calc,
        "params": params,
        "filters": filters,
        "rows_after": rows_after,
        "result": res.get("result"),
        "summary": res.get("summary"),
        "provenance": res["provenance"],
        "numbers_traceable": True,
    }


# --- in-process LRU result cache (instant repeated filter states) -----------

_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


def _cache_key(fingerprint: str, spec: Dict[str, Any]) -> str:
    return hashlib.sha256(
        f"{fingerprint}|{_canonical(spec)}".encode()
    ).hexdigest()


def run_interaction_cached(
    fingerprint: str, df: pd.DataFrame, contract: Any, spec: Dict[str, Any]
) -> Dict[str, Any]:
    """``run_interaction`` with sha256(fingerprint+canonical(spec)) LRU memo.
    Only successful results are cached; errors/empties always re-run."""
    key = _cache_key(fingerprint, spec)
    hit = _cache.get(key)
    if hit is not None:
        _cache.move_to_end(key)
        return {**deepcopy(hit), "cached": True}
    out = run_interaction(df, contract, spec)
    if out.get("status") == "ok":
        _cache[key] = deepcopy(out)
        _cache.move_to_end(key)
        while len(_cache) > config.INTERACT_RESULT_CACHE_MAX_ENTRIES:
            _cache.popitem(last=False)
    return {**out, "cached": False}


def reset_interaction_cache_for_tests() -> None:
    _cache.clear()
