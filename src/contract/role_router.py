"""Phase 3 — the Role-Aware Router.

A single place that answers "what may I legitimately *do* with this column?".
It works off either a compiled :class:`FieldContract` (Phase 2) or, before the
Phase 5 wiring lands, a Layer-2 ``EnrichedProfile`` — deriving the same
year/ratio/identifier facts heuristically so the analysis layers can consult
it today.

Rules enforced:
  * identifiers and years are never correlated as measures,
  * ratios/rates are never summed — a total ratio is recomputed from its
    component sums, not averaged,
  * a panel (long) frame is collapsed to its grain before cross-row maths,
    aggregating each column per its semantic aggregation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple

import pandas as pd

# Reuse the compiler's name heuristics so routing and compilation agree.
from src.contract.compiler import _RATIO_NAME_RE, _YEAR_NAME_RE

_RATE_TAGS = {"rate", "ratio", "percent", "percentage"}

_ALLOWED_AGG_BY_ROLE = {
    "identifier": ("count", "nunique"),
    "year": ("min", "max", "range", "nunique"),
    "datetime": ("min", "max", "range"),
    "ratio": ("mean", "median", "min", "max"),
    "numeric": ("sum", "mean", "min", "max", "median"),
    "categorical": ("count", "nunique"),
    "boolean": ("count", "sum", "mean"),
    "text": ("count", "nunique"),
}


@dataclass(frozen=True)
class _FieldView:
    name: str
    role: str
    is_identifier: bool
    is_year: bool
    is_ratio: bool
    aggregation: str  # 'additive' | 'rate' | 'none'


def field_view(obj: Any) -> _FieldView:
    """Normalise a FieldContract or an EnrichedProfile into a routing view."""
    name = getattr(obj, "name", "")
    role = getattr(obj, "role", "unknown")
    tags = set(getattr(obj, "semantic_tags", []) or [])

    # FieldContract carries the facts explicitly.
    if hasattr(obj, "is_identifier") and hasattr(obj, "aggregation"):
        return _FieldView(
            name=name,
            role=role,
            is_identifier=bool(getattr(obj, "is_identifier")),
            is_year=bool(getattr(obj, "is_year", False)),
            is_ratio=bool(getattr(obj, "is_ratio", False)),
            aggregation=getattr(obj, "aggregation", "none"),
        )

    # EnrichedProfile: derive heuristically (matches compiler intent).
    is_identifier = role == "identifier"
    is_year = role == "numeric" and bool(_YEAR_NAME_RE.search(name))
    is_ratio = role == "numeric" and (
        bool(tags & _RATE_TAGS) or bool(_RATIO_NAME_RE.search(name))
    )
    if "additive" in tags and not is_ratio:
        aggregation = "additive"
    elif is_ratio or "rate" in tags:
        aggregation = "rate"
    elif role == "boolean":
        aggregation = "additive"
    else:
        aggregation = "none"
    return _FieldView(name, role, is_identifier, is_year, is_ratio, aggregation)


def get_allowed_aggregations(obj: Any) -> Tuple[str, ...]:
    """Deterministic aggregations permitted for this column."""
    explicit = getattr(obj, "allowed_aggregations", None)
    if explicit:
        return tuple(explicit)
    v = field_view(obj)
    if v.is_year:
        return _ALLOWED_AGG_BY_ROLE["year"]
    if v.is_ratio:
        return _ALLOWED_AGG_BY_ROLE["ratio"]
    if v.is_identifier:
        return _ALLOWED_AGG_BY_ROLE["identifier"]
    return _ALLOWED_AGG_BY_ROLE.get(v.role, ("count",))


def is_correlatable(obj: Any) -> bool:
    """Only continuous numeric *measures* may enter correlation. Identifiers
    and year columns are numeric by storage but not by meaning."""
    v = field_view(obj)
    if v.role != "numeric":
        return False
    return not (v.is_identifier or v.is_year)


def can_sum(obj: Any) -> bool:
    """True only for additive measures. Ratios/years/ids must never be summed."""
    v = field_view(obj)
    if v.is_ratio or v.is_year or v.is_identifier:
        return False
    return "sum" in get_allowed_aggregations(obj)


def recompute_ratio(
    numerator: pd.Series | float,
    denominator: pd.Series | float,
) -> float:
    """A ratio aggregated across rows is sum(numerator) / sum(denominator) —
    never the mean of per-row ratios. Returns 0.0 on a zero denominator."""
    num = float(pd.Series(numerator).sum()) if not isinstance(numerator, (int, float)) else float(numerator)
    den = float(pd.Series(denominator).sum()) if not isinstance(denominator, (int, float)) else float(denominator)
    if den == 0:
        return 0.0
    return num / den


def _agg_func_for(view: _FieldView) -> str:
    if view.aggregation == "additive":
        return "sum"
    if view.is_year:
        return "max"
    if view.is_ratio or view.aggregation == "rate":
        return "mean"
    return "first" if view.role in ("identifier", "categorical", "text", "datetime") else "mean"


def collapse_to_grain(
    df: pd.DataFrame,
    grain: Sequence[str],
    profiles: Dict[str, Any],
) -> pd.DataFrame:
    """Collapse a panel/long frame to one row per grain tuple, aggregating each
    non-grain column per its semantic rule (additive→sum, rate→mean, …)."""
    grain = [g for g in grain if g in df.columns]
    if not grain:
        return df.reset_index(drop=True)
    agg_map: Dict[str, str] = {}
    for col in df.columns:
        if col in grain:
            continue
        prof = profiles.get(col)
        agg_map[col] = _agg_func_for(field_view(prof)) if prof is not None else "first"
    if not agg_map:
        return df.drop_duplicates(subset=grain).reset_index(drop=True)
    out = df.groupby(list(grain), dropna=False, as_index=False).agg(agg_map)
    return out.reset_index(drop=True)


def correlatable_columns(profiles: Dict[str, Any]) -> list[str]:
    """Helper for Layer 3: names of columns eligible for correlation."""
    return [name for name, p in profiles.items() if is_correlatable(p)]
