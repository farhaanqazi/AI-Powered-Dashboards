"""Deterministic, contract-guarded tools the Ask agent may invoke.

Each tool computes numbers from the cleaned DataFrame and attaches a
provenance token. Contract guards are enforced here, not trusted to the LLM:
a ratio is never summed, only real measures correlate, only dimensions group.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from src.contract.role_router import can_sum, is_correlatable

_NUMERIC_METRICS = {"mean", "median", "min", "max", "std", "count", "sum"}
_AGG_FUNCS = {"sum", "mean", "median", "min", "max", "count"}
_OPS = {
    "==": lambda s, v: s == v,
    "!=": lambda s, v: s != v,
    ">": lambda s, v: s > v,
    ">=": lambda s, v: s >= v,
    "<": lambda s, v: s < v,
    "<=": lambda s, v: s <= v,
}


class ToolError(ValueError):
    """Invalid tool params or a contract-guard violation."""


def _field(contract: Any, name: str):
    fields = getattr(contract, "fields", {}) or {}
    fc = fields.get(name)
    if fc is None:
        raise ToolError(f"Unknown column '{name}'")
    return fc


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").dropna()


def column_stat(df, contract, params) -> Dict[str, Any]:
    col = params.get("column")
    metric = str(params.get("metric", "mean")).lower()
    if metric not in _NUMERIC_METRICS:
        raise ToolError(f"Unsupported metric '{metric}'")
    fc = _field(contract, col)
    if metric == "sum" and not can_sum(fc):
        raise ToolError(
            f"'{col}' is a {getattr(fc, 'role', '?')} — summing it is "
            "semantically invalid (contract guard)."
        )
    if metric == "count":
        value = int(df[col].notna().sum())
    else:
        s = _num(df, col)
        if s.empty:
            raise ToolError(f"'{col}' has no numeric values to {metric}.")
        value = float(getattr(s, metric)())
    return {
        "result": {"column": col, "metric": metric, "value": value},
        "provenance": {"source": f"column:{col}", "metric": metric,
                       "token": f"L1.{col}.{metric}"},
        "summary": f"{metric} of {col} = {value:,.4g}"
        if metric != "count" else f"{col}: {value} non-null rows",
    }


def aggregate(df, contract, params) -> Dict[str, Any]:
    group_by = params.get("group_by")
    value = params.get("value")
    agg = str(params.get("agg", "mean")).lower()
    if agg not in _AGG_FUNCS:
        raise ToolError(f"Unsupported aggregation '{agg}'")
    gfc = _field(contract, group_by)
    if getattr(gfc, "role", "") not in ("categorical", "identifier", "year",
                                          "boolean", "datetime"):
        raise ToolError(f"'{group_by}' is not a dimension to group by.")
    if agg == "count":
        ser = df.groupby(group_by, observed=True).size()
    else:
        vfc = _field(contract, value)
        if agg == "sum" and not can_sum(vfc):
            raise ToolError(
                f"'{value}' cannot be summed (contract guard)."
            )
        tmp = df[[group_by]].copy()
        tmp["_v"] = pd.to_numeric(df[value], errors="coerce")
        ser = tmp.groupby(group_by, observed=True)["_v"].agg(agg)
    ser = ser.sort_values(ascending=False).head(20)
    rows = [{"group": str(k), "value": float(v)} for k, v in ser.items()]
    return {
        "result": {"group_by": group_by, "agg": agg, "value": value,
                   "rows": rows},
        "provenance": {"source": f"agg:{group_by}",
                       "token": f"agg.{group_by}.{agg}"
                                + (f"({value})" if agg != "count" else "")},
        "summary": f"{agg} by {group_by}: "
                   + ", ".join(f"{r['group']}={r['value']:,.4g}"
                               for r in rows[:5]),
    }


def top_categories(df, contract, params) -> Dict[str, Any]:
    col = params.get("column")
    n = int(params.get("n", 5))
    _field(contract, col)
    vc = df[col].astype(str).value_counts().head(max(1, min(n, 25)))
    rows = [{"value": k, "count": int(v)} for k, v in vc.items()]
    return {
        "result": {"column": col, "top": rows},
        "provenance": {"source": f"column:{col}",
                       "token": f"L1.{col}.top_categories"},
        "summary": f"Top {col}: "
                   + ", ".join(f"{r['value']} ({r['count']})" for r in rows),
    }


def correlation(df, contract, params) -> Dict[str, Any]:
    a, b = params.get("a"), params.get("b")
    fa, fb = _field(contract, a), _field(contract, b)
    if not (is_correlatable(fa) and is_correlatable(fb)):
        raise ToolError(
            f"Correlation needs two real measures; '{a}'/'{b}' are not both "
            "correlatable (ids/years excluded by contract)."
        )
    pair = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 3:
        raise ToolError("Not enough overlapping numeric rows to correlate.")
    r = float(pair[a].corr(pair[b]))
    return {
        "result": {"a": a, "b": b, "pearson": r, "n": int(len(pair))},
        "provenance": {"source": f"corr:{a}|{b}",
                       "token": f"L3.correlation.{a}|{b}"},
        "summary": f"Pearson r({a}, {b}) = {r:.4f} (n={len(pair)})",
    }


def filter_count(df, contract, params) -> Dict[str, Any]:
    col = params.get("column")
    op = params.get("op", "==")
    val = params.get("value")
    _field(contract, col)
    if op not in _OPS:
        raise ToolError(f"Unsupported operator '{op}'")
    series = df[col]
    if op in (">", ">=", "<", "<="):
        series = pd.to_numeric(series, errors="coerce")
        try:
            val = float(val)
        except (TypeError, ValueError):
            raise ToolError("Numeric comparison needs a numeric value.")
    else:
        series = series.astype(str)
        val = str(val)
    count = int(_OPS[op](series, val).sum())
    return {
        "result": {"column": col, "op": op, "value": val, "count": count},
        "provenance": {"source": f"filter:{col}",
                       "token": f"filter.{col}{op}{val}"},
        "summary": f"Rows where {col} {op} {val}: {count}",
    }


TOOLS = {
    "column_stat": column_stat,
    "aggregate": aggregate,
    "top_categories": top_categories,
    "correlation": correlation,
    "filter_count": filter_count,
}

# Compact catalog given to the planner LLM (no data, just capability).
TOOL_CATALOG = {
    "column_stat": {"column": "str", "metric": sorted(_NUMERIC_METRICS)},
    "aggregate": {"group_by": "dimension", "value": "measure",
                  "agg": sorted(_AGG_FUNCS)},
    "top_categories": {"column": "str", "n": "int (<=25)"},
    "correlation": {"a": "measure", "b": "measure"},
    "filter_count": {"column": "str", "op": sorted(_OPS),
                     "value": "str|number"},
}
