"""LLM Analyst — the "real AI" interpretation layer.

Hard rule: the LLM never computes or supplies a number. It only (a) selects
which columns/relationships matter, (b) writes business-grade labels, titles
and narrative, and (c) chooses chart types. Every numeric KPI value is filled
in by THIS module from the deterministic Layer 1-3 / EDA output. That is the
guarantee that separates this from a hallucinating demo: figures are
ground-truth; the model only narrates and selects over them.

If the key is missing, the SDK is absent, the call errors, or the model
returns something unusable, every path falls back to the heuristic Layer 4
result that the caller already computed. The app must never break because of
this layer.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from src import config
from src.analysis.data_structures import EnrichedProfile, RelationalInsight

try:
    from src.logger import get_logger
    logger = get_logger(__name__)
except Exception:  # pragma: no cover - logging is best-effort
    import logging
    logger = logging.getLogger(__name__)

# Intents the renderer actually understands (src/viz/plotly_renderer.py).
# Anything else is silently dropped downstream, so the model is constrained
# to exactly this set and we force chart_type from intent.
_INTENT_CHART_TYPE = {
    "category_count": "bar",
    "distribution": "histogram",
    "histogram": "histogram",
    "category_summary": "bar",
    "group_comparison": "bar",
    "time_series": "line",
    "scatter": "scatter",
}

_STAT_METRICS = {"sum", "mean", "median", "min", "max", "std", "variance", "count"}


def _f(value: Any) -> Any:
    """Round floats for a compact prompt; pass everything else through."""
    if isinstance(value, float):
        return round(value, 4)
    return value


def _ground_truth(
    enriched_profiles: Dict[str, EnrichedProfile],
    relational_insights: List[RelationalInsight],
    eda_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """The ONLY numbers the model is allowed to reason about."""
    columns = []
    for name, p in list(enriched_profiles.items())[: config.MAX_COLS]:
        stats = {k: _f(v) for k, v in (p.stats or {}).items()}
        columns.append(
            {
                "name": name,
                "role": p.role,
                "semantic_tags": p.semantic_tags or [],
                "null_count": p.null_count,
                "unique_count": p.unique_count,
                "stats": stats,
                "top_categories": (p.top_categories or [])[:5],
            }
        )

    correlations = []
    for ins in relational_insights:
        if ins.type == "correlation" and len(ins.columns) == 2:
            d = ins.details or {}
            correlations.append(
                {
                    "columns": ins.columns,
                    "coefficient": _f(d.get("correlation_coefficient")),
                    "p_value": _f(d.get("p_value")),
                    "strength": d.get("strength"),
                }
            )

    return {
        "column_count": len(enriched_profiles),
        "columns": columns,
        "correlations": correlations[:15],
        "critical_totals": (eda_summary or {}).get("critical_totals", {}),
    }


_SYSTEM_PROMPT = (
    "You are a senior data analyst writing an executive dashboard brief. "
    "You are given ONLY pre-computed, ground-truth statistics about a dataset. "
    "STRICT RULES:\n"
    "1. NEVER state, compute, estimate, or invent any number. Do not put numeric "
    "values in your output at all — the system fills those in from ground truth.\n"
    "2. Only reference column names that appear in the provided data.\n"
    "3. Choose KPIs and charts that a business stakeholder would actually care "
    "about; explain WHY each matters in plain language. Aim for 6-10 KPIs and "
    "10-18 charts covering distributions, category breakdowns, trends and "
    "relationships.\n"
    "4. Charts: intent must be one of "
    "category_count, distribution, category_summary, group_comparison, "
    "time_series, scatter. Use real column names for x_field/y_field.\n"
    "Return ONLY valid JSON matching the requested schema."
)

_SCHEMA_HINT = {
    "narrative": "2-4 sentence executive summary of what this dataset is about and the single most important takeaway (NO numbers).",
    "kpis": [
        {
            "column": "exact column name OR omit if correlation",
            "metric": "one of: sum, mean, median, min, max, std, count, top_category",
            "correlation": ["colA", "colB"],
            "label": "business-meaningful KPI name (NO numbers)",
            "why": "one short clause on why this is a headline metric",
        }
    ],
    "charts": [
        {
            "intent": "one of the allowed intents",
            "x_field": "real column name",
            "y_field": "real column name or null",
            "agg_func": "sum or mean or null",
            "title": "specific, contextual chart title (NO numbers)",
            "rationale": "why this chart is worth showing",
        }
    ],
}


def _kpi_value(profile: EnrichedProfile, metric: str) -> str:
    """Deterministically format a KPI value FROM GROUND TRUTH. The model chose
    the column and metric; the number itself never comes from the model."""
    stats = profile.stats or {}
    if metric == "top_category" or profile.role in ("categorical", "text"):
        if profile.top_categories:
            top = profile.top_categories[0]
            return f"Top: '{top.get('value')}' ({top.get('count')})"
        return "N/A"
    if metric == "count":
        return f"{stats.get('count', 'N/A')}"
    if metric in _STAT_METRICS and stats.get(metric) is not None:
        return f"{stats.get(metric):,.2f}"
    if stats.get("mean") is not None:
        return f"{stats.get('mean'):,.2f} (±{stats.get('std', 0.0):,.2f})"
    return "N/A"


def _build_kpis(
    raw: List[Dict[str, Any]],
    enriched_profiles: Dict[str, EnrichedProfile],
    relational_insights: List[RelationalInsight],
) -> List[Dict[str, Any]]:
    corr_lookup = {
        frozenset(i.columns): i.details
        for i in relational_insights
        if i.type == "correlation" and len(i.columns) == 2
    }
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw):
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        score = round(1.0 - idx * 0.01, 4)  # preserve model ordering downstream

        corr = item.get("correlation")
        if corr and len(corr) == 2 and frozenset(corr) in corr_lookup:
            coeff = (corr_lookup[frozenset(corr)] or {}).get("correlation_coefficient")
            if coeff is None:
                continue
            out.append({"label": label, "value": f"{coeff:.2f}",
                        "type": "correlation", "score": score})
            continue

        col = item.get("column")
        if col in enriched_profiles:
            profile = enriched_profiles[col]
            metric = str(item.get("metric") or "mean").lower()
            out.append({
                "label": label,
                "value": _kpi_value(profile, metric),
                "type": profile.role,
                "score": score,
            })
    return out


def _build_charts(
    raw: List[Dict[str, Any]],
    enriched_profiles: Dict[str, EnrichedProfile],
) -> List[Dict[str, Any]]:
    names = set(enriched_profiles.keys())
    out: List[Dict[str, Any]] = []
    seen = set()
    for idx, item in enumerate(raw):
        intent = str(item.get("intent") or "").strip()
        if intent not in _INTENT_CHART_TYPE:
            continue
        x = item.get("x_field")
        if x not in names:
            continue
        y = item.get("y_field")
        if y is not None and y not in names:
            y = None
        # Intents that are meaningless without a y_field: drop if missing.
        if intent in ("category_summary", "group_comparison", "time_series",
                      "scatter") and y is None:
            continue
        sig = (intent, x, y)
        if sig in seen:
            continue
        seen.add(sig)

        agg = item.get("agg_func")
        agg = agg if agg in ("sum", "mean") else None
        spec = {
            "id": f"ai_{intent}_{x}_{y or ''}_{idx}",
            "title": str(item.get("title") or f"{x}").strip(),
            "chart_type": _INTENT_CHART_TYPE[intent],
            "intent": intent,
            "x_field": x,
            "priority": idx,
            "rationale": str(item.get("rationale") or "").strip(),
            "dimensions": {"width": "responsive", "height": "400px"},
            "responsive": True,
            "ai_generated": True,
        }
        if y is not None:
            spec["y_field"] = y
        if intent == "time_series":
            spec["agg_func"] = agg or "mean"
        elif agg is not None:
            spec["agg_func"] = agg
        out.append(spec)
    return out


def run_ai_analyst(
    enriched_profiles: Dict[str, EnrichedProfile],
    relational_insights: List[RelationalInsight],
    eda_summary: Dict[str, Any],
    *,
    fallback_kpis: List[Dict[str, Any]],
    fallback_specs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Returns {"kpis", "chart_specs", "narrative"}.

    Never raises. Any failure returns the caller's heuristic fallback so the
    pipeline behaves exactly as before the AI layer existed.
    """
    fallback = {"kpis": fallback_kpis, "chart_specs": fallback_specs, "narrative": ""}

    if not config.AI_ANALYST_ENABLED or not config.GROQ_API_KEY:
        return fallback
    if not enriched_profiles:
        return fallback

    try:
        from groq import Groq
    except Exception as e:  # pragma: no cover - SDK optional
        logger.warning(f"AI analyst disabled: groq SDK unavailable ({e})")
        return fallback

    try:
        payload = _ground_truth(enriched_profiles, relational_insights, eda_summary)
        client = Groq(api_key=config.GROQ_API_KEY, timeout=config.GROQ_TIMEOUT_SECONDS)
        resp = client.chat.completions.create(
            model=config.GROQ_MODEL,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "GROUND-TRUTH DATA (the only facts you may use):\n"
                        + json.dumps(payload, default=str)
                        + "\n\nReturn JSON with exactly these keys: "
                        + json.dumps(_SCHEMA_HINT)
                    ),
                },
            ],
        )
        content = resp.choices[0].message.content
        parsed = json.loads(content)
    except Exception as e:
        logger.warning(f"AI analyst call failed, using heuristic fallback: {e}")
        return fallback

    try:
        kpis = _build_kpis(
            parsed.get("kpis") or [], enriched_profiles, relational_insights
        )
        charts = _build_charts(parsed.get("charts") or [], enriched_profiles)
        narrative = str(parsed.get("narrative") or "").strip()
    except Exception as e:
        logger.warning(f"AI analyst output rejected, using heuristic fallback: {e}")
        return fallback

    # Merge, never shrink: AI-labelled items come first (better titles), then
    # every heuristic item the model didn't already cover is appended. This
    # keeps the full chart breadth the heuristic produced (20+) while the AI
    # only improves ordering and naming on top.
    def _sig(spec):
        return (spec.get("intent"), spec.get("x_field"), spec.get("y_field"))

    ai_sigs = {_sig(c) for c in charts}
    charts += [s for s in fallback_specs if _sig(s) not in ai_sigs]

    ai_labels = {k["label"] for k in kpis}
    kpis += [k for k in fallback_kpis if k.get("label") not in ai_labels]

    if not kpis and not charts:
        return fallback

    logger.info(
        f"AI analyst: {len(kpis)} KPIs, {len(charts)} charts, "
        f"narrative={'yes' if narrative else 'no'}"
    )
    return {"kpis": kpis, "chart_specs": charts, "narrative": narrative}
