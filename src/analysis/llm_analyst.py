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


def _sensitive_columns(contract: Any) -> set:
    """Names of columns the contract marks sensitive / PII-bearing."""
    if contract is None:
        return set()
    fields = getattr(contract, "fields", {}) or {}
    out = set()
    for name, fc in fields.items():
        if getattr(fc, "sensitivity", "public") == "sensitive" or getattr(
            fc, "pii_entities", ()
        ):
            out.add(name)
    return out


def _ground_truth(
    enriched_profiles: Dict[str, EnrichedProfile],
    relational_insights: List[RelationalInsight],
    eda_summary: Dict[str, Any],
    contract: Any = None,
    redact_sensitive: bool = True,
) -> Dict[str, Any]:
    """The ONLY numbers the model is allowed to reason about.

    Contract-validated: by default sensitive/PII columns never expose raw
    category values (``top_categories`` can leak emails, names, account
    numbers). When the user has explicitly consented to AI on a PII dataset,
    ``redact_sensitive=False`` sends everything (full consent — the user's
    prerogative and responsibility).
    """
    sensitive = set() if not redact_sensitive else _sensitive_columns(contract)
    columns = []
    for name, p in list(enriched_profiles.items())[: config.MAX_COLS]:
        stats = {k: _f(v) for k, v in (p.stats or {}).items()}
        is_sensitive = name in sensitive
        columns.append(
            {
                "name": name,
                "role": p.role,
                "semantic_tags": p.semantic_tags or [],
                "null_count": p.null_count,
                "unique_count": p.unique_count,
                # Drop count-style numeric stats for sensitive cols too:
                # min/max of an SSN/account number is still identifying.
                "stats": {} if is_sensitive else stats,
                "top_categories": [] if is_sensitive else (p.top_categories or [])[:5],
                "redacted": is_sensitive,
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
    "5. Also produce 3-5 business use_cases and 3-6 recommendations. These are "
    "qualitative strategy — no statistics, no fabricated figures. key_inputs "
    "must be real column names.\n"
    "6. Also produce 6-12 key_indicators: the metrics a stakeholder would "
    "actually track. NEVER pick an identifier/ID column (averaging an ID is "
    "meaningless). Humanize names (spaces, no underscores). No numbers in your "
    "text — the system fills exact figures from ground truth.\n"
    "4. Charts: intent must be one of "
    "category_count, distribution, category_summary, group_comparison, "
    "time_series, scatter. Use real column names for x_field/y_field.\n"
    "Return ONLY valid JSON matching the requested schema."
)

_SCHEMA_HINT = {
    "narrative": "2-4 sentence executive summary of what this dataset is about and the single most important takeaway (NO numbers).",
    "key_indicators": [
        {
            "column": "real column name (NEVER an identifier/ID column)",
            "metric": "one of: mean, median, sum, min, max, std, top_category",
            "indicator": "humanized headline metric name, spaces not underscores (NO numbers)",
            "description": "one short business sentence on what it tells you (NO numbers)",
        }
    ],
    "use_cases": [
        {
            "use_case": "short business use-case title",
            "description": "what analysis to do and the value it unlocks (NO numbers)",
            "key_inputs": ["real column names involved"],
        }
    ],
    "recommendations": [
        {
            "title": "short recommended next action",
            "description": "concrete analytical or business recommendation (NO numbers)",
            "priority": "high or medium or low",
        }
    ],
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
            "x_label": "human-readable x-axis label (NO numbers)",
            "y_label": "human-readable y-axis label (NO numbers)",
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
            a, b = list(corr)[0], list(corr)[1]
            out.append({"label": label, "value": f"{coeff:.2f}",
                        "type": "correlation", "score": score,
                        "_src": ("corr", frozenset(corr)),
                        # Provenance: this number traces to the L3 correlation.
                        "provenance": {
                            "source": f"corr:{a}|{b}",
                            "metric": "correlation",
                            "token": f"L3.correlation.{a}|{b}",
                        }})
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
                "_src": ("col", col),
                # Provenance: every figure traces to a deterministic L1/L2 stat.
                "provenance": {
                    "source": f"column:{col}",
                    "metric": metric,
                    "token": f"L1.{col}.{metric}",
                },
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
            "x_label": str(item.get("x_label") or "").strip(),
            "y_label": str(item.get("y_label") or "").strip(),
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


def _build_key_indicators(
    raw: List[Dict[str, Any]], enriched_profiles: Dict[str, EnrichedProfile]
) -> List[Dict[str, Any]]:
    """AI picks which columns matter and writes the prose; the figure itself
    is computed here from ground truth. Identifiers are hard-excluded — an
    averaged ID is the nonsense we are removing."""
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in raw[:14]:
        col = item.get("column")
        if col not in enriched_profiles or col in seen:
            continue
        profile = enriched_profiles[col]
        if profile.role == "identifier":
            continue
        indicator = str(item.get("indicator") or "").strip()
        desc = str(item.get("description") or "").strip()
        if not indicator:
            continue
        metric = str(item.get("metric") or "mean").lower()
        value = _kpi_value(profile, metric)
        if value == "N/A":
            continue
        seen.add(col)
        out.append({
            "indicator": indicator,
            "description": f"{desc} ({value})" if desc else str(value),
            "value": value,
            "type": metric,
        })
    return out


def _build_eda(
    parsed: Dict[str, Any], enriched_profiles: Dict[str, EnrichedProfile]
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Rebuild key_indicators / use_cases / recommendations in the exact
    heuristic shape the frontend expects, keeping only real, non-identifier
    columns."""
    names = set(enriched_profiles.keys())
    key_indicators = _build_key_indicators(
        parsed.get("key_indicators") or [], enriched_profiles
    )
    use_cases: List[Dict[str, Any]] = []
    for u in (parsed.get("use_cases") or [])[:6]:
        title = str(u.get("use_case") or "").strip()
        desc = str(u.get("description") or "").strip()
        if not title or not desc:
            continue
        key_inputs = [c for c in (u.get("key_inputs") or []) if c in names][:6]
        use_cases.append(
            {"use_case": title, "description": desc, "key_inputs": key_inputs}
        )

    recommendations: List[Dict[str, Any]] = []
    for r in (parsed.get("recommendations") or [])[:8]:
        title = str(r.get("title") or "").strip()
        desc = str(r.get("description") or "").strip()
        if not title or not desc:
            continue
        priority = str(r.get("priority") or "medium").lower()
        if priority not in ("high", "medium", "low"):
            priority = "medium"
        recommendations.append(
            {"title": title, "description": desc, "priority": priority}
        )
    return key_indicators, use_cases, recommendations


def run_ai_analyst(
    enriched_profiles: Dict[str, EnrichedProfile],
    relational_insights: List[RelationalInsight],
    eda_summary: Dict[str, Any],
    *,
    fallback_kpis: List[Dict[str, Any]],
    fallback_specs: List[Dict[str, Any]],
    contract: Any = None,
    redact_sensitive: bool = True,
) -> Dict[str, Any]:
    """Returns {"kpis", "chart_specs", "narrative"}.

    Never raises. Any failure returns the caller's heuristic fallback so the
    pipeline behaves exactly as before the AI layer existed.
    """
    fallback = {"kpis": fallback_kpis, "chart_specs": fallback_specs,
                "narrative": "", "key_indicators": [],
                "use_cases": [], "recommendations": []}

    if not config.AI_ANALYST_ENABLED:
        return fallback
    if not enriched_profiles:
        return fallback

    from src.analysis.llm import get_llm_provider

    provider = get_llm_provider()
    if not provider.available():
        logger.warning("AI analyst disabled: no usable LLM provider")
        return fallback

    try:
        payload = _ground_truth(
            enriched_profiles, relational_insights, eda_summary, contract,
            redact_sensitive=redact_sensitive,
        )
        parsed = provider.complete_json(
            system=_SYSTEM_PROMPT,
            user=(
                "GROUND-TRUTH DATA (the only facts you may use):\n"
                + json.dumps(payload, default=str)
                + "\n\nReturn JSON with exactly these keys: "
                + json.dumps(_SCHEMA_HINT)
            ),
            temperature=0.2,
        )
    except Exception as e:
        logger.warning(f"AI analyst call failed, using heuristic fallback: {e}")
        return fallback

    try:
        kpis = _build_kpis(
            parsed.get("kpis") or [], enriched_profiles, relational_insights
        )
        charts = _build_charts(parsed.get("charts") or [], enriched_profiles)
        narrative = str(parsed.get("narrative") or "").strip()
        ai_kis, ai_use_cases, ai_recs = _build_eda(parsed, enriched_profiles)
    except Exception as e:
        logger.warning(f"AI analyst output rejected, using heuristic fallback: {e}")
        return fallback

    # S6.1: validate the AI-built output against ground truth BEFORE merging in
    # heuristic backfill. Every KPI must carry a provenance token tracing its
    # number to a deterministic source; unknown columns / bad intents are
    # rejected. Failure ⇒ explicit logged fallback.
    from src.contract.models import LLMOutputContract

    verdict = LLMOutputContract.validate_output(
        kpis, charts, set(enriched_profiles.keys())
    )
    if not verdict.ok:
        logger.warning(
            "AI analyst output failed LLMOutputContract "
            f"({len(verdict.reasons)} issues): {verdict.reasons[:5]} "
            "— falling back to heuristic Layer 4."
        )
        return fallback

    # Merge, never shrink: AI-labelled items come first (better titles), then
    # every heuristic item the model didn't already cover is appended. This
    # keeps the full chart breadth the heuristic produced (20+) while the AI
    # only improves ordering and naming on top.
    def _sig(spec):
        return (spec.get("intent"), spec.get("x_field"), spec.get("y_field"))

    ai_sigs = {_sig(c) for c in charts}
    charts += [s for s in fallback_specs if _sig(s) not in ai_sigs]

    # KPI merge: the heuristic labels columns by raw name ("HBA1C_LEVEL") and
    # would duplicate an AI KPI that already covers the same column with a
    # human label. Dedup by SOURCE (column / corr-pair), not label text. If the
    # model gave a full set, drop the raw heuristic backfill entirely — that
    # raw-name leak is exactly the "amateur" look we're removing.
    ai_src = {k.get("_src") for k in kpis if k.get("_src")}
    if len(kpis) < 6:
        for hk in fallback_kpis:
            lbl = str(hk.get("label", ""))
            if lbl.startswith("Corr: ") and " & " in lbl:
                a, b = lbl[6:].split(" & ", 1)
                src = ("corr", frozenset([a.strip(), b.strip()]))
            else:
                src = ("col", lbl)
            if src not in ai_src:
                kpis.append(hk)
    for k in kpis:
        k.pop("_src", None)

    if not kpis and not charts:
        return fallback

    logger.info(
        f"AI analyst: {len(kpis)} KPIs, {len(charts)} charts, "
        f"narrative={'yes' if narrative else 'no'}, "
        f"{len(ai_kis)} key-indicators, "
        f"{len(ai_use_cases)} use-cases, {len(ai_recs)} recs"
    )
    return {
        "kpis": kpis,
        "chart_specs": charts,
        "narrative": narrative,
        "key_indicators": ai_kis,
        "use_cases": ai_use_cases,
        "recommendations": ai_recs,
    }


_VALID_ROLES = {"boolean", "datetime", "numeric", "identifier",
                "categorical", "text"}

_ROLE_SYSTEM_PROMPT = (
    "You are a data-typing expert. For each listed column you are given its "
    "name, the heuristic's current role, identifier-confidence, unique ratio, "
    "dtype and sample values. The heuristic is known to mislabel real metrics "
    "and categories as 'identifier', and dates as 'text'. Decide the correct "
    "role for each from EXACTLY this set: boolean, datetime, numeric, "
    "identifier, categorical, text. A column is 'identifier' ONLY if its values "
    "exist to uniquely key a row (IDs, codes, UUIDs, sequential keys) — NOT if "
    "it is a measurable quantity or a real category. Return ONLY JSON: "
    '{"roles": {"<column>": "<role>", ...}} including only columns whose role '
    "should CHANGE."
)


def _role_candidates(
    enriched_profiles: Dict[str, EnrichedProfile], df
) -> List[str]:
    """Cheap, no-AI pre-filter: only columns the heuristic is likely wrong
    about. Returns [] for clean datasets so no Groq call is made at all."""
    out = []
    for name, p in enriched_profiles.items():
        tags = p.semantic_tags or []
        if (
            p.role in ("identifier", "text")
            or "ambiguous_date" in tags
            or (p.role == "numeric" and p.unique_count <= 30)
        ):
            if name in getattr(df, "columns", []):
                out.append(name)
    return out[:25]


def arbitrate_column_roles(
    enriched_profiles: Dict[str, EnrichedProfile], df
) -> Dict[str, str]:
    """Let the LLM correct ambiguous column roles BEFORE Layers 3/4 consume
    them. Mutates enriched_profiles in place. Returns the applied changes
    (empty when nothing changed / AI unavailable). Never raises."""
    if not config.AI_ANALYST_ENABLED:
        return {}
    candidates = _role_candidates(enriched_profiles, df)
    if not candidates:
        return {}

    from src.analysis.llm import get_llm_provider

    provider = get_llm_provider()
    if not provider.available():
        return {}

    try:
        from src.utils.identifier_detector import (
            is_likely_identifier_with_confidence,
        )
    except Exception as e:  # pragma: no cover
        logger.warning(f"Role arbitration disabled: {e}")
        return {}

    cols_payload = []
    for name in candidates:
        p = enriched_profiles[name]
        try:
            series = df[name].dropna()
            samples = [str(v) for v in series.unique()[:8]]
            conf = round(
                is_likely_identifier_with_confidence(df[name], name)[2], 3
            )
        except Exception:
            samples, conf = [], 0.0
        cols_payload.append({
            "name": name,
            "current_role": p.role,
            "identifier_confidence": conf,
            "unique_ratio": round(
                p.unique_count / max(len(df), 1), 4
            ),
            "dtype": p.dtype,
            "samples": samples,
        })

    try:
        parsed = provider.complete_json(
            system=_ROLE_SYSTEM_PROMPT,
            user=json.dumps({"columns": cols_payload}, default=str),
            temperature=0.0,
        )
    except Exception as e:
        logger.warning(f"Role arbitration call failed, keeping heuristic: {e}")
        return {}

    import re
    _ID_NAME = re.compile(
        r"(?:^|_)(id|uuid|guid|nhs|ssn|number|num|no|code|key|ref|account|"
        r"acct|iban|mrn|index)(?:_|$)", re.IGNORECASE
    )

    changes: Dict[str, str] = {}
    for col, new_role in (parsed.get("roles") or {}).items():
        new_role = str(new_role).strip().lower()
        if (
            col in enriched_profiles
            and new_role in _VALID_ROLES
            and new_role != enriched_profiles[col].role
        ):
            old = enriched_profiles[col].role
            # Guard: never let the model un-identify a true key. Sequential /
            # near-unique values or an ID-style name mean it IS an identifier
            # regardless of what the model thinks (NHS_Number_Formatted etc.).
            if old == "identifier" and new_role != "identifier":
                uniq_ratio = enriched_profiles[col].unique_count / max(
                    len(df), 1
                )
                if uniq_ratio > 0.9 or _ID_NAME.search(col):
                    continue
            enriched_profiles[col].role = new_role
            if new_role != "identifier":
                enriched_profiles[col].semantic_tags = [
                    t for t in (enriched_profiles[col].semantic_tags or [])
                    if t not in ("id", "identifier", "code", "index")
                ]
            changes[col] = f"{old}->{new_role}"

    if changes:
        logger.info(f"AI role arbitration applied: {changes}")
    return changes
