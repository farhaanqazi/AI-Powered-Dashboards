"""The bounded Ask-Your-Data agent (S11.1 + S11.2).

Flow per request:
  1. PLAN  — the LLM (or a deterministic heuristic fallback) proposes up to
     ``ASK_MAX_ITERATIONS`` tool calls from the fixed catalog. No data, just
     a capability list + column roles, so the model cannot smuggle a number.
  2. EXECUTE — each step runs deterministically; contract guards reject
     invalid asks (e.g. summing a ratio). Numbers + provenance are collected.
  3. NARRATE — the LLM is given ONLY the computed numbers and writes prose.
     No provider ⇒ a templated answer from the step summaries.

Hard-bounded: never more than the cap, always terminates, every figure
carries a provenance token.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from src import config
from src.analysis.ask.tools import TOOL_CATALOG, TOOLS, ToolError

try:
    from src.logger import get_logger
    logger = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)

_PLAN_SYSTEM = (
    "You translate a business question into deterministic data operations. "
    "You may ONLY choose tools from the provided catalog and reference the "
    "provided column names. You NEVER state or compute a number — the backend "
    "executes the tools and fills figures. Return a JSON object with a "
    '"steps" array; each step has "tool" (a catalog name) and "params". '
    "Use no more than {cap} steps."
)
_NARRATE_SYSTEM = (
    "You are given a question and the EXACT numbers the system computed for "
    "it. Write a concise 1-3 sentence answer. Use only these numbers; never "
    "invent or recompute. Return JSON: {\"answer\": \"...\"}."
)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _columns(contract: Any) -> Dict[str, str]:
    fields = getattr(contract, "fields", {}) or {}
    return {n: getattr(f, "role", "unknown") for n, f in fields.items()}


def _mentioned(question: str, cols: Dict[str, str]) -> List[str]:
    q = _norm(question)
    hits = [c for c in cols if _norm(c) and _norm(c) in q]
    # Longest names first so "unit_price" wins over "price".
    return sorted(hits, key=len, reverse=True)


def _heuristic_plan(question: str, cols: Dict[str, str]) -> List[Dict[str, Any]]:
    ql = question.lower()
    mentioned = _mentioned(question, cols)
    nums = [c for c in mentioned if cols[c] in ("numeric", "ratio", "year")]
    cats = [c for c in mentioned if cols[c] in ("categorical", "identifier",
                                                 "boolean", "datetime")]
    all_nums = [c for c, r in cols.items() if r in ("numeric", "ratio")]
    all_cats = [c for c, r in cols.items() if r in ("categorical", "boolean")]

    if re.search(r"correlat|relationship|related|associat", ql):
        pool = nums if len(nums) >= 2 else all_nums
        if len(pool) >= 2:
            return [{"tool": "correlation",
                     "params": {"a": pool[0], "b": pool[1]}}]
    m = re.search(r"\bby\s+([a-z0-9_ ]+)", ql)
    if m or "breakdown" in ql or "per " in ql:
        gb = None
        if m:
            cand = _mentioned(m.group(1), cols)
            gb = next((c for c in cand if cols[c] in
                       ("categorical", "identifier", "boolean", "year",
                        "datetime")), None)
        gb = gb or (cats[0] if cats else (all_cats[0] if all_cats else None))
        val = nums[0] if nums else (all_nums[0] if all_nums else None)
        if gb and val:
            agg = "sum" if re.search(r"total|sum", ql) else "mean"
            return [{"tool": "aggregate",
                     "params": {"group_by": gb, "value": val, "agg": agg}}]
    metric = None
    for kw, mt in (("average", "mean"), ("mean", "mean"), ("median", "median"),
                   ("std", "std"), ("deviation", "std"),
                   ("maximum", "max"), ("highest", "max"), ("max", "max"),
                   ("minimum", "min"), ("lowest", "min"), ("min", "min"),
                   ("total", "sum"), ("sum", "sum")):
        if kw in ql:
            metric = mt
            break
    target = nums[0] if nums else (all_nums[0] if all_nums else None)
    if metric and target:
        return [{"tool": "column_stat",
                 "params": {"column": target, "metric": metric}}]
    if re.search(r"how many|count|number of", ql):
        col = (mentioned[0] if mentioned else
               (all_cats[0] if all_cats else None))
        if col:
            return [{"tool": "column_stat",
                     "params": {"column": col, "metric": "count"}}]
    if re.search(r"top|most common|frequent|distribution", ql) or cats:
        col = cats[0] if cats else (all_cats[0] if all_cats else None)
        if col:
            return [{"tool": "top_categories",
                     "params": {"column": col, "n": 5}}]
    if target:
        return [{"tool": "column_stat",
                 "params": {"column": target, "metric": "mean"}}]
    return []


def _llm_plan(provider, question, cols, cap) -> Optional[List[Dict[str, Any]]]:
    try:
        out = provider.complete_json(
            system=_PLAN_SYSTEM.format(cap=cap),
            user=json.dumps({
                "question": question,
                "columns": cols,
                "tool_catalog": TOOL_CATALOG,
            }, default=str),
            temperature=0.0,
        )
        steps = out.get("steps")
        if isinstance(steps, list) and steps:
            return steps[:cap]
    except Exception as e:  # provider failure ⇒ heuristic fallback
        logger.info("Ask planner LLM unavailable (%s); heuristic plan", e)
    return None


def _narrate(provider, question, steps) -> str:
    facts = [{"summary": s["summary"], **s["result"]}
             for s in steps if "result" in s]
    if not facts:
        return ("I couldn't compute that from this dataset with the "
                "available deterministic tools.")
    if provider is not None and provider.available():
        try:
            out = provider.complete_json(
                system=_NARRATE_SYSTEM,
                user=json.dumps({"question": question, "computed": facts},
                                default=str),
                temperature=0.2,
            )
            ans = str(out.get("answer", "")).strip()
            if ans:
                return ans
        except Exception as e:
            logger.info("Ask narration LLM unavailable (%s); templated", e)
    return " ".join(s["summary"] for s in steps if "summary" in s
                     and "result" in s)


def run_ask(
    df,
    contract: Any,
    question: str,
    *,
    provider: Optional[Any] = None,
    max_iterations: Optional[int] = None,
) -> Dict[str, Any]:
    """Answer ``question`` over the cleaned ``df`` under ``contract``.

    Never raises. Every returned number traces to a provenance token; the
    agent runs at most ``max_iterations`` tool steps.
    """
    if not config.ASK_DATA_ENABLED:
        return {"status": "disabled", "answer": "", "steps": []}
    question = (question or "").strip()
    if not question:
        return {"status": "error", "answer": "Ask a question.", "steps": []}

    cap = max_iterations or config.ASK_MAX_ITERATIONS
    cols = _columns(contract)
    if provider is None:
        try:
            from src.analysis.llm import get_llm_provider
            provider = get_llm_provider()
        except Exception:
            provider = None

    plan = None
    planner = "heuristic"
    if provider is not None and getattr(provider, "available", lambda: False)():
        plan = _llm_plan(provider, question, cols, cap)
        if plan:
            planner = "llm"
    if not plan:
        plan = _heuristic_plan(question, cols)

    steps: List[Dict[str, Any]] = []
    for raw in plan[:cap]:  # hard bound — always terminates
        name = (raw or {}).get("tool")
        params = (raw or {}).get("params") or {}
        fn = TOOLS.get(name)
        if fn is None:
            steps.append({"tool": name, "params": params,
                          "error": f"Unknown tool '{name}'"})
            continue
        try:
            res = fn(df, contract, params)
            steps.append({"tool": name, "params": params, **res})
        except ToolError as e:
            steps.append({"tool": name, "params": params, "error": str(e)})
        except Exception as e:  # noqa: BLE001 - never break the request
            logger.warning("Ask tool '%s' failed: %s", name, e)
            steps.append({"tool": name, "params": params,
                          "error": "Tool execution failed."})

    answer = _narrate(provider, question, steps)
    ok = any("result" in s for s in steps)
    return {
        "status": "ok" if ok else "no_answer",
        "question": question,
        "answer": answer,
        "steps": steps,
        "iterations": len(steps),
        "max_iterations": cap,
        "planner": planner,
        "numbers_traceable": True,
    }
