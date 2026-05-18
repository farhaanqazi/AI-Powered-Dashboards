"""Phase 11 — Ask Your Data: deterministic tools + bounded agent.

The invariant under test: the LLM never produces a number. Tools compute
everything with a provenance token, contract guards hold (no summing a
ratio), and the agent is hard-bounded and never raises.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.analysis.ask import run_ask
from src.analysis.ask.tools import ToolError, aggregate, column_stat, correlation
from src.analysis.layer_1_profiler import run_syntactic_profiling
from src.analysis.layer_2_classifier import run_semantic_classification
from src.contract import compile_contract, run_ingest_gate


@pytest.fixture
def df_and_contract():
    df = pd.DataFrame({
        "region": ["N", "S", "E", "W"] * 9,
        "revenue": [f"${v:,.2f}" for v in [100.0, 250.0, 320.0, 90.0] * 9],
        "margin_pct": [round(0.1 * (i % 5), 3) for i in range(36)],
        "units": [3, 7, 12, 5] * 9,
    })
    res = run_ingest_gate(df)
    profiles = run_semantic_classification(
        run_syntactic_profiling(res.df), res.df
    )
    contract = compile_contract(res.df, profiles, res)
    return res.df, contract


# --- tools ---------------------------------------------------------------

def test_column_stat_has_value_and_provenance(df_and_contract):
    df, c = df_and_contract
    out = column_stat(df, c, {"column": "revenue", "metric": "mean"})
    assert out["result"]["value"] > 0
    assert out["provenance"]["token"] == "L1.revenue.mean"


def test_ratio_cannot_be_summed(df_and_contract):
    df, c = df_and_contract
    with pytest.raises(ToolError):
        column_stat(df, c, {"column": "margin_pct", "metric": "sum"})


def test_aggregate_groups_and_guards(df_and_contract):
    df, c = df_and_contract
    out = aggregate(df, c, {"group_by": "region", "value": "revenue",
                            "agg": "sum"})
    assert {r["group"] for r in out["result"]["rows"]} == {"N", "S", "E", "W"}
    assert out["provenance"]["token"].startswith("agg.region.sum")
    with pytest.raises(ToolError):
        aggregate(df, c, {"group_by": "region", "value": "margin_pct",
                          "agg": "sum"})


def test_correlation_rejects_non_measures(df_and_contract):
    df, c = df_and_contract
    out = correlation(df, c, {"a": "revenue", "b": "units"})
    assert -1.0 <= out["result"]["pearson"] <= 1.0
    with pytest.raises(ToolError):
        correlation(df, c, {"a": "region", "b": "revenue"})


# --- heuristic agent (no LLM) --------------------------------------------

def test_heuristic_average_question(df_and_contract):
    df, c = df_and_contract
    out = run_ask(df, c, "what is the average revenue?", provider=None)
    assert out["status"] == "ok"
    assert out["planner"] == "heuristic"
    assert out["numbers_traceable"] is True
    step = out["steps"][0]
    assert step["tool"] == "column_stat"
    assert step["provenance"]["token"] == "L1.revenue.mean"


def test_heuristic_breakdown_question(df_and_contract):
    df, c = df_and_contract
    out = run_ask(df, c, "show total revenue by region", provider=None)
    assert out["steps"][0]["tool"] == "aggregate"
    assert out["steps"][0]["result"]["agg"] == "sum"


def test_heuristic_correlation_question(df_and_contract):
    df, c = df_and_contract
    out = run_ask(df, c, "is there a correlation between revenue and units?",
                  provider=None)
    assert out["steps"][0]["tool"] == "correlation"


# --- LLM-planned + bounded -----------------------------------------------

class _FakeProvider:
    def __init__(self, steps, answer="Narrated."):
        self._steps = steps
        self._answer = answer

    def available(self):
        return True

    def complete_json(self, *, system, user, temperature=0.2):
        if "translate a business question" in system:
            return {"steps": self._steps}
        return {"answer": self._answer}


def test_llm_planner_path_and_narration(df_and_contract):
    df, c = df_and_contract
    prov = _FakeProvider(
        [{"tool": "column_stat",
          "params": {"column": "units", "metric": "mean"}}],
        answer="Average units is computed.",
    )
    out = run_ask(df, c, "tell me about units", provider=prov)
    assert out["planner"] == "llm"
    assert out["answer"] == "Average units is computed."
    assert out["steps"][0]["provenance"]["token"] == "L1.units.mean"


def test_agent_is_hard_bounded(df_and_contract):
    df, c = df_and_contract
    many = [{"tool": "column_stat",
             "params": {"column": "units", "metric": "mean"}}] * 10
    prov = _FakeProvider(many)
    out = run_ask(df, c, "q", provider=prov, max_iterations=2)
    assert out["iterations"] == 2
    assert len(out["steps"]) == 2


def test_unknown_tool_does_not_raise(df_and_contract):
    df, c = df_and_contract
    prov = _FakeProvider([{"tool": "rm_rf", "params": {}}])
    out = run_ask(df, c, "q", provider=prov)
    assert out["status"] == "no_answer"
    assert "error" in out["steps"][0]


def test_disabled_short_circuits(df_and_contract, monkeypatch):
    df, c = df_and_contract
    monkeypatch.setattr(config, "ASK_DATA_ENABLED", False)
    assert run_ask(df, c, "anything")["status"] == "disabled"


# --- API end-to-end ------------------------------------------------------

def test_ask_api_requires_prior_analysis(client):
    r = client.post("/api/ask", json={"question": "hi"})
    assert r.status_code == 404


def test_ask_api_end_to_end(client):
    import io, time
    csv = b"region,revenue\nN,100\nS,250\nE,320\nW,90\nN,110\nS,260\n"
    sub = client.post(
        "/api/jobs/upload",
        files={"dataset": ("s.csv", io.BytesIO(csv), "text/csv")},
    )
    job_id = sub.json()["job_id"]
    deadline = time.time() + 60
    while time.time() < deadline:
        if client.get(f"/api/jobs/{job_id}").json()["status"] in (
            "done", "failed", "cancelled"
        ):
            break
        time.sleep(0.5)

    r = client.post("/api/ask", json={"question": "average revenue?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] in ("ok", "unavailable")
    if body["status"] == "ok":
        assert body["numbers_traceable"] is True
        assert any("provenance" in s for s in body["steps"])
