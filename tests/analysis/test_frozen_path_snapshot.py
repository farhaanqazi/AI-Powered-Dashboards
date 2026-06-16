"""Frozen-path regression guard (Document B belt-and-suspenders).

The authority knob is additive and flag-gated; its code only executes on the
LLM path. This test pins the DETERMINISTIC path (NullProvider, fixed fixture):
its canonical output must match `deterministic_golden.json`, which was generated
from the `benchmark-baseline-v1` tag via a throwaway worktree. It guards the
non-gate path against accidental drift — the closest honest analog to a
"default path unchanged" assertion. (The gate path itself is covered by
test_authority_knob.py's flag_off == level_2 identity.)
"""
from __future__ import annotations

import json
import os

import pandas as pd

from src import config
from src.analysis.llm.factory import reset_llm_provider_for_tests

GOLDEN = os.path.join("tests", "fixtures", "deterministic_golden.json")
FIXTURE = os.path.join("tests", "fixtures", "sample_data.csv")


def _canon(p):
    return {
        "kpis": sorted([[k.get("label"), str(k.get("value")), (k.get("provenance") or {}).get("token")]
                        for k in (p.get("kpis") or [])]),
        "charts": sorted([[c.get("intent"), c.get("x_field"), c.get("y_field"), c.get("chart_type")]
                          for c in (p.get("all_charts") or p.get("charts") or [])]),
        "roles": sorted([[c.get("name"), c.get("role")]
                         for c in ((p.get("dataset_profile") or {}).get("columns") or [])]),
        "eda_keys": sorted((p.get("eda_summary") or {}).keys()),
    }


def test_deterministic_path_matches_baseline_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LLM_PROVIDER", "null")
    monkeypatch.setattr(config, "SCHEMA_REVIEW_ENABLED", False)
    monkeypatch.setattr(config, "JOB_SPOOL_DIR", str(tmp_path / "spool"))
    monkeypatch.setattr(config, "CLEANED_DF_DURABLE_DIR", str(tmp_path / "frames"))
    monkeypatch.delenv("BENCHMARK_AUTHORITY_LEVEL", raising=False)
    reset_llm_provider_for_tests()

    from src.core.pipeline import build_dashboard_from_df
    from src.core.state_payload import state_to_payload

    df = pd.read_csv(FIXTURE)
    payload = state_to_payload(
        build_dashboard_from_df(df, original_filename="sample_data.csv"), "sample_data.csv")

    with open(GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)
    assert _canon(payload) == golden
    reset_llm_provider_for_tests()
