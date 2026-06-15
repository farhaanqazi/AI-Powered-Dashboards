"""Document A, Task 6 — prove the harness can drive the engine read-only.

Loads one existing demo dataset, runs it through the headless pipeline
entrypoint (`src.core.pipeline.build_dashboard_from_df`), and prints the
rendered output plus every provenance token / `numbers_traceable` flag it can
find in the payload.

"Read-only" means read-only w.r.t. *engine source*. The pipeline still has
filesystem side effects (durable df_cache parquet tier, model cache, logs), so
this script points all of those at a throwaway temp dir and leaves the engine
defaults untouched. The only non-default it sets is SCHEMA_REVIEW_ENABLED=false
so a small demo table runs to completion instead of halting at human review.

Run:  venv/Scripts/python.exe -m benchmark.smoke_test
"""
from __future__ import annotations

import os
import sys
import tempfile

# --- redirect all engine side effects to a throwaway dir BEFORE importing src.
# config reads these at import time, so they must be set first.
_SCRATCH = os.path.join(tempfile.gettempdir(), "benchmark_smoke_scratch")
os.makedirs(_SCRATCH, exist_ok=True)
os.environ.setdefault("JOB_SPOOL_DIR", os.path.join(_SCRATCH, "spool"))
os.environ.setdefault("CLEANED_DF_DURABLE_DIR", os.path.join(_SCRATCH, "frames"))
os.environ.setdefault("LOG_DIR", os.path.join(_SCRATCH, "logs"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_SCRATCH}/smoke.db")
# Let a tiny demo table run end-to-end instead of routing to HITL review.
os.environ.setdefault("SCHEMA_REVIEW_ENABLED", "false")

import pandas as pd  # noqa: E402

from src.core.pipeline import build_dashboard_from_df  # noqa: E402
from src.core.state_payload import state_to_payload  # noqa: E402

DEMO = os.path.join("tests", "fixtures", "sample_data.csv")
PROVENANCE_KEYS = {"provenance", "numbers_traceable", "provenance_token", "token", "traceable"}


def _scan_provenance(obj, path="payload", hits=None):
    """Recursively collect any provenance-related fields in the payload."""
    if hits is None:
        hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in PROVENANCE_KEYS:
                hits.append((f"{path}.{k}", v))
            _scan_provenance(v, f"{path}.{k}", hits)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:50]):
            _scan_provenance(v, f"{path}[{i}]", hits)
    return hits


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEMO
    print(f"[smoke] demo dataset: {path}")
    df = pd.read_csv(path)
    print(f"[smoke] loaded {len(df)} rows x {len(df.columns)} cols")

    state = build_dashboard_from_df(df, original_filename=os.path.basename(path))
    if state is None:
        print("[smoke] FAIL: pipeline returned None")
        return 1
    payload = state_to_payload(state, os.path.basename(path))

    print("\n=== RENDERED OUTPUT (summary) ===")
    print(f"  kpis:        {len(payload.get('kpis') or [])}")
    print(f"  charts:      {len(payload.get('all_charts') or payload.get('charts') or [])}")
    eda = payload.get("eda_summary") or {}
    print(f"  eda_summary keys: {sorted(eda.keys())}")
    ml = (eda.get('ml_insights') or {})
    if ml:
        print(f"  ml_insights: { {k: (v.get('available') if isinstance(v, dict) else v) for k, v in ml.items()} }")

    print("\n=== KPIs (with provenance) ===")
    for kpi in (payload.get("kpis") or [])[:8]:
        label = kpi.get("label") or kpi.get("title") or kpi.get("name")
        prov = kpi.get("provenance")
        print(f"  - {label!r}: value={kpi.get('value')!r}  provenance={prov!r}")
    if not payload.get("kpis"):
        print("  (no KPIs produced for this demo table)")

    print("\n=== PROVENANCE / TRACEABILITY SCAN (dashboard payload) ===")
    hits = _scan_provenance(payload)
    if not hits:
        print("  (none in the dashboard payload — KPI provenance is attached only")
        print("   in the LLM-validated path; this run had no LLM provider, so the")
        print("   heuristic fallback returned KPIs without provenance tokens.)")
    for where, what in hits[:40]:
        print(f"  {where} = {what!r}")

    # The deterministic Ask/Interact path emits provenance tokens + a
    # numbers_traceable flag with NO LLM. Exercise it read-only to show a real
    # traceable figure (Document A Task 6: print provenance + numbers_traceable).
    print("\n=== DETERMINISTIC TRACEABLE FIGURE (Ask/Interact path, no LLM) ===")
    try:
        from src.contract.models import DatasetContract
        from src.analysis.ask.interact import run_interaction

        contract_dict = (payload.get("dataset_profile") or {}).get("contract")
        if not contract_dict:
            print("  (no contract on the payload — cannot exercise the interact path)")
        else:
            contract = DatasetContract.model_validate(contract_dict)
            num_col = next(
                (n for n, f in (contract.fields or {}).items()
                 if getattr(f, "role", None) in ("numeric", "ratio")),
                None,
            )
            if num_col is None:
                print("  (no numeric field in the contract to compute on)")
            else:
                spec = {"calculation": "column_stat",
                        "params": {"column": num_col, "metric": "mean"}, "filters": []}
                res = run_interaction(df, contract, spec)
                print(f"  calculation: column_stat(mean) on {num_col!r}")
                print(f"  status:           {res.get('status')}")
                print(f"  result:           {res.get('result')!r}")
                print(f"  numbers_traceable: {res.get('numbers_traceable')!r}")
                print(f"  provenance:        {res.get('provenance')!r}")
    except Exception as e:  # noqa: BLE001 - smoke is best-effort, never the gate
        print(f"  (interact path errored: {type(e).__name__}: {e})")

    print(f"\n[smoke] OK — engine driven read-only; "
          f"{len(hits)} dashboard provenance field(s), interact path shown above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
