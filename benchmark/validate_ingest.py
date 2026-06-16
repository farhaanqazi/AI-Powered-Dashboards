"""Validate the harness loader + consent on representative tables.

Proves: (1) the engine's sniffing loader parses `;`-delimited and clean CSVs
correctly (vs pd.read_csv collapsing them), and (2) force_consent re-runs the
fenced path on PII-false-positive tables so they yield tokened KPIs.
Run: venv/Scripts/python.exe -m benchmark.validate_ingest
"""
from benchmark.harness.env import setup_scratch_env
setup_scratch_env("benchmark_validate")  # MUST precede src imports

import os
from benchmark.harness.ingest import load_fenced

CASES = [
    "data (2).csv",          # ;-delimited — pd.read_csv collapsed to 1 col before
    "paddy.csv",             # clean, fenced path, expect tokened KPIs
    "Titanic-Dataset.csv",   # PII false positive (Ticket) -> consent path
    "study_performance.csv", # clean small
]

for name in CASES:
    path = os.path.join("benchmark", "datasets", name)
    if not os.path.exists(path):
        print(f"\n{name}: MISSING"); continue
    r = load_fenced(path)
    print("\n" + "=" * 64)
    print(f"{name}")
    if not r["ok"]:
        print(f"  FAILED: {r['error']}"); continue
    g = r["gate"]
    print(f"  parsed: {r['n_rows']} rows x {r['n_cols']} cols")
    print(f"  schema: would_halt={g['would_schema_halt']} status={g['schema_status']}")
    print(f"  pii:    tripped={g['pii_gate_tripped']} consented={r['consented']} cols={g.get('pii_columns')}")
    print(f"  KPIs:   {r['n_kpis']} total, {r['n_kpis_tokened']} carry a provenance token")
