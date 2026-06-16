"""Materialize deterministic capped copies of oversized tables.

For each manifest table with cap_rows, if the table exceeds the cap, write a
seeded (random_state=seed) sample to benchmark/datasets/capped/<file>. Both arms
AND ground truth later run on this frozen file, so nothing drifts. Loaded via the
engine loader (robust encoding/sniffing), so the capped copy matches what the
engine would have parsed. Run: -m benchmark.materialize_caps
"""
from benchmark.harness.env import setup_scratch_env
setup_scratch_env("benchmark_caps")

import os, json
from src.data.parser import load_table_from_file

MANIFEST = os.path.join("benchmark", "datasets", "manifest.json")
CAP_DIR = os.path.join("benchmark", "datasets", "capped")
os.makedirs(CAP_DIR, exist_ok=True)

with open(MANIFEST, encoding="utf-8") as f:
    man = json.load(f)
seed = man.get("seed", 42)

for t in man["tables"]:
    cap = t.get("cap_rows")
    if not cap:
        continue
    name = t["file"]
    src = os.path.join("benchmark", "datasets", name)
    with open(src, "rb") as fh:
        lr = load_table_from_file(fh, filename=name, max_rows=10_000_000)
    if not lr.success or lr.df is None:
        print(f"  {name}: LOAD-FAIL {lr.detail}"); continue
    df = lr.df
    if len(df) <= cap:
        print(f"  {name}: {len(df)} rows <= cap {cap} — no capping needed (use original)")
        continue
    capped = df.sample(n=cap, random_state=seed).sort_index()
    out = os.path.join(CAP_DIR, name)
    capped.to_csv(out, index=False)
    print(f"  {name}: {len(df)} -> {len(capped)} rows (seed {seed}) -> {out}")

print("done.")
