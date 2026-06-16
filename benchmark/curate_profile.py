"""Profile candidate eval tables through the ENGINE loader (no LLM) to confirm
clean ingest + capture the real role mix, so the manifest only proposes tables
that actually parse. Cheap: no Groq calls. Run: -m benchmark.curate_profile
"""
from benchmark.harness.env import setup_scratch_env
setup_scratch_env("benchmark_curate")

import os
from collections import Counter

CANDIDATES = [
    "Titanic-Dataset.csv", "supermarket_sales.csv", "study_performance.csv",
    "paddy.csv", "data (2).csv", "House_Rent_Dataset.csv", "netflix.csv",
    "Electric_Vehicle_Population_Size_History_By_County_.csv",
    "youth_unemployment_global.csv", "goalscorers.csv", "hotel_bookings.csv",
    "Online Retail.csv", "owid-covid-data.csv", "healthcare_dataset.csv",
    "car_prices.csv", "ai_student_impact_dataset (1).csv",
    "List of Unicorns in the World.csv", "Air_bnb.csv", "drug use_final.csv",
    "supermarket_sales.csv",
]
CANDIDATES = list(dict.fromkeys(CANDIDATES))  # dedupe

from src.data.parser import load_table_from_file
from src.analysis.layer_1_profiler import run_syntactic_profiling
from src.analysis.layer_2_classifier import run_semantic_classification
from src.contract.ingest_gate import run_ingest_gate

OUT = open(os.path.join("benchmark", "_curate_out.tsv"), "w", encoding="utf-8")
def emit(line):
    print(line); OUT.write(line + "\n"); OUT.flush()

emit(f"{'FILE':46}\t{'rows':>8} {'C':>3} {'num':>3} {'cat':>3} {'dt':>2} {'id':>2}  PII")
for name in CANDIDATES:
    path = os.path.join("benchmark", "datasets", name)
    if not os.path.exists(path):
        emit(f"{name[:46]:46}  MISSING"); continue
    try:
        with open(path, "rb") as fh:
            lr = load_table_from_file(fh, filename=name, max_rows=120000)
        if not lr.success or lr.df is None:
            emit(f"{name[:46]:46}  LOAD-FAIL: {lr.detail}"); continue
        df = lr.df
        prof = run_semantic_classification(run_syntactic_profiling(df, max_cols=50), df)
        roles = Counter(p.role for p in prof.values())
        ing = run_ingest_gate(df)
        pii = ",".join((ing.pii_columns or {}).keys()) if getattr(ing, "pii_columns", None) else ""
        emit(f"{name[:46]:46}	{len(df):>8} {len(df.columns):>3} "
              f"{roles.get('numeric',0):>3} {roles.get('categorical',0):>3} "
              f"{roles.get('datetime',0):>2} {roles.get('identifier',0):>2}  {pii[:40]}")
    except Exception as e:
        emit(f"{name[:46]:46}  ERR: {type(e).__name__}: {e}")
