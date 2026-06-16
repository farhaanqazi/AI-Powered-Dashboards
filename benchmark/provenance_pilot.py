"""Document B precondition pilot: is Trust machine-computable on Path A?

Runs clean tables through the FROZEN engine with the LLM on (Groq), then checks
whether every quoted figure (KPI) carries a provenance token we can (a) parse
into a deterministic computation and (b) recompute + match. Also reports whether
the table was PII-gated (which blocks the provenance-emitting LLM path).
"""
import os, sys, json, tempfile, re
from dotenv import load_dotenv
load_dotenv()  # GROQ_API_KEY into env BEFORE importing src.config

_S = os.path.join(tempfile.gettempdir(), "prov_pilot")
os.makedirs(_S, exist_ok=True)
os.environ["JOB_SPOOL_DIR"] = os.path.join(_S, "spool")
os.environ["CLEANED_DF_DURABLE_DIR"] = os.path.join(_S, "frames")
os.environ["LOG_DIR"] = os.path.join(_S, "logs")
os.environ["DATABASE_URL"] = f"sqlite:///{_S}/p.db"
os.environ["SCHEMA_REVIEW_ENABLED"] = "false"

import pandas as pd
from src import config
from src.analysis.llm.factory import get_llm_provider

prov = get_llm_provider()
print(f"[pilot] provider={prov.name!r} available={prov.available()} "
      f"ai_analyst={config.AI_ANALYST_ENABLED} model={getattr(config,'GROQ_MODEL',None)}")
if not prov.available():
    print("[pilot] FAIL: no usable LLM provider. STOP."); sys.exit(2)

from src.core.pipeline import build_dashboard_from_df
from src.core.state_payload import state_to_payload


def recompute(df, provd):
    """Best-effort: turn a provenance dict into a deterministic number."""
    if not isinstance(provd, dict):
        return None, "no-dict"
    blob = json.dumps(provd).lower()
    m = re.search(r'corr[:\.]([^"|]+)\|([^"|}]+)', blob)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        cols = {c.lower(): c for c in df.columns}
        if a in cols and b in cols:
            try:
                return float(pd.to_numeric(df[cols[a]], errors="coerce").corr(
                    pd.to_numeric(df[cols[b]], errors="coerce"))), f"corr({a},{b})"
            except Exception as e:
                return None, f"corr-err:{e}"
    m = re.search(r'l1\.([^."]+)\.(mean|sum|min|max|median|count|std)', blob)
    if m:
        col, metric = m.group(1), m.group(2)
        cols = {c.lower(): c for c in df.columns}
        if col in cols:
            s = pd.to_numeric(df[cols[col]], errors="coerce")
            return float(getattr(s, metric)()), f"{metric}({col})"
    return None, f"unparsed:{blob[:120]}"


TABLES = ["study_performance.csv", "data (2).csv", "paddy.csv", "Titanic-Dataset.csv"]
for t in TABLES:
    path = os.path.join("benchmark", "datasets", t)
    print("\n" + "=" * 72)
    print(f"[pilot] {t}")
    df = pd.read_csv(path)
    print(f"  {len(df)} rows x {len(df.columns)} cols")
    state = build_dashboard_from_df(df, original_filename=t)
    if state is None:
        print("  pipeline returned None. SKIP."); continue
    payload = state_to_payload(state, t)
    dq = ((payload.get("dataset_profile") or {}).get("data_quality") or {})
    rep = dq.get("report") or {}
    print(f"  pii_blocked={rep.get('pii_blocked')} ai_consent_required={rep.get('ai_consent_required')} "
          f"pii_cols={rep.get('pii_columns')}")
    kpis = payload.get("kpis") or []
    n_prov = sum(1 for k in kpis if k.get("provenance"))
    print(f"  KPIs={len(kpis)}  with_provenance={n_prov}")
    checked = 0
    for k in kpis[:12]:
        provd = k.get("provenance")
        if not provd:
            continue
        got, how = recompute(df, provd)
        quoted = k.get("value")
        match = ""
        try:
            qv = float(str(quoted).replace(",", ""))
            if got is not None:
                match = "MATCH" if abs(qv - got) < max(0.01, abs(got) * 0.02) else f"MISMATCH(got {got:.4g})"
        except Exception:
            pass
        checked += 1
        print(f"    KPI {k.get('label')!r}={quoted!r}  token->{how}  {match}")
    if n_prov and checked == 0:
        print("    (provenance present but none parsed by the pilot recompute)")
