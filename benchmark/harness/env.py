"""Run-environment setup for the benchmark — MUST run before importing `src`.

Routes every engine filesystem side effect (df_cache parquet, model cache, logs,
sqlite) to a throwaway scratch dir, loads the Groq key from `.env`, and disables
the schema-review HALT so batches run unattended (Document B run-env rule). The
schema-review *verdict* is still recorded per dataset by the ingest layer, so the
disabled gate is visible, not hidden.
"""
from __future__ import annotations

import os
import tempfile


def setup_scratch_env(scratch_name: str = "benchmark_run") -> str:
    """Idempotent. Returns the scratch dir. Call before importing `src`."""
    from dotenv import load_dotenv
    load_dotenv()  # GROQ_API_KEY → env, for Path A (AI-on)

    scratch = os.path.join(tempfile.gettempdir(), scratch_name)
    os.makedirs(scratch, exist_ok=True)
    # Force side effects to scratch (overrides .env's blank/real values).
    os.environ["JOB_SPOOL_DIR"] = os.path.join(scratch, "spool")
    os.environ["CLEANED_DF_DURABLE_DIR"] = os.path.join(scratch, "frames")
    os.environ["LOG_DIR"] = os.path.join(scratch, "logs")
    os.environ["DATABASE_URL"] = f"sqlite:///{scratch}/bench.db"
    # Disable the schema-review HALT so batches run end-to-end. The verdict is
    # still captured per dataset (ingest.gate.would_schema_halt) for the
    # include/excluding-would-have-halted Trust split.
    os.environ["SCHEMA_REVIEW_ENABLED"] = "false"
    return scratch
