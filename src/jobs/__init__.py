"""Async analysis jobs (Phase 10 S10.1, pulled forward).

Decouples authentication from the long-running pipeline. The upload request
authenticates and returns immediately with a ``job_id``; the analysis runs as
a job (out-of-process Arq worker when ``JOB_QUEUE_ENABLED``, else an in-process
asyncio task) and progress is streamed by id.

The pipeline itself (``src.core.pipeline``) is unchanged — this is pure
orchestration around it.
"""
from src.jobs.store import JobStore, get_job_store, TERMINAL
from src.jobs.queue import submit_analysis_job

__all__ = ["JobStore", "get_job_store", "TERMINAL", "submit_analysis_job"]
