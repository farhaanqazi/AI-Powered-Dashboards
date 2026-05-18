"""Arq worker entrypoint (the production-correct, out-of-process path).

Run alongside the API:

    arq src.jobs.worker.WorkerSettings

Requires ``REDIS_URL`` and ``JOB_QUEUE_ENABLED=true``. The task simply runs the
shared job body on a thread so the pandas pipeline never blocks the worker's
event loop.
"""
from __future__ import annotations

import asyncio

from src import config
from src.jobs.runner import run_analysis_job


async def run_analysis_task(ctx, **kwargs) -> None:
    await asyncio.to_thread(run_analysis_job, **kwargs)


async def _on_startup(ctx) -> None:
    # The worker is a separate process and never imports main.py, so it would
    # otherwise have NO structured logging. Initialise the same structlog/JSON
    # stack the API uses so worker logs are parseable and correlated.
    try:
        from src.observability.logging import configure_observability_logging

        configure_observability_logging()
    except Exception:  # pragma: no cover - never fail the worker on logging
        pass
    import logging

    logging.getLogger(__name__).info("arq worker started")


def _redis_settings():
    from arq.connections import RedisSettings

    return RedisSettings.from_dsn(config.REDIS_URL or "redis://localhost:6379")


class WorkerSettings:
    functions = [run_analysis_task]
    on_startup = _on_startup
    redis_settings = _redis_settings()
    # A single big analysis at a time per worker keeps memory predictable;
    # scale by running more worker processes.
    max_jobs = int(__import__("os").environ.get("JOB_WORKER_MAX_JOBS", 2))
    job_timeout = int(__import__("os").environ.get("JOB_TIMEOUT_SECONDS", 3600))
    keep_result = 0
