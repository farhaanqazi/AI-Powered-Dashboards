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


def _redis_settings():
    from arq.connections import RedisSettings

    return RedisSettings.from_dsn(config.REDIS_URL or "redis://localhost:6379")


class WorkerSettings:
    functions = [run_analysis_task]
    redis_settings = _redis_settings()
    # A single big analysis at a time per worker keeps memory predictable;
    # scale by running more worker processes.
    max_jobs = int(__import__("os").environ.get("JOB_WORKER_MAX_JOBS", 2))
    job_timeout = int(__import__("os").environ.get("JOB_TIMEOUT_SECONDS", 3600))
    keep_result = 0
