"""Job dispatch: pick the execution strategy, run the same job body.

``JOB_QUEUE_ENABLED`` → enqueue onto an Arq worker over Redis (out-of-process,
production-correct). Otherwise (or on any Arq/Redis failure) run it in-process
as an ``asyncio`` task on a worker thread — still off the request, so the auth
token only has to survive the sub-second submit.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from src import config
from src.jobs.runner import run_analysis_job

logger = logging.getLogger(__name__)

_arq_pool = None


async def _get_arq_pool():
    global _arq_pool
    if _arq_pool is not None:
        return _arq_pool
    from arq import create_pool
    from arq.connections import RedisSettings

    _arq_pool = await create_pool(RedisSettings.from_dsn(config.REDIS_URL))
    return _arq_pool


def _run_in_process(**kwargs) -> None:
    """Fire-and-forget the job on a daemon thread. Deliberately NOT tied to
    the request's event loop — the thread runs to completion regardless of
    when the HTTP response returns or the loop pauses."""
    threading.Thread(
        target=run_analysis_job, kwargs=kwargs,
        name=f"analysis-{kwargs.get('job_id', '?')[:8]}", daemon=True,
    ).start()


async def submit_analysis_job(
    *,
    job_id: str,
    session_key: str,
    trace_id: str,
    file_path: str,
    filename: str,
    encoding: Optional[str] = None,
) -> str:
    """Dispatch the job. Returns the chosen backend name ('arq' | 'inprocess')
    for observability. Always succeeds (falls back to in-process)."""
    kwargs = dict(
        job_id=job_id, session_key=session_key, trace_id=trace_id,
        file_path=file_path, filename=filename, encoding=encoding,
    )
    if config.JOB_QUEUE_ENABLED and config.REDIS_URL:
        try:
            pool = await _get_arq_pool()
            await pool.enqueue_job("run_analysis_task", **kwargs)
            return "arq"
        except Exception as exc:  # pragma: no cover - dev never gets stuck
            logger.warning(
                "Arq enqueue failed (%s); running job in-process instead.", exc
            )
    _run_in_process(**kwargs)
    return "inprocess"
