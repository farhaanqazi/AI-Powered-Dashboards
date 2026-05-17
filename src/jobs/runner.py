"""The analysis job body — framework-agnostic and synchronous.

Called identically from the Arq worker (``src.jobs.worker``) and the
in-process fallback (``src.jobs.queue``). It reuses the existing pipeline
generator unchanged; only the *caller* moved off the request thread.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from src.core.pipeline import build_dashboard_from_file_generator
from src.core.state_payload import state_to_payload
from src.jobs.store import get_job_store

logger = logging.getLogger(__name__)


class _Cancelled(Exception):
    pass


def run_analysis_job(
    *,
    job_id: str,
    session_key: str,
    trace_id: str,
    file_path: str,
    filename: str,
    encoding: Optional[str] = None,
) -> None:
    """Run the pipeline for a spooled upload, streaming progress into the job
    store and persisting the finished dashboard. Never raises — terminal
    failure is recorded as a job event."""
    store = get_job_store()
    try:
        with open(file_path, "rb") as fh:
            import io

            stream = io.BytesIO(fh.read())

        for event in build_dashboard_from_file_generator(
            stream, original_filename=filename, encoding=encoding
        ):
            if store.is_cancelled(job_id):
                raise _Cancelled()

            phase = event.get("phase")
            if phase == "done":
                state = event.get("state")
                if state is None:
                    store.set_error(job_id, "The analysis produced no result.")
                    return
                payload = state_to_payload(state, filename)
                # Same persistence the synchronous path uses (session-keyed).
                from src.persistence.repository import get_repository

                get_repository().save(
                    session_key, trace_id=trace_id, payload=payload
                )
                store.append_event(job_id, {
                    "phase": "done",
                    "message": event.get("message", "Complete"),
                    "percent": 100,
                    "trace_id": trace_id,
                    "data": payload,
                })
                return
            else:
                store.append_event(job_id, event)

    except _Cancelled:
        store.append_event(job_id, {
            "phase": "cancelled", "message": "Analysis cancelled.",
            "percent": 100,
        })
    except Exception as exc:  # noqa: BLE001 - terminal failure -> job event
        logger.exception("Analysis job %s failed", job_id)
        store.set_error(job_id, f"Server error: {exc}")
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass
