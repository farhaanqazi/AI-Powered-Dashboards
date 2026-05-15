"""Structlog configuration: JSON stdout logs with request-id propagation.

Existing call sites use stdlib `logging.getLogger(...)`. Structlog patches the
root handlers so those calls automatically render as JSON on stdout — no per-
module migration needed. The existing file-rotation handlers in `src/logger.py`
remain available behind the `LOG_FILE_HANDLERS=true` env flag for local dev.
"""
from __future__ import annotations

import logging
import os
import sys
from contextvars import ContextVar
from typing import Optional

import structlog

request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

_CONFIGURED = False


def _add_request_id(_, __, event_dict):
    rid = request_id_var.get()
    if rid is not None:
        event_dict["request_id"] = rid
    return event_dict


def configure_observability_logging(force: bool = False) -> None:
    """Idempotently configure structlog + stdlib logging.

    Args:
        force: bypass the sentinel — used by tests to reconfigure between cases.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        timestamper,
        _add_request_id,
    ]

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(level)

    root = logging.getLogger()
    root.handlers = [
        h for h in root.handlers
        if not isinstance(h, logging.StreamHandler) or h.stream is not sys.stdout
    ]
    root.addHandler(handler)
    root.setLevel(level)

    for noisy in ("uvicorn.access", "watchfiles", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
