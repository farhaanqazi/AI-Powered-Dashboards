"""
src/logger.py
─────────────
Centralised logging configuration for AI-Powered Dashboards.

Log files (written to the `logs/` directory at the project root):
  • logs/app.log    – INFO and above, rotating, human-readable
  • logs/errors.log – WARNING and above, rotating, JSON-structured for
                      easy machine-parsing / alerting pipelines

Usage in any module:
    from src.logger import get_logger
    logger = get_logger(__name__)

Environment variables (all optional):
    LOG_LEVEL       – Root log level for the file handlers (default: INFO)
    LOG_DIR         – Directory to write log files to (default: logs)
    LOG_MAX_BYTES   – Max size per log file before rotation (default: 10 MB)
    LOG_BACKUP_COUNT – Number of rotated backups to keep (default: 5)
    LOG_JSON_ERRORS – Set to "false" to write errors.log in plain text
"""

import json
import logging
import logging.handlers
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ──────────────────────────────────────────────
# Configuration (from env or defaults)
# ──────────────────────────────────────────────
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_DIR: Path = Path(os.environ.get("LOG_DIR", "logs"))
LOG_MAX_BYTES: int = int(os.environ.get("LOG_MAX_BYTES", 10 * 1024 * 1024))  # 10 MB
LOG_BACKUP_COUNT: int = int(os.environ.get("LOG_BACKUP_COUNT", 5))
LOG_JSON_ERRORS: bool = os.environ.get("LOG_JSON_ERRORS", "true").lower() != "false"

APP_LOG_FILE: Path = LOG_DIR / "app.log"
ERROR_LOG_FILE: Path = LOG_DIR / "errors.log"

# Sentinel – prevents double-initialisation if the module is imported twice
_LOGGING_CONFIGURED: bool = False


# ──────────────────────────────────────────────
# Formatters
# ──────────────────────────────────────────────

class _PlainFormatter(logging.Formatter):
    """Human-readable formatter for app.log and the console."""

    FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATEFMT = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        return super().format(record)


class _ColouredConsoleFormatter(_PlainFormatter):
    """Adds ANSI colour codes to the console output for quick visual scanning."""

    _COLOURS = {
        logging.DEBUG:    "\033[36m",    # Cyan
        logging.INFO:     "\033[32m",    # Green
        logging.WARNING:  "\033[33m",    # Yellow
        logging.ERROR:    "\033[31m",    # Red
        logging.CRITICAL: "\033[1;31m",  # Bold Red
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self._COLOURS.get(record.levelno, "")
        message = super().format(record)
        return f"{colour}{message}{self._RESET}"


class _JsonFormatter(logging.Formatter):
    """
    Structured JSON formatter for errors.log.
    Each log record is a single-line JSON object, making it trivially
    parseable by tools like jq, Datadog, Splunk, CloudWatch, etc.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_object: dict = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # Attach exception info when present
        if record.exc_info:
            log_object["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Attach any extra fields callers pass via `extra={}`
        _STANDARD_FIELDS = {
            "args", "asctime", "created", "exc_info", "exc_text", "filename",
            "funcName", "id", "levelname", "levelno", "lineno", "module",
            "msecs", "message", "msg", "name", "pathname", "process",
            "processName", "relativeCreated", "stack_info", "thread",
            "threadName",
        }
        extras = {k: v for k, v in record.__dict__.items() if k not in _STANDARD_FIELDS}
        if extras:
            log_object["extra"] = extras

        try:
            return json.dumps(log_object, default=str)
        except Exception:  # pragma: no cover
            return json.dumps({"level": "ERROR", "message": "Log serialisation failed"})


# ──────────────────────────────────────────────
# Public initialiser
# ──────────────────────────────────────────────

def configure_logging() -> None:
    """
    Call once at application start-up (e.g., in main.py before the
    FastAPI app is created).  Subsequent calls are no-ops.
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, LOG_LEVEL, logging.INFO)

    # ── Root logger ──────────────────────────────
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # Handlers apply their own level filters

    # ── Handler 1: Coloured console (stdout) ────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(_ColouredConsoleFormatter(
        fmt=_PlainFormatter.FMT,
        datefmt=_PlainFormatter.DATEFMT,
    ))
    root.addHandler(console_handler)

    # ── Handler 2: app.log (rotating, plain text) ──
    app_handler = logging.handlers.RotatingFileHandler(
        filename=APP_LOG_FILE,
        mode="a",
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    app_handler.setLevel(numeric_level)
    app_handler.setFormatter(_PlainFormatter(
        fmt=_PlainFormatter.FMT,
        datefmt=_PlainFormatter.DATEFMT,
    ))
    root.addHandler(app_handler)

    # ── Handler 3: errors.log (rotating, JSON) ──
    error_handler = logging.handlers.RotatingFileHandler(
        filename=ERROR_LOG_FILE,
        mode="a",
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(
        _JsonFormatter() if LOG_JSON_ERRORS else _PlainFormatter(
            fmt=_PlainFormatter.FMT,
            datefmt=_PlainFormatter.DATEFMT,
        )
    )
    root.addHandler(error_handler)

    # ── Silence chatty third-party loggers ──────
    _QUIET_LOGGERS = [
        "uvicorn.access",   # Already handled by our middleware
        "watchfiles",       # File-watcher noise
        "multipart",
    ]
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _LOGGING_CONFIGURED = True

    # First log line – confirms the system is live
    _boot_logger = logging.getLogger("app.logger")
    _boot_logger.info(
        "Logging system initialised | level=%s | app_log=%s | error_log=%s",
        LOG_LEVEL, APP_LOG_FILE, ERROR_LOG_FILE,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Drop-in replacement for `logging.getLogger(name)`.
    Guarantees that `configure_logging()` has been called at least once
    before the logger is used (safe for module-level usage).
    """
    configure_logging()
    return logging.getLogger(name)
