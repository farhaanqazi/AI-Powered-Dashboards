"""Observability layer — logging, request id, health, metrics, tracing."""
import json
import logging

import pytest

from src.observability import logging as obs_logging


def test_structlog_emits_json_to_stdout(capsys):
    obs_logging.configure_observability_logging(force=True)
    log = logging.getLogger("test.json.emit")
    log.info("hello", extra={"request_id": "r-1", "user": "alice"})
    captured = capsys.readouterr().out.strip().splitlines()
    assert captured, "expected at least one log line on stdout"
    parsed = json.loads(captured[-1])
    assert parsed["event"] == "hello" or parsed.get("message") == "hello"
    assert parsed["level"].lower() == "info"


def test_structlog_includes_request_id_from_contextvar(capsys):
    obs_logging.configure_observability_logging(force=True)
    token = obs_logging.request_id_var.set("ctx-req-99")
    try:
        log = logging.getLogger("test.ctx.req")
        log.info("contextual")
    finally:
        obs_logging.request_id_var.reset(token)
    captured = capsys.readouterr().out.strip().splitlines()
    parsed = json.loads(captured[-1])
    assert parsed.get("request_id") == "ctx-req-99"
