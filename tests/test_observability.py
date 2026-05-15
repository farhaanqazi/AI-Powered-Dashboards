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


def test_request_id_header_round_trips(client):
    """X-Request-ID supplied by the caller is echoed in the response headers."""
    response = client.get("/api/dashboard", headers={"X-Request-ID": "trace-abc"})
    assert response.status_code == 200
    assert response.headers.get("x-request-id") == "trace-abc"


def test_request_id_is_generated_if_missing(client):
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    rid = response.headers.get("x-request-id")
    assert rid and len(rid) >= 8


def test_healthz_returns_200(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_returns_200_in_phase_0(client):
    """Phase 0 readiness has no real deps; it returns ok unconditionally.
    Later phases add Postgres + Redis dependency checks here."""
    response = client.get("/readyz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert "checks" in body


def test_metrics_endpoint_returns_prometheus_format(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "# HELP" in body
    assert "http_requests_total" in body


def test_metrics_records_request(client):
    client.get("/api/dashboard")
    response = client.get("/metrics")
    assert 'http_requests_total{' in response.text
    assert 'path="/api/dashboard"' in response.text


def test_pipeline_records_layer_metrics(client, upload_files):
    from src.observability import metrics as obs_metrics

    obs_metrics.pipeline_layer_seconds.clear()
    client.post("/api/upload", files=upload_files)

    samples = list(obs_metrics.pipeline_layer_seconds.collect())[0].samples
    layer_counts = {
        s.labels["layer"]: s.value
        for s in samples
        if s.name.endswith("_count")
    }
    for expected_layer in ("profiling", "classifying", "relating", "eda", "interpreting", "rendering"):
        assert layer_counts.get(expected_layer, 0) >= 1, f"no observation for layer={expected_layer}"


def test_tracing_init_is_noop_when_endpoint_unset(monkeypatch):
    """Without OTEL_EXPORTER_OTLP_ENDPOINT, tracing init must not raise or set
    a real exporter."""
    from src.observability import tracing

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    tracing.configure_tracing(force=True)
    assert tracing.is_enabled() is False


def test_tracing_init_succeeds_with_endpoint(monkeypatch):
    from src.observability import tracing

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "ai-powered-dashboards-test")
    tracing.configure_tracing(force=True)
    assert tracing.is_enabled() is True


def test_sentry_init_is_noop_without_dsn(monkeypatch):
    from src.observability import sentry as obs_sentry

    monkeypatch.delenv("SENTRY_DSN", raising=False)
    obs_sentry.configure_sentry(force=True)
    assert obs_sentry.is_enabled() is False


def test_sentry_init_is_enabled_with_dsn(monkeypatch):
    from src.observability import sentry as obs_sentry

    monkeypatch.setenv("SENTRY_DSN", "https://public@o0.ingest.sentry.io/0")
    obs_sentry.configure_sentry(force=True)
    assert obs_sentry.is_enabled() is True
