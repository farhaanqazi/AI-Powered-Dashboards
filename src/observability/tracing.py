"""OpenTelemetry tracing — opt-in via OTEL_EXPORTER_OTLP_ENDPOINT.

When the endpoint env var is absent, tracing init is a no-op. This keeps the
current Hugging Face Space deployment working unchanged while letting any
managed deployment turn on traces with a single env var.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI

_enabled: bool = False
_configured: bool = False


def is_enabled() -> bool:
    return _enabled


def configure_tracing(app: Optional[FastAPI] = None, force: bool = False) -> None:
    global _enabled, _configured
    if _configured and not force:
        return
    _configured = True
    _enabled = False

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:
        return

    service_name = os.environ.get("OTEL_SERVICE_NAME", "ai-powered-dashboards")
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)

    if app is not None:
        FastAPIInstrumentor.instrument_app(app)

    _enabled = True
