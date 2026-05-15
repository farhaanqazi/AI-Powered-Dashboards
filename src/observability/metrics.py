"""Prometheus metrics — request counters/histograms + a /metrics endpoint.

The pipeline-layer histogram (`pipeline_layer_seconds`) is consumed by
`src/core/pipeline.py` in Task 16.
"""
from __future__ import annotations

import time
from typing import Callable

from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

registry = CollectorRegistry()

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    labelnames=("method", "path", "status"),
    registry=registry,
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    labelnames=("method", "path"),
    registry=registry,
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

http_requests_in_flight = Gauge(
    "http_requests_in_flight",
    "Currently in-flight HTTP requests",
    registry=registry,
)

pipeline_layer_seconds = Histogram(
    "pipeline_layer_seconds",
    "Time spent in each pipeline layer",
    labelnames=("layer",),
    registry=registry,
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        path = request.url.path
        method = request.method
        http_requests_in_flight.inc()
        start = time.perf_counter()
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            http_requests_total.labels(method=method, path=path, status="500").inc()
            raise
        finally:
            http_requests_in_flight.dec()
            http_request_duration_seconds.labels(method=method, path=path).observe(
                time.perf_counter() - start
            )
        http_requests_total.labels(method=method, path=path, status=str(status)).inc()
        return response


def build_router() -> APIRouter:
    router = APIRouter(tags=["observability"])

    @router.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(
            content=generate_latest(registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    return router
