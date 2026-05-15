"""Sentry SDK init — opt-in via SENTRY_DSN."""
from __future__ import annotations

import os

_enabled: bool = False
_configured: bool = False


def is_enabled() -> bool:
    return _enabled


def configure_sentry(force: bool = False) -> None:
    global _enabled, _configured
    if _configured and not force:
        return
    _configured = True
    _enabled = False

    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        return

    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
    except ImportError:
        return

    environment = os.environ.get("SENTRY_ENVIRONMENT", "production")
    traces_sample_rate = float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.0"))

    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        traces_sample_rate=traces_sample_rate,
        integrations=[StarletteIntegration(), FastApiIntegration()],
    )
    _enabled = True
