"""Liveness (`/healthz`) and readiness (`/readyz`) endpoints.

`/healthz` should be cheap and never fail — it answers "is the process alive?"
`/readyz` should fail when a hard dependency is unreachable. In Phase 0 there
are no external deps yet, so it always returns ready. Future phases (Postgres,
Redis, S3) extend `_readiness_checks()`.
"""
from __future__ import annotations

from typing import Awaitable, Callable, Dict, List, Tuple

from fastapi import APIRouter, Response

ReadinessCheck = Callable[[], Awaitable[Tuple[bool, str]]]

_checks: List[Tuple[str, ReadinessCheck]] = []


def register_readiness_check(name: str, check: ReadinessCheck) -> None:
    _checks.append((name, check))


def build_router() -> APIRouter:
    router = APIRouter(tags=["observability"])

    @router.get("/healthz")
    async def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @router.get("/readyz")
    async def readyz(response: Response) -> Dict[str, object]:
        results = {}
        all_ok = True
        for name, check in _checks:
            try:
                ok, detail = await check()
            except Exception as exc:
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            results[name] = {"ok": ok, "detail": detail}
            all_ok = all_ok and ok
        if not all_ok:
            response.status_code = 503
            return {"status": "not_ready", "checks": results}
        return {"status": "ready", "checks": results}

    return router
