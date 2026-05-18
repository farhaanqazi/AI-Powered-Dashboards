"""Optional Redis read-through cache for DashboardRepository.

Disabled by default: when REDIS_URL is empty, build_cache_client() returns None
and CachedRepository becomes a transparent passthrough. This mirrors the Phase 0
optional-OTel/Sentry pattern — adding a scaling capability without forcing infra
on dev or CI.
"""
from __future__ import annotations

import json
from typing import Optional

from src import config
from src.persistence.repository import DashboardRepository

_KEY_PREFIX = "dash:"


def build_cache_client(url: str | None = None):
    url = url if url is not None else config.REDIS_URL
    if not url:
        return None
    try:
        import redis
    except ImportError:
        return None
    return redis.Redis.from_url(url, decode_responses=True)


class CachedRepository:
    """Wraps a DashboardRepository. Read-through on get(); write-through +
    cache-invalidation on save(). Transparent passthrough when client is None."""

    def __init__(self, base: DashboardRepository, client=None, ttl_seconds: int | None = None):
        self._base = base
        self._client = client
        self._ttl = ttl_seconds if ttl_seconds is not None else config.DASHBOARD_TTL_SECONDS

    @staticmethod
    def _key(session_key: str) -> str:
        return f"{_KEY_PREFIX}{session_key}"

    def save(self, session_key: str, *, trace_id: str, payload: dict) -> None:
        self._base.save(session_key, trace_id=trace_id, payload=payload)
        if self._client is not None:
            self._client.delete(self._key(session_key))

    def get(self, session_key: str) -> Optional[dict]:
        if self._client is not None:
            cached = self._client.get(self._key(session_key))
            if cached is not None:
                return json.loads(cached)
        value = self._base.get(session_key)
        if value is not None and self._client is not None:
            self._client.setex(self._key(session_key), self._ttl, json.dumps(value))
        return value

    # Phase 10 S10.4 — history is not read-through cached (list/reopen are
    # infrequent, owner-scoped, and must always reflect the source of truth).
    def record_history(self, owner_key: str, *, session_key: str,
                        trace_id: str, payload: dict) -> None:
        self._base.record_history(
            owner_key, session_key=session_key, trace_id=trace_id,
            payload=payload,
        )

    def list_history(self, owner_key: str, limit: int = 50) -> list:
        return self._base.list_history(owner_key, limit=limit)

    def get_history(self, owner_key: str, trace_id: str):
        return self._base.get_history(owner_key, trace_id)

    def purge_expired(self) -> int:
        return self._base.purge_expired()

    def count(self) -> int:
        return self._base.count()

    def dispose(self) -> None:
        self._base.dispose()
