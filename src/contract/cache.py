"""Phase 2 — fingerprint-keyed contract cache.

Reuses the optional Redis client from ``src.persistence.cache`` (in-memory
dict fallback when Redis is absent — same pattern as the rest of the app).

A *locked hit* (a cached contract whose ``locked`` flag is True, i.e. a
HITL-approved contract from Phase 7) lets the pipeline skip recompilation and
the LLM entirely — that gating lives in Phase 5; this module just stores and
retrieves contracts and exposes :func:`is_locked_hit`.
"""
from __future__ import annotations

from typing import Optional

from src import config
from src.contract.models import DatasetContract
from src.persistence.cache import build_cache_client

_KEY_PREFIX = "contract:"


class ContractCache:
    def __init__(self, client=None, ttl_seconds: int | None = None):
        # ``build_cache_client`` returns None when REDIS_URL is unset.
        self._client = client if client is not None else build_cache_client()
        self._ttl = (
            ttl_seconds if ttl_seconds is not None else config.DASHBOARD_TTL_SECONDS
        )
        self._mem: dict[str, str] = {}

    @staticmethod
    def _key(fingerprint: str) -> str:
        return f"{_KEY_PREFIX}{fingerprint}"

    def get(self, fingerprint: str) -> Optional[DatasetContract]:
        key = self._key(fingerprint)
        raw: Optional[str] = None
        if self._client is not None:
            raw = self._client.get(key)
        if raw is None:
            raw = self._mem.get(key)
        if raw is None:
            return None
        try:
            return DatasetContract.model_validate_json(raw)
        except Exception:
            return None

    def put(self, contract: DatasetContract) -> None:
        key = self._key(contract.schema_fingerprint)
        raw = contract.model_dump_json()
        self._mem[key] = raw
        if self._client is not None:
            self._client.setex(key, self._ttl, raw)

    def is_locked_hit(self, fingerprint: str) -> bool:
        """True iff a cached contract exists for this schema and is locked —
        the pipeline may then skip recompile + LLM."""
        c = self.get(fingerprint)
        return bool(c and c.locked)


_singleton: Optional[ContractCache] = None


def get_contract_cache() -> ContractCache:
    global _singleton
    if _singleton is None:
        _singleton = ContractCache()
    return _singleton


def reset_contract_cache_for_tests() -> None:
    global _singleton
    _singleton = None
