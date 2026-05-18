"""Deterministic LLM response cache, keyed on the ground-truth hash.

The user prompt embeds the full ground-truth payload, so hashing
(provider | model | temperature | system | user) is exactly "key on the
ground-truth hash" required by S9.1: identical inputs ⇒ identical (free)
result, never a duplicate paid call.

In-process + TTL + bounded (LRU eviction). No new storage backend — same
posture as :mod:`src.contract.df_cache`. Best-effort: a cache fault must never
break the request.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from src import config


def ground_truth_key(
    *, provider: str, model: str, temperature: float, system: str, user: str
) -> str:
    h = hashlib.sha256()
    h.update(
        json.dumps(
            [provider, model, round(float(temperature), 4), system, user],
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    return h.hexdigest()


class LLMResponseCache:
    def __init__(self, ttl_seconds: int | None = None, max_entries: int | None = None):
        self._ttl = (
            ttl_seconds
            if ttl_seconds is not None
            else config.LLM_RESPONSE_CACHE_TTL_SECONDS
        )
        self._max = (
            max_entries
            if max_entries is not None
            else config.LLM_RESPONSE_CACHE_MAX_ENTRIES
        )
        # key -> (expires_at, parsed_json)
        self._store: "OrderedDict[str, tuple[float, Dict[str, Any]]]" = OrderedDict()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not config.LLM_RESPONSE_CACHE_ENABLED:
            return None
        hit = self._store.get(key)
        if hit is None:
            return None
        expires_at, value = hit
        if time.monotonic() >= expires_at:
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)  # LRU bump
        return json.loads(json.dumps(value))  # defensive copy

    def put(self, key: str, value: Dict[str, Any]) -> None:
        if not config.LLM_RESPONSE_CACHE_ENABLED or not isinstance(value, dict):
            return
        self._store[key] = (time.monotonic() + self._ttl, value)
        self._store.move_to_end(key)
        while len(self._store) > self._max:
            self._store.popitem(last=False)  # evict least-recently-used

    def clear(self) -> None:
        self._store.clear()


_singleton: Optional[LLMResponseCache] = None


def get_response_cache() -> LLMResponseCache:
    global _singleton
    if _singleton is None:
        _singleton = LLMResponseCache()
    return _singleton


def reset_response_cache_for_tests() -> None:
    global _singleton
    _singleton = None
