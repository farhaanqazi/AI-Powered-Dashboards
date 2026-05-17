"""Transient cleaned-DataFrame cache (Phase 7 HITL re-render support).

Holds the post-ingest cleaned frame just long enough for a human schema
override to re-run L3→render against the corrected roles. Deliberately NOT a
durable backend: TTL-bound, schema-fingerprint-keyed, Redis-or-in-process,
and gated by ``config.CLEANED_DF_CACHE_ENABLED``. When disabled or on a miss,
callers fall back to contract-only re-derivation.
"""
from __future__ import annotations

import base64
import pickle
from typing import Optional

import pandas as pd

from src import config
from src.persistence.cache import build_cache_client

_KEY_PREFIX = "df:"


class DataFrameCache:
    def __init__(self, client=None, ttl_seconds: int | None = None):
        self._client = client if client is not None else build_cache_client()
        self._ttl = (
            ttl_seconds
            if ttl_seconds is not None
            else config.CLEANED_DF_CACHE_TTL_SECONDS
        )
        self._mem: dict[str, pd.DataFrame] = {}

    @staticmethod
    def _key(fingerprint: str) -> str:
        return f"{_KEY_PREFIX}{fingerprint}"

    def put(self, fingerprint: str, df: pd.DataFrame) -> None:
        if not config.CLEANED_DF_CACHE_ENABLED or df is None:
            return
        key = self._key(fingerprint)
        self._mem[key] = df.copy()
        if self._client is not None:
            try:
                blob = base64.b64encode(pickle.dumps(df)).decode("ascii")
                self._client.setex(key, self._ttl, blob)
            except Exception:
                pass  # cache is best-effort; never break the request

    def get(self, fingerprint: str) -> Optional[pd.DataFrame]:
        if not config.CLEANED_DF_CACHE_ENABLED:
            return None
        key = self._key(fingerprint)
        if key in self._mem:
            return self._mem[key].copy()
        if self._client is not None:
            try:
                raw = self._client.get(key)
                if raw:
                    # pickle is safe here: we are the only writer of df:* keys.
                    return pickle.loads(base64.b64decode(raw))
            except Exception:
                return None
        return None


_singleton: Optional[DataFrameCache] = None


def get_df_cache() -> DataFrameCache:
    global _singleton
    if _singleton is None:
        _singleton = DataFrameCache()
    return _singleton


def reset_df_cache_for_tests() -> None:
    global _singleton
    _singleton = None
