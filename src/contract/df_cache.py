"""Cleaned-DataFrame cache (Phase 7 HITL re-render + Phase 14 interactivity).

Two tiers, same fingerprint key:

* **Transient** (Phase 7): in-process dict + optional Redis, TTL-bound. Fast,
  but wiped on container restart — which previously killed follow-up
  Ask/Interact ("working data has expired").
* **Durable** (Phase 14 S14.1 — Gap A): a fingerprint-keyed Parquet file on a
  local spool dir. This mirrors the already-accepted ``JOB_SPOOL_DIR``
  filesystem-spool pattern; it is NOT a new storage backend. It lets
  Ask/Interact survive a restart instead of failing.

``get()`` falls back mem → client → parquet. Everything is gated by config and
fully disableable; on a miss/disabled callers degrade gracefully exactly as
before.
"""
from __future__ import annotations

import base64
import os
import pickle
import time
from typing import Optional

import pandas as pd

from src import config
from src.persistence.cache import build_cache_client

try:  # structured logger if available, std logging otherwise
    from src.logger import get_logger
    logger = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)

_KEY_PREFIX = "df:"


def _safe_fp(fingerprint: str) -> str:
    """Filesystem-safe form of a schema fingerprint (it is already a hex/slug
    in practice, but never trust it into a path)."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_"
                    for c in str(fingerprint))[:128]


class DataFrameCache:
    def __init__(self, client=None, ttl_seconds: int | None = None,
                 durable_dir: str | None = None):
        self._client = client if client is not None else build_cache_client()
        self._ttl = (
            ttl_seconds
            if ttl_seconds is not None
            else config.CLEANED_DF_CACHE_TTL_SECONDS
        )
        self._mem: dict[str, pd.DataFrame] = {}
        self._durable_dir = (
            durable_dir
            if durable_dir is not None
            else config.CLEANED_DF_DURABLE_DIR
        )

    @staticmethod
    def _key(fingerprint: str) -> str:
        return f"{_KEY_PREFIX}{fingerprint}"

    # --- durable Parquet tier (Gap A) ---------------------------------------

    def _durable_path(self, fingerprint: str) -> str:
        return os.path.join(self._durable_dir, f"{_safe_fp(fingerprint)}.parquet")

    def _durable_put(self, fingerprint: str, df: pd.DataFrame) -> None:
        if not config.CLEANED_DF_DURABLE_ENABLED:
            return
        try:
            os.makedirs(self._durable_dir, exist_ok=True)
            path = self._durable_path(fingerprint)
            try:
                df.to_parquet(path, index=False)
            except Exception:
                # No parquet engine (pyarrow/fastparquet) → pickle fallback so
                # restart-survivability still holds. Best-effort, never raise.
                with open(path + ".pkl", "wb") as fh:
                    pickle.dump(df, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:  # pragma: no cover - defensive
            logger.info("durable df write skipped (%s)", e)

    def _durable_get(self, fingerprint: str) -> Optional[pd.DataFrame]:
        if not config.CLEANED_DF_DURABLE_ENABLED:
            return None
        path = self._durable_path(fingerprint)
        pkl = path + ".pkl"
        try:
            target = path if os.path.exists(path) else (
                pkl if os.path.exists(pkl) else None
            )
            if target is None:
                return None
            age = time.time() - os.path.getmtime(target)
            if age > config.CLEANED_DF_DURABLE_TTL_SECONDS:
                self._durable_evict(fingerprint)
                return None
            if target.endswith(".pkl"):
                with open(target, "rb") as fh:
                    return pickle.load(fh)
            return pd.read_parquet(target)
        except Exception as e:  # pragma: no cover - defensive
            logger.info("durable df read failed (%s)", e)
            return None

    def _durable_evict(self, fingerprint: str) -> None:
        for p in (self._durable_path(fingerprint),
                  self._durable_path(fingerprint) + ".pkl"):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    # --- public API ---------------------------------------------------------

    def put(self, fingerprint: str, df: pd.DataFrame) -> None:
        if df is None:
            return
        if config.CLEANED_DF_CACHE_ENABLED:
            key = self._key(fingerprint)
            self._mem[key] = df.copy()
            if self._client is not None:
                try:
                    blob = base64.b64encode(pickle.dumps(df)).decode("ascii")
                    self._client.setex(key, self._ttl, blob)
                except Exception:
                    pass  # transient cache is best-effort
        # Durable tier is independent of the transient toggle: it is the
        # restart-survivability guarantee Phase 14 depends on.
        self._durable_put(fingerprint, df)

    def get(self, fingerprint: str) -> Optional[pd.DataFrame]:
        key = self._key(fingerprint)
        if config.CLEANED_DF_CACHE_ENABLED:
            if key in self._mem:
                return self._mem[key].copy()
            if self._client is not None:
                try:
                    raw = self._client.get(key)
                    if raw:
                        # pickle is safe: we are the only writer of df:* keys.
                        return pickle.loads(base64.b64decode(raw))
                except Exception:
                    pass  # fall through to durable tier
        # Restart / eviction path: rehydrate from the durable Parquet copy and
        # re-warm the transient tier so subsequent hits are fast.
        df = self._durable_get(fingerprint)
        if df is not None and config.CLEANED_DF_CACHE_ENABLED:
            self._mem[key] = df.copy()
        return df


_singleton: Optional[DataFrameCache] = None


def get_df_cache() -> DataFrameCache:
    global _singleton
    if _singleton is None:
        _singleton = DataFrameCache()
    return _singleton


def reset_df_cache_for_tests() -> None:
    global _singleton
    _singleton = None
