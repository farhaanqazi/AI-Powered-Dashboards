"""Phase 15 S15.4 — in-process model cache for what-if prediction.

The supervised model fitted during analysis (S15.1) is kept here, keyed on
``schema_fingerprint|target``, so ``/api/interact`` can score user-supplied
feature values with NO retrain. TTL + LRU bounded — single-container memory
stays small. On a miss the caller surfaces the same "re-run the analysis" UX as
an expired ``df_cache`` frame; we never silently refit.

The cached object is an opaque bundle (fitted estimator + the metadata needed to
rebuild a single aligned feature row). It holds a live sklearn estimator, so it
is deliberately process-local and never persisted/JSON-serialised.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from src import config

_LOCK = threading.Lock()
_STORE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


def _key(fingerprint: str, target: str) -> str:
    return f"{fingerprint}|{target}"


def put(fingerprint: str, target: str, bundle: Dict[str, Any]) -> None:
    if not config.ML_MODEL_CACHE_ENABLED or not fingerprint:
        return
    k = _key(fingerprint, target)
    with _LOCK:
        _STORE[k] = {"bundle": bundle, "ts": time.monotonic()}
        _STORE.move_to_end(k)
        while len(_STORE) > config.ML_MODEL_CACHE_MAX_ENTRIES:
            _STORE.popitem(last=False)


def get(fingerprint: str, target: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return the freshest non-expired bundle for ``fingerprint``.

    When ``target`` is omitted, the most-recently-stored model for this
    fingerprint is returned (the UI rarely needs to name the target)."""
    if not config.ML_MODEL_CACHE_ENABLED or not fingerprint:
        return None
    ttl = config.ML_MODEL_CACHE_TTL_SECONDS
    now = time.monotonic()
    with _LOCK:
        if target is not None:
            entry = _STORE.get(_key(fingerprint, target))
            candidates = [(_key(fingerprint, target), entry)] if entry else []
        else:
            candidates = [
                (k, v) for k, v in _STORE.items()
                if k.startswith(f"{fingerprint}|")
            ]
        # Evaluate newest-first; drop anything expired.
        best = None
        for k, entry in candidates:
            if entry is None:
                continue
            if now - entry["ts"] > ttl:
                _STORE.pop(k, None)
                continue
            if best is None or entry["ts"] > best[1]["ts"]:
                best = (k, entry)
        if best is None:
            return None
        _STORE.move_to_end(best[0])
        return best[1]["bundle"]


def reset_for_tests() -> None:
    with _LOCK:
        _STORE.clear()
