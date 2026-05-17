"""Job state + progress store.

Backed by Redis when ``config.REDIS_URL`` is set (shared across the API and a
separate Arq worker process — the production-correct path), otherwise an
in-process dict (single-container HF fallback; in-flight jobs lost on restart).

The interface is synchronous on purpose: the pipeline body runs in a worker
thread (``asyncio.to_thread`` / Arq), and the SSE endpoint reads via a cheap
threadpool hop. Records and events carry a TTL so nothing leaks.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional

from src import config

TERMINAL = {"done", "failed", "cancelled"}


def _now() -> float:
    return time.time()


class JobStore:
    def create(self, job_id: str, *, session_key: str, trace_id: str,
               filename: str) -> None: ...
    def append_event(self, job_id: str, event: Dict[str, Any]) -> None: ...
    def set_error(self, job_id: str, message: str) -> None: ...
    def get(self, job_id: str) -> Optional[Dict[str, Any]]: ...
    def events(self, job_id: str, since: int = 0) -> List[Dict[str, Any]]: ...
    def request_cancel(self, job_id: str) -> bool: ...
    def is_cancelled(self, job_id: str) -> bool: ...


def _apply_event(rec: Dict[str, Any], event: Dict[str, Any]) -> None:
    phase = event.get("phase")
    rec["updated_at"] = _now()
    if "percent" in event:
        rec["percent"] = event["percent"]
    if event.get("message"):
        rec["message"] = event["message"]
    if phase == "done":
        rec["status"] = "done"
        rec["phase"] = "done"
        if event.get("trace_id"):
            rec["trace_id"] = event["trace_id"]
    elif phase == "cancelled":
        rec["status"] = "cancelled"
        rec["phase"] = "cancelled"
    elif phase == "error":
        rec["status"] = "failed"
        rec["phase"] = "error"
        rec["error"] = event.get("message", "Analysis failed.")
    else:
        rec["status"] = "running"
        if phase:
            rec["phase"] = phase


class _MemoryJobStore(JobStore):
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._rec: Dict[str, Dict[str, Any]] = {}
        self._ev: Dict[str, List[Dict[str, Any]]] = {}
        self._cancel: set[str] = set()

    def create(self, job_id, *, session_key, trace_id, filename):
        with self._lock:
            self._rec[job_id] = {
                "job_id": job_id, "session_key": session_key,
                "trace_id": trace_id, "filename": filename,
                "status": "queued", "phase": "queued", "message": "Queued…",
                "percent": 0, "error": None,
                "created_at": _now(), "updated_at": _now(),
            }
            self._ev[job_id] = []

    def append_event(self, job_id, event):
        with self._lock:
            rec = self._rec.get(job_id)
            if rec is None:
                return
            _apply_event(rec, event)
            self._ev.setdefault(job_id, []).append(event)

    def set_error(self, job_id, message):
        self.append_event(job_id, {"phase": "error", "message": message,
                                   "percent": 100})

    def get(self, job_id):
        with self._lock:
            rec = self._rec.get(job_id)
            return dict(rec) if rec else None

    def events(self, job_id, since=0):
        with self._lock:
            return list(self._ev.get(job_id, [])[since:])

    def request_cancel(self, job_id):
        with self._lock:
            if job_id not in self._rec:
                return False
            self._cancel.add(job_id)
            return True

    def is_cancelled(self, job_id):
        with self._lock:
            return job_id in self._cancel


class _RedisJobStore(JobStore):
    def __init__(self, url: str) -> None:
        import redis  # redis==5.x, sync client

        self._r = redis.Redis.from_url(url, decode_responses=True)
        self._ttl = config.JOB_TTL_SECONDS

    def _k(self, job_id: str) -> str:
        return f"dijob:{job_id}"

    def create(self, job_id, *, session_key, trace_id, filename):
        rec = {
            "job_id": job_id, "session_key": session_key,
            "trace_id": trace_id, "filename": filename,
            "status": "queued", "phase": "queued", "message": "Queued…",
            "percent": 0, "error": None,
            "created_at": _now(), "updated_at": _now(),
        }
        p = self._r.pipeline()
        p.set(self._k(job_id), json.dumps(rec), ex=self._ttl)
        p.delete(f"{self._k(job_id)}:ev")
        p.execute()

    def append_event(self, job_id, event):
        raw = self._r.get(self._k(job_id))
        if raw is None:
            return
        rec = json.loads(raw)
        _apply_event(rec, event)
        p = self._r.pipeline()
        p.set(self._k(job_id), json.dumps(rec), ex=self._ttl)
        p.rpush(f"{self._k(job_id)}:ev", json.dumps(event))
        p.expire(f"{self._k(job_id)}:ev", self._ttl)
        p.execute()

    def set_error(self, job_id, message):
        self.append_event(job_id, {"phase": "error", "message": message,
                                   "percent": 100})

    def get(self, job_id):
        raw = self._r.get(self._k(job_id))
        return json.loads(raw) if raw else None

    def events(self, job_id, since=0):
        rows = self._r.lrange(f"{self._k(job_id)}:ev", since, -1)
        return [json.loads(x) for x in rows]

    def request_cancel(self, job_id):
        if self._r.get(self._k(job_id)) is None:
            return False
        self._r.set(f"{self._k(job_id)}:cancel", "1", ex=self._ttl)
        return True

    def is_cancelled(self, job_id):
        return self._r.get(f"{self._k(job_id)}:cancel") == "1"


_store: Optional[JobStore] = None
_store_lock = threading.Lock()


def get_job_store() -> JobStore:
    """Singleton. Redis-backed iff REDIS_URL is configured, else in-memory."""
    global _store
    if _store is not None:
        return _store
    with _store_lock:
        if _store is None:
            if config.REDIS_URL:
                try:
                    _store = _RedisJobStore(config.REDIS_URL)
                except Exception:  # pragma: no cover - fall back gracefully
                    _store = _MemoryJobStore()
            else:
                _store = _MemoryJobStore()
    return _store
