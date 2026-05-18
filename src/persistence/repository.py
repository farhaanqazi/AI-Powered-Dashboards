"""DashboardRepository: the single facade the API uses for dashboard storage.

Replaces the in-process `dashboard_storage` dict. Keyed ONLY by session_key —
there is no trace_id-keyed entry (fixes §11 issue 5). Rows carry expires_at;
expired rows are invisible to get() and removed by purge_expired() (issue 21).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import uuid

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from src import config
from src.persistence.models import AnalysisHistoryRecord, DashboardRecord


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _as_aware(dt: datetime) -> datetime:
    """SQLite has no native tz support, so DateTime(timezone=True) columns load
    back offset-naive. Treat such values as UTC so they can be compared against
    the offset-aware _utcnow(). Postgres returns aware datetimes unchanged."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class DashboardRepository:
    def __init__(self, session_factory: sessionmaker[Session], ttl_seconds: int | None = None):
        self._sf = session_factory
        self._ttl = ttl_seconds if ttl_seconds is not None else config.DASHBOARD_TTL_SECONDS

    def save(self, session_key: str, *, trace_id: str, payload: dict) -> None:
        now = _utcnow()
        expires_at = now + timedelta(seconds=self._ttl)
        original_filename = payload.get("original_filename", "") or ""
        with self._sf() as s:
            existing = s.get(DashboardRecord, session_key)
            if existing is None:
                s.add(DashboardRecord(
                    session_key=session_key,
                    trace_id=trace_id,
                    original_filename=original_filename,
                    payload=payload,
                    created_at=now,
                    updated_at=now,
                    expires_at=expires_at,
                ))
            else:
                existing.trace_id = trace_id
                existing.original_filename = original_filename
                existing.payload = payload
                existing.updated_at = now
                existing.expires_at = expires_at
            s.commit()
        self.purge_expired()

    def get(self, session_key: str) -> Optional[dict]:
        now = _utcnow()
        with self._sf() as s:
            rec = s.get(DashboardRecord, session_key)
            if rec is None:
                return None
            if _as_aware(rec.expires_at) <= now:
                return None
            return rec.payload

    # --- Phase 10 S10.4: per-owner analysis history (append-only) ---

    def record_history(
        self, owner_key: str, *, session_key: str, trace_id: str, payload: dict
    ) -> None:
        """Append an immutable snapshot of a finished analysis."""
        now = _utcnow()
        with self._sf() as s:
            s.add(AnalysisHistoryRecord(
                id=str(uuid.uuid4()),
                owner_key=owner_key,
                session_key=session_key,
                trace_id=trace_id,
                original_filename=payload.get("original_filename", "") or "",
                payload=payload,
                created_at=now,
                expires_at=now + timedelta(seconds=self._ttl),
            ))
            s.commit()

    def list_history(self, owner_key: str, limit: int = 50) -> list[dict]:
        """Lightweight summaries (no payload) for the owner, newest first."""
        now = _utcnow()
        with self._sf() as s:
            rows = s.execute(
                select(AnalysisHistoryRecord)
                .where(
                    AnalysisHistoryRecord.owner_key == owner_key,
                    AnalysisHistoryRecord.expires_at > now,
                )
                .order_by(AnalysisHistoryRecord.created_at.desc())
                .limit(limit)
            ).scalars().all()
            return [
                {
                    "trace_id": r.trace_id,
                    "original_filename": r.original_filename,
                    "created_at": _as_aware(r.created_at).isoformat(),
                }
                for r in rows
            ]

    def get_history(self, owner_key: str, trace_id: str) -> Optional[dict]:
        """Reopen a past analysis the owner is entitled to (org/user/guest
        scoped). Returns the stored payload, or None if absent/expired."""
        now = _utcnow()
        with self._sf() as s:
            r = s.execute(
                select(AnalysisHistoryRecord)
                .where(
                    AnalysisHistoryRecord.owner_key == owner_key,
                    AnalysisHistoryRecord.trace_id == trace_id,
                    AnalysisHistoryRecord.expires_at > now,
                )
                .order_by(AnalysisHistoryRecord.created_at.desc())
                .limit(1)
            ).scalars().first()
            return r.payload if r is not None else None

    def purge_expired(self) -> int:
        now = _utcnow()
        with self._sf() as s:
            result = s.execute(
                delete(DashboardRecord).where(DashboardRecord.expires_at <= now)
            )
            s.execute(
                delete(AnalysisHistoryRecord).where(
                    AnalysisHistoryRecord.expires_at <= now
                )
            )
            s.commit()
            return result.rowcount or 0

    def count(self) -> int:
        with self._sf() as s:
            return s.execute(select(func.count()).select_from(DashboardRecord)).scalar() or 0

    def dispose(self) -> None:
        """Release the underlying engine's connection pool. Mainly for tests
        that simulate a process restart by rebuilding the singleton."""
        self._sf.kw["bind"].dispose()


_repository: Optional[DashboardRepository] = None


def get_repository() -> "CachedRepository":
    """App-wide singleton. Lazily builds the engine + optional Redis cache."""
    global _repository
    if _repository is None:
        from src.persistence import db
        from src.persistence.cache import CachedRepository, build_cache_client
        engine = db.make_engine()
        db.init_db(engine)
        base = DashboardRepository(db.make_session_factory(engine))
        _repository = CachedRepository(base, client=build_cache_client())
    return _repository


def reset_repository_for_tests(session_factory) -> None:
    global _repository
    from src.persistence.cache import CachedRepository, build_cache_client
    base = DashboardRepository(session_factory)
    _repository = CachedRepository(base, client=build_cache_client())
