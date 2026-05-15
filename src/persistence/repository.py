"""DashboardRepository: the single facade the API uses for dashboard storage.

Replaces the in-process `dashboard_storage` dict. Keyed ONLY by session_key —
there is no trace_id-keyed entry (fixes §11 issue 5). Rows carry expires_at;
expired rows are invisible to get() and removed by purge_expired() (issue 21).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from src import config
from src.persistence.models import DashboardRecord


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

    def purge_expired(self) -> int:
        now = _utcnow()
        with self._sf() as s:
            result = s.execute(
                delete(DashboardRecord).where(DashboardRecord.expires_at <= now)
            )
            s.commit()
            return result.rowcount or 0

    def count(self) -> int:
        with self._sf() as s:
            return s.execute(select(func.count()).select_from(DashboardRecord)).scalar() or 0
