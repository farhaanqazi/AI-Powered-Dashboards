"""SQLAlchemy models for the persistence layer.

DashboardRecord stores one row per session_key (the only tenancy key — see
src/auth.py allow_clerk_or_guest). `payload` holds the exact dict the API
previously stored in the in-process dashboard_storage dict. `trace_id` is
recorded for traceability but is NEVER a lookup key (this is the fix for
§11 issue 5 — there are no more orphan trace_id-keyed entries).
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class Base(DeclarativeBase):
    pass


class DashboardRecord(Base):
    __tablename__ = "dashboard_records"

    session_key: Mapped[str] = mapped_column(String(255), primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False)
    original_filename: Mapped[str] = mapped_column(Text, default="", nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )


class AnalysisHistoryRecord(Base):
    """Phase 10 S10.4 — per-owner history of past analyses.

    Append-only snapshots (unlike DashboardRecord, which holds only the
    *current* dashboard per session). ``owner_key`` is org-scoped when a Clerk
    org token is present (multi-tenancy: org members share history), else the
    Clerk user, else the guest session. Same DB/engine as DashboardRecord —
    a new table, not a new storage backend. TTL-bound like dashboards.
    """

    __tablename__ = "analysis_history"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner_key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    session_key: Mapped[str] = mapped_column(String(255), nullable=False)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    original_filename: Mapped[str] = mapped_column(Text, default="", nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
