"""Database engine and session-factory construction.

Synchronous SQLAlchemy by design (see plan rationale). Default backend is a
local SQLite file so dev and CI need zero external infrastructure; production
sets DATABASE_URL to a Postgres-compatible URL. The schema in models.py is
written to be portable across both.
"""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from src import config
from src.persistence.models import Base


def _ensure_sqlite_dir(url: str) -> None:
    """SQLite needs its parent directory to exist before first connect."""
    prefix = "sqlite:///"
    if url.startswith(prefix):
        db_path = url[len(prefix):]
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def make_engine(url: str | None = None) -> Engine:
    url = url or config.DATABASE_URL
    _ensure_sqlite_dir(url)
    connect_args = {}
    if url.startswith("sqlite"):
        # Allow cross-thread use: FastAPI's threadpool + the SSE sync generator
        # touch the engine from different threads.
        connect_args["check_same_thread"] = False
    return create_engine(url, future=True, pool_pre_ping=True, connect_args=connect_args)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db(engine: Engine) -> None:
    """Create tables if they do not exist. Idempotent. Used for SQLite/dev and
    as a safety net; production runs Alembic migrations (Task 4)."""
    Base.metadata.create_all(engine)


def reset_db_for_tests(engine: Engine) -> None:
    """Drop and recreate all tables. Test-only."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
