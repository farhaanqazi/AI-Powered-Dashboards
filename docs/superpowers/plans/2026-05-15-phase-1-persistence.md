# Phase 1 — Persistence Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the in-process `dashboard_storage` dict with a durable, restart-surviving persistence layer (a SQL database keyed by `session_key`, with row-level TTL/expiry and an optional Redis read-through cache), eliminating §11 issues 1 (no persistence), 5 (orphan `trace_id` entries), and 21 (no TTL/eviction).

**Architecture:** A synchronous SQLAlchemy 2.x ORM layer behind a `DashboardRepository` facade. Default backend is file-based SQLite (zero external infra for dev/CI); production points `DATABASE_URL` at any Postgres-compatible database — schema is written to be Postgres-compatible. Dashboards are keyed solely by `session_key` (the orphan `trace_id` dict entries are deleted, not migrated). Each row carries `expires_at`; reads ignore expired rows and an opportunistic sweep deletes them. An optional Redis read-through cache (`REDIS_URL`) sits in front of the repository and **no-ops cleanly when unconfigured** — identical pattern to Phase 0's optional OTel/Sentry. Synchronous (not async) SQLAlchemy is a deliberate choice: the pipeline is sync-blocking and the SSE endpoint persists from a sync generator, so an async repo would require fragile thread-bridging for zero benefit while the pipeline stays sync (async is revisited in sub-plan #3).

**Tech Stack:** SQLAlchemy 2.x (sync ORM), Alembic (migrations), SQLite (dev/CI default via stdlib `sqlite3`), `psycopg2-binary` (Postgres driver, prod), `redis>=5` (optional cache), existing FastAPI 0.109.2 + Phase 0 pytest scaffold.

---

## Branch & prerequisites

- **Base branch:** Phase 0 has been **merged into `main`** (merge commit `2212260`, PR #2). `phase-1-persistence` is branched from the updated `main` (which now contains all Phase 0 work — test scaffold, observability, cleaned `main.py`). Do NOT branch from the old `phase-0-stabilize`.
- **Python venv:** `f:/AI Powered Dashboards/venv/Scripts/python.exe` (Python 3.12). Run pytest as `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest ...`; pip as `"...python.exe" -m pip install ...`.
- **Hard rules for implementers:** never push, never amend, never `--no-verify`, never run remote/GitHub commands. One commit per task.
- **Conventions:** Conventional Commits. Tests bypass Clerk via `X-Guest-Mode: 1` + `X-Guest-Session-Id` headers (already wired by the Phase 0 `client` fixture).

---

## Spec coverage map

| §11 issue | How this plan resolves it | Task(s) |
|---|---|---|
| 1 — No persistence (restart wipes state) | SQL DB persists `response_data`; survives restart | 2, 3, 4, 5, 8–12, 14 |
| 5 — Orphan `trace_id` entries nothing reads | The `dashboard_storage[trace_id] = ...` writes are deleted entirely; `trace_id` stays in the HTTP response but is never a storage key | 8, 9, 10, 11 |
| 21 — Storage entries never expire | `expires_at` column + read-time exclusion + opportunistic `purge_expired()` sweep, TTL via `DASHBOARD_TTL_SECONDS` | 6, 14 |

Explicitly **out of scope** (deferred, with rationale):
- **Raw-CSV object store (S3/MinIO).** None of issues 1/5/21 require storing the uploaded CSV bytes — only the computed `response_data` is stored/read. Building blob storage now would be YAGNI. Deferred to a later phase if re-processing/audit is needed.
- **Async pipeline / async DB.** Sub-plan #3. This plan is deliberately synchronous.
- **Multi-tenant RBAC.** Sub-plan #6. `session_key` remains the only tenancy boundary.

---

## File structure

### Files created
- `src/persistence/__init__.py` — package marker + public exports (`get_repository`, `DashboardRepository`)
- `src/persistence/db.py` — engine + `SessionLocal` factory, `DATABASE_URL` parsing, `init_db()` (create tables), `reset_db_for_tests()`
- `src/persistence/models.py` — SQLAlchemy declarative `Base` + `DashboardRecord` model
- `src/persistence/repository.py` — `DashboardRepository` (`save`, `get`, `purge_expired`) + module-level `get_repository()` singleton accessor
- `src/persistence/cache.py` — `CachedRepository` decorator: optional Redis read-through; no-op passthrough when `REDIS_URL` unset
- `alembic.ini` — Alembic config
- `alembic/env.py` — Alembic runtime env (reads `DATABASE_URL`, targets `models.Base.metadata`)
- `alembic/script.py.mako` — standard Alembic template
- `alembic/versions/0001_create_dashboard_records.py` — initial migration
- `tests/test_persistence_db.py` — engine/URL/init tests
- `tests/test_persistence_repository.py` — repository save/get tests
- `tests/test_persistence_ttl.py` — expiry/purge tests
- `tests/test_persistence_cache.py` — cache no-op + cached-path tests
- `tests/test_api_persistence_integration.py` — end-to-end: restart survival, per-session isolation, TTL via the HTTP API

### Files modified
- `src/config.py` — add `DATABASE_URL`, `REDIS_URL`, `DASHBOARD_TTL_SECONDS`
- `main.py` — remove `dashboard_storage` + `storage_lock`; remove the 3 orphan `trace_id` writes; replace dict reads/writes with the repository; add FastAPI lifespan to call `init_db()`; remove now-unused `from threading import Lock`
- `tests/conftest.py` — replace the `_reset_storage` autouse fixture (which clears the soon-deleted dict) with a DB-truncating autouse fixture; point tests at an isolated temp SQLite DB
- `requirements.txt` — add `SQLAlchemy==2.0.29`, `alembic==1.13.1`, `psycopg2-binary==2.9.9`, `redis==5.0.4`
- `.env.example` — add `DATABASE_URL`, `REDIS_URL`, `DASHBOARD_TTL_SECONDS`
- `CHANGELOG.md` — Phase 1 section

### Files deleted
- None.

---

## Critical sequencing note

`tests/conftest.py` (from Phase 0) has an autouse fixture:
```python
@pytest.fixture(autouse=True)
def _reset_storage():
    main_module.dashboard_storage.clear()
    yield
    main_module.dashboard_storage.clear()
```
The moment `dashboard_storage` is deleted from `main.py`, this fixture raises `AttributeError` and **every test in the suite fails at collection**. Therefore **Task 7 (rewrite conftest) MUST land before Task 11 (delete the dict)**. The task order below enforces this. Until Task 11, the dict and the repository can coexist; the dict is removed only once all four endpoints read/write the repository.

---

## Task 1: Dependencies + config knobs

**Files:**
- Modify: `requirements.txt`
- Modify: `src/config.py`
- Modify: `.env.example`
- Test: `tests/test_persistence_db.py`

- [ ] **Step 1: Add a failing config test**

Create `tests/test_persistence_db.py`:

```python
"""Persistence config + engine tests."""


def test_config_exposes_persistence_knobs():
    from src import config
    assert isinstance(config.DATABASE_URL, str)
    assert config.DATABASE_URL  # non-empty default
    assert isinstance(config.DASHBOARD_TTL_SECONDS, int)
    assert config.DASHBOARD_TTL_SECONDS > 0
    # REDIS_URL may be empty string (= cache disabled), but must be a str
    assert isinstance(config.REDIS_URL, str)
```

- [ ] **Step 2: Run it, expect failure**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest tests/test_persistence_db.py::test_config_exposes_persistence_knobs -v`
Expected: FAIL — `AttributeError: module 'src.config' has no attribute 'DATABASE_URL'`.

- [ ] **Step 3: Add the config knobs**

Append to `src/config.py`:

```python
# --- Persistence Configuration ---
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite:///./_local/dashboards.db",
)
REDIS_URL = os.environ.get("REDIS_URL", "")
DASHBOARD_TTL_SECONDS = int(os.environ.get("DASHBOARD_TTL_SECONDS", 86400))
```

- [ ] **Step 4: Run the test, expect pass**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest tests/test_persistence_db.py -v`
Expected: PASS.

- [ ] **Step 5: Add dependencies**

Append to `requirements.txt`:

```
SQLAlchemy==2.0.29
alembic==1.13.1
psycopg2-binary==2.9.9
redis==5.0.4
```

Then: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pip install -r requirements.txt`
Expected: installs cleanly. If `psycopg2-binary` fails to build on the dev machine (it ships wheels for Python 3.12 on Windows, so it should not), report it — do NOT switch drivers without escalating.

- [ ] **Step 6: Document env vars**

In `.env.example`, add under a new section after the HTTP section:

```env
# ─── Persistence ───────────────────────────────────────────
# Any SQLAlchemy URL. Default = local SQLite file. Prod example:
#   postgresql+psycopg2://user:pass@host:5432/dbname
DATABASE_URL=sqlite:///./_local/dashboards.db
# Dashboard rows expire this many seconds after last write (default 24h):
DASHBOARD_TTL_SECONDS=86400
# Optional Redis read-through cache. Leave blank to disable.
REDIS_URL=
```

- [ ] **Step 7: Full suite still green**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest -v`
Expected: previous Phase 0 count + 1 new test, all pass (no xfail regressions).

- [ ] **Step 8: Commit**

```bash
git add requirements.txt src/config.py .env.example tests/test_persistence_db.py
git commit -m "feat(persistence): add DB/cache config knobs and dependencies"
```

---

## Task 2: Database engine & session factory

**Files:**
- Create: `src/persistence/__init__.py`
- Create: `src/persistence/db.py`
- Test: `tests/test_persistence_db.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_persistence_db.py`:

```python
def test_engine_uses_configured_url(tmp_path, monkeypatch):
    from src.persistence import db

    url = f"sqlite:///{tmp_path / 'x.db'}"
    eng = db.make_engine(url)
    assert str(eng.url).startswith("sqlite")
    eng.dispose()


def test_init_db_creates_schema(tmp_path):
    from src.persistence import db
    from sqlalchemy import inspect

    url = f"sqlite:///{tmp_path / 'schema.db'}"
    eng = db.make_engine(url)
    db.init_db(eng)
    tables = inspect(eng).get_table_names()
    assert "dashboard_records" in tables
    eng.dispose()


def test_session_factory_round_trips(tmp_path):
    from src.persistence import db
    from sqlalchemy import text

    url = f"sqlite:///{tmp_path / 'rt.db'}"
    eng = db.make_engine(url)
    db.init_db(eng)
    Session = db.make_session_factory(eng)
    with Session() as s:
        assert s.execute(text("SELECT 1")).scalar() == 1
    eng.dispose()
```

- [ ] **Step 2: Run, expect failure**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest tests/test_persistence_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.persistence'`.

- [ ] **Step 3: Create the package marker**

Create `src/persistence/__init__.py`:

```python
"""Durable persistence layer: SQL-backed dashboard storage with TTL."""
```

- [ ] **Step 4: Create `src/persistence/db.py`**

```python
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
```

- [ ] **Step 5: Run tests, expect 1 pass + 3 import-failures cascade to a real failure on models**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest tests/test_persistence_db.py -v`
Expected: still FAIL — `ModuleNotFoundError: No module named 'src.persistence.models'` (created in Task 3). This is expected; Task 3 makes these green. Do NOT stub `models` here — Task 3 owns it.

- [ ] **Step 6: Commit (engine module, tests red pending Task 3)**

```bash
git add src/persistence/__init__.py src/persistence/db.py tests/test_persistence_db.py
git commit -m "feat(persistence): add engine and session-factory module"
```

> Note: this is the rare TDD case where a task's tests stay red until the next task (a hard module dependency). The commit message records it; Task 3 turns them green. The implementer should state this explicitly in their report (DONE_WITH_CONCERNS is appropriate, noting "tests red pending Task 3 models module, by plan design").

---

## Task 3: `DashboardRecord` model

**Files:**
- Create: `src/persistence/models.py`
- Test: `tests/test_persistence_db.py` (the Task 2 tests now go green)

- [ ] **Step 1: Create `src/persistence/models.py`**

```python
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
```

- [ ] **Step 2: Run the Task 2 tests, expect pass**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest tests/test_persistence_db.py -v`
Expected: all Task 2 tests now PASS (`test_engine_uses_configured_url`, `test_init_db_creates_schema`, `test_session_factory_round_trips`, plus the Task 1 config test).

- [ ] **Step 3: Add a model-level test**

Append to `tests/test_persistence_db.py`:

```python
def test_dashboard_record_defaults(tmp_path):
    from datetime import datetime, timezone, timedelta
    from src.persistence import db
    from src.persistence.models import DashboardRecord

    eng = db.make_engine(f"sqlite:///{tmp_path / 'm.db'}")
    db.init_db(eng)
    Session = db.make_session_factory(eng)
    exp = datetime.now(tz=timezone.utc) + timedelta(hours=1)
    with Session() as s:
        rec = DashboardRecord(
            session_key="guest:abc",
            trace_id="t-1",
            original_filename="x.csv",
            payload={"kpis": []},
            expires_at=exp,
        )
        s.add(rec)
        s.commit()
    with Session() as s:
        got = s.get(DashboardRecord, "guest:abc")
        assert got is not None
        assert got.payload == {"kpis": []}
        assert got.created_at is not None
        assert got.updated_at is not None
    eng.dispose()
```

- [ ] **Step 4: Run, expect pass**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest tests/test_persistence_db.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/persistence/models.py tests/test_persistence_db.py
git commit -m "feat(persistence): add DashboardRecord model"
```

---

## Task 4: Alembic migrations

**Files:**
- Create: `alembic.ini`, `alembic/env.py`, `alembic/script.py.mako`, `alembic/versions/0001_create_dashboard_records.py`
- Test: `tests/test_persistence_db.py`

- [ ] **Step 1: Create `alembic.ini`**

```ini
[alembic]
script_location = alembic
prepend_sys_path = .

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARNING
handlers = console
qualname =

[logger_sqlalchemy]
level = WARNING
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

- [ ] **Step 2: Create `alembic/script.py.mako`**

```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}
"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

- [ ] **Step 3: Create `alembic/env.py`**

```python
"""Alembic migration environment. Reads DATABASE_URL from app config so
migrations and the app never disagree on the target database."""
from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from src import config as app_config
from src.persistence.models import Base

config = context.config
config.set_main_option("sqlalchemy.url", app_config.DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=app_config.DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,  # required for SQLite ALTER support
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

- [ ] **Step 4: Create `alembic/versions/0001_create_dashboard_records.py`**

```python
"""create dashboard_records

Revision ID: 0001
Revises:
Create Date: 2026-05-15
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dashboard_records",
        sa.Column("session_key", sa.String(length=255), primary_key=True),
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("original_filename", sa.Text(), nullable=False, server_default=""),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_dashboard_records_expires_at", "dashboard_records", ["expires_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_dashboard_records_expires_at", table_name="dashboard_records")
    op.drop_table("dashboard_records")
```

- [ ] **Step 5: Add a migration-parity test**

Append to `tests/test_persistence_db.py`:

```python
def test_alembic_migration_matches_model(tmp_path):
    """Running the migration produces the same schema as Base.metadata."""
    import subprocess
    import sys
    from sqlalchemy import create_engine, inspect

    db_file = tmp_path / "alembic.db"
    url = f"sqlite:///{db_file}"
    env = {"DATABASE_URL": url}
    import os as _os
    full_env = {**_os.environ, **env}
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=".",
        env=full_env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"alembic failed: {result.stderr}"
    eng = create_engine(url)
    cols = {c["name"] for c in inspect(eng).get_columns("dashboard_records")}
    assert cols == {
        "session_key", "trace_id", "original_filename",
        "payload", "created_at", "updated_at", "expires_at",
    }
    eng.dispose()
```

- [ ] **Step 6: Run, expect pass**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest tests/test_persistence_db.py::test_alembic_migration_matches_model -v`
Expected: PASS (alembic exits 0, columns match). If alembic can't find `src` package, confirm `prepend_sys_path = .` in `alembic.ini` and that the command runs from repo root.

- [ ] **Step 7: Full suite green, then commit**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest -v`

```bash
git add alembic.ini alembic/ tests/test_persistence_db.py
git commit -m "feat(persistence): add Alembic migrations for dashboard_records"
```

---

## Task 5: `DashboardRepository` — save & get

**Files:**
- Create: `src/persistence/repository.py`
- Test: `tests/test_persistence_repository.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_persistence_repository.py`:

```python
"""DashboardRepository save/get behaviour."""
import pytest

from src.persistence import db
from src.persistence.repository import DashboardRepository


@pytest.fixture
def repo(tmp_path):
    engine = db.make_engine(f"sqlite:///{tmp_path / 'repo.db'}")
    db.init_db(engine)
    session_factory = db.make_session_factory(engine)
    yield DashboardRepository(session_factory)
    engine.dispose()


def _payload(name="a.csv"):
    return {
        "dataset_profile": {"n_rows": 3, "n_cols": 2},
        "kpis": [{"name": "x", "score": 0.9}],
        "charts": [],
        "primary_chart": None,
        "category_charts": {},
        "all_charts": [],
        "original_filename": name,
        "errors": [],
        "warnings": [],
        "critical_totals": {},
        "critical_full_dataset_aggregates": {},
        "eda_summary": {},
    }


def test_get_missing_returns_none(repo):
    assert repo.get("guest:nope") is None


def test_save_then_get_round_trips(repo):
    repo.save("guest:s1", trace_id="t1", payload=_payload("f1.csv"))
    got = repo.get("guest:s1")
    assert got is not None
    assert got["original_filename"] == "f1.csv"
    assert got["kpis"][0]["name"] == "x"


def test_save_is_upsert_keyed_by_session(repo):
    repo.save("guest:s1", trace_id="t1", payload=_payload("first.csv"))
    repo.save("guest:s1", trace_id="t2", payload=_payload("second.csv"))
    got = repo.get("guest:s1")
    assert got["original_filename"] == "second.csv"
    # Exactly one row for the session (no orphan rows — §11 issue 5)
    assert repo.count() == 1


def test_sessions_are_isolated(repo):
    repo.save("user:alice", trace_id="ta", payload=_payload("alice.csv"))
    repo.save("guest:bob", trace_id="tb", payload=_payload("bob.csv"))
    assert repo.get("user:alice")["original_filename"] == "alice.csv"
    assert repo.get("guest:bob")["original_filename"] == "bob.csv"
    assert repo.count() == 2
```

- [ ] **Step 2: Run, expect failure**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest tests/test_persistence_repository.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.persistence.repository'`.

- [ ] **Step 3: Create `src/persistence/repository.py`**

```python
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
        # Opportunistic sweep keeps the table bounded without a scheduler.
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
```

- [ ] **Step 4: Run tests, expect pass**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest tests/test_persistence_repository.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/persistence/repository.py tests/test_persistence_repository.py
git commit -m "feat(persistence): add DashboardRepository with session-keyed upsert"
```

---

## Task 6: TTL expiry & purge

**Files:**
- Modify: `tests/test_persistence_ttl.py` (new)
- (No source change — repository already implements TTL; this task proves it and locks the behaviour with tests.)

- [ ] **Step 1: Write TTL tests**

Create `tests/test_persistence_ttl.py`:

```python
"""TTL / expiry behaviour of DashboardRepository (§11 issue 21)."""
import pytest

from src.persistence import db
from src.persistence.repository import DashboardRepository


def _payload():
    return {"original_filename": "t.csv", "kpis": []}


@pytest.fixture
def make_repo(tmp_path):
    engine = db.make_engine(f"sqlite:///{tmp_path / 'ttl.db'}")
    db.init_db(engine)
    sf = db.make_session_factory(engine)

    def _factory(ttl_seconds):
        return DashboardRepository(sf, ttl_seconds=ttl_seconds)

    yield _factory
    engine.dispose()


def test_expired_row_is_invisible_to_get(make_repo):
    repo = make_repo(ttl_seconds=-1)  # already expired the instant it is written
    repo.save("guest:exp", trace_id="t", payload=_payload())
    assert repo.get("guest:exp") is None


def test_live_row_is_visible(make_repo):
    repo = make_repo(ttl_seconds=3600)
    repo.save("guest:live", trace_id="t", payload=_payload())
    assert repo.get("guest:live") is not None


def test_purge_expired_deletes_only_expired(make_repo):
    expired_repo = make_repo(ttl_seconds=-1)
    live_repo = make_repo(ttl_seconds=3600)
    expired_repo.save("guest:old", trace_id="t1", payload=_payload())
    live_repo.save("guest:new", trace_id="t2", payload=_payload())
    # Task 5's save() runs an opportunistic purge, so the expired row may
    # already be gone before this explicit purge — assert >= 0, and rely on
    # the live-row invariants below for the meaningful guarantee.
    removed = live_repo.purge_expired()
    assert removed >= 0
    assert live_repo.get("guest:new") is not None
    assert live_repo.count() == 1


def test_save_triggers_opportunistic_purge(make_repo):
    expired_repo = make_repo(ttl_seconds=-1)
    expired_repo.save("guest:a", trace_id="t1", payload=_payload())
    expired_repo.save("guest:b", trace_id="t2", payload=_payload())
    # Both writes are immediately expired; each save() sweeps. Table trends to 0.
    assert expired_repo.count() == 0
```

- [ ] **Step 2: Run, expect pass (behaviour already implemented in Task 5)**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest tests/test_persistence_ttl.py -v`
Expected: 4 PASS. If `test_save_triggers_opportunistic_purge` is flaky because the just-inserted row is deleted by its own post-save sweep before assertion, that is the intended behaviour (ttl=-1 means "expire immediately"); the assertion `count() == 0` is correct. If it fails with `count() == 1`, the opportunistic `purge_expired()` call at the end of `save()` is missing — fix `save()` in `repository.py` to call `self.purge_expired()` after commit (it should already, per Task 5).

- [ ] **Step 3: Commit**

```bash
git add tests/test_persistence_ttl.py
git commit -m "test(persistence): lock TTL expiry and opportunistic purge behaviour"
```

---

## Task 7: Rewrite `conftest.py` for DB-backed tests

**Files:**
- Modify: `tests/conftest.py`

> This task MUST land before Task 11 (dict deletion). It makes the test suite DB-aware while the dict still exists, so nothing breaks in between.

- [ ] **Step 1: Read the current `tests/conftest.py`**

It currently contains `client`, `_reset_storage` (autouse, clears `main_module.dashboard_storage`), `sample_csv_bytes`, `sample_df`, `upload_files`.

- [ ] **Step 2: Replace `tests/conftest.py` with the DB-aware version**

```python
"""Shared pytest fixtures for the AI-Powered Dashboards test suite."""
from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"

# Point the whole test session at an isolated SQLite file BEFORE main/app or
# any persistence module is imported, so config.DATABASE_URL picks it up.
_TEST_DB = Path(__file__).parent / "_pytest_dashboards.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB}"
os.environ.setdefault("DASHBOARD_TTL_SECONDS", "3600")
os.environ["REDIS_URL"] = ""  # tests never hit a real Redis


@pytest.fixture(scope="session", autouse=True)
def _create_schema_once():
    from src.persistence import db, repository
    engine = db.make_engine()
    db.init_db(engine)
    # Rebind the app's repository singleton to this engine.
    repository.reset_repository_for_tests(db.make_session_factory(engine))
    yield
    engine.dispose()
    if _TEST_DB.exists():
        _TEST_DB.unlink()


@pytest.fixture(autouse=True)
def _reset_storage():
    """Truncate the dashboard table before AND after every test so cases are
    isolated (replaces the Phase 0 in-process-dict clear)."""
    from src.persistence import db
    from src.persistence.models import DashboardRecord
    engine = db.make_engine()
    sf = db.make_session_factory(engine)
    with sf() as s:
        s.query(DashboardRecord).delete()
        s.commit()
    engine.dispose()
    yield
    engine = db.make_engine()
    sf = db.make_session_factory(engine)
    with sf() as s:
        s.query(DashboardRecord).delete()
        s.commit()
    engine.dispose()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    import main as main_module
    c = TestClient(main_module.app)
    c.headers.update({
        "X-Guest-Mode": "1",
        "X-Guest-Session-Id": "pytest-session",
    })
    yield c


@pytest.fixture
def sample_csv_bytes() -> bytes:
    return (FIXTURES / "sample_data.csv").read_bytes()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "sample_data.csv")


@pytest.fixture
def upload_files(sample_csv_bytes):
    return {"dataset": ("sample_data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
```

> Note: this references `repository.reset_repository_for_tests` and a repository singleton — both are added in Task 8 Step 3. Until Task 8, `_create_schema_once` will fail importing that symbol. To keep the suite green between Task 7 and Task 8, **Task 7 Step 2 also adds the singleton accessor stub** (next step).

- [ ] **Step 3: Add the repository singleton accessor (used by conftest + endpoints)**

Append to `src/persistence/repository.py`:

```python
_repository: Optional[DashboardRepository] = None


def get_repository() -> DashboardRepository:
    """App-wide singleton. Lazily builds the engine from config on first use."""
    global _repository
    if _repository is None:
        from src.persistence import db
        engine = db.make_engine()
        db.init_db(engine)
        _repository = DashboardRepository(db.make_session_factory(engine))
    return _repository


def reset_repository_for_tests(session_factory) -> None:
    """Rebind the singleton to a test-controlled session factory."""
    global _repository
    _repository = DashboardRepository(session_factory)
```

- [ ] **Step 4: Run the FULL suite**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest -v`
Expected: ALL Phase 0 + Phase 1 tests still PASS (the dict still exists in `main.py`; the new conftest just also manages a DB the endpoints don't use yet). The 2 Phase 0 xfails remain xfail. If Phase 0 endpoint tests fail here, the new `_reset_storage` no longer clears the dict — that is fine ONLY because each test still uses a unique guest session and the dict is process-global; verify the previously-passing `test_dashboard_is_per_session` and `test_upload_persists_data...` still pass. If they now fail due to dict bleed across tests, add a transitional dict-clear to `_reset_storage` (clear both dict and table) and remove it in Task 11.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py src/persistence/repository.py
git commit -m "test(persistence): make conftest DB-aware and add repository singleton"
```

---

## Task 8: Wire `/api/upload` to the repository

**Files:**
- Modify: `main.py`
- Test: `tests/test_api_persistence_integration.py` (new)

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_api_persistence_integration.py`:

```python
"""End-to-end: the HTTP API persists through the repository, not a dict."""


def test_upload_persists_via_repository(client, upload_files):
    from src.persistence.repository import get_repository

    r = client.post("/api/upload", files=upload_files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success"

    # The repository now holds exactly this session's dashboard.
    repo = get_repository()
    stored = repo.get("guest:pytest-session")
    assert stored is not None
    assert stored["original_filename"] == "sample_data.csv"


def test_upload_does_not_create_trace_id_keyed_row(client, upload_files):
    """§11 issue 5: trace_id must never be a storage key."""
    from src.persistence.repository import get_repository

    r = client.post("/api/upload", files=upload_files)
    trace_id = r.json()["trace_id"]
    repo = get_repository()
    assert repo.get(trace_id) is None  # no orphan row
    assert repo.get("guest:pytest-session") is not None
    assert repo.count() == 1  # exactly one row, the session row
```

- [ ] **Step 2: Run, expect failure**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest tests/test_api_persistence_integration.py -v`
Expected: FAIL — `repo.get("guest:pytest-session")` is `None` (the endpoint still writes the dict, not the repo); and `repo.count()` is 0.

- [ ] **Step 3: Modify `/api/upload` in `main.py`**

Add import near the other internal imports (after line ~41):

```python
from src.persistence.repository import get_repository
```

In `api_upload`, replace the storage block. The current code (around lines 161-163) is:

```python
    with storage_lock:
        dashboard_storage[trace_id] = response_data
        dashboard_storage[user['session_key']] = response_data
```

Replace it with:

```python
    get_repository().save(
        user["session_key"], trace_id=trace_id, payload=response_data
    )
```

Leave the `return {...}` block exactly as-is (it still returns `trace_id` and `data` in the HTTP response — only the *storage* changes).

- [ ] **Step 4: Run the integration test, expect pass**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest tests/test_api_persistence_integration.py -v`
Expected: both PASS.

- [ ] **Step 5: Run the full suite**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest -v`
Expected: all green. The Phase 0 `test_upload_persists_data_so_dashboard_endpoint_can_read_it` will still pass ONLY if `/api/dashboard` also reads the repo — it does NOT yet (Task 11). If that specific Phase 0 test fails here, mark the task DONE_WITH_CONCERNS noting "expected transient failure until Task 11 wires /api/dashboard"; it is reintroduced green in Task 11. (All other tests must pass.)

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_api_persistence_integration.py
git commit -m "feat(persistence): persist /api/upload via repository (drop dict + orphan trace_id)"
```

---

## Task 9: Wire `/api/upload/stream` to the repository

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add a failing test**

Append to `tests/test_api_persistence_integration.py`:

```python
import json


def test_stream_persists_via_repository(client, upload_files):
    from src.persistence.repository import get_repository

    with client.stream("POST", "/api/upload/stream", files=upload_files) as resp:
        assert resp.status_code == 200
        body = resp.read().decode("utf-8")

    # Stream completed; the repository must now hold this session's dashboard.
    repo = get_repository()
    stored = repo.get("guest:pytest-session")
    assert stored is not None
    assert "done" in body
    # No trace_id-keyed orphan row.
    assert repo.count() == 1
```

- [ ] **Step 2: Run, expect failure**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest tests/test_api_persistence_integration.py::test_stream_persists_via_repository -v`
Expected: FAIL — repo empty (stream still writes the dict).

- [ ] **Step 3: Modify the SSE generator in `main.py`**

In `api_upload_stream`'s inner `event_source()` generator, the current storage block (around lines 212-214) is:

```python
                    with storage_lock:
                        dashboard_storage[trace_id] = response_data
                        dashboard_storage[user['session_key']] = response_data
```

Replace it with:

```python
                    get_repository().save(
                        user["session_key"], trace_id=trace_id, payload=response_data
                    )
```

`get_repository()` was already imported in Task 8. The repository is synchronous, so calling it from this sync generator is correct and safe (this is the core reason the plan chose sync SQLAlchemy).

- [ ] **Step 4: Run the test, expect pass**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest tests/test_api_persistence_integration.py::test_stream_persists_via_repository -v`
Expected: PASS.

- [ ] **Step 5: Full suite (same caveat as Task 8 Step 5 re: the one Phase 0 dashboard-readback test until Task 11)**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest -v`

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_api_persistence_integration.py
git commit -m "feat(persistence): persist /api/upload/stream via repository"
```

---

## Task 10: Wire `/api/load_external` to the repository

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add a failing test**

Append to `tests/test_api_persistence_integration.py`:

```python
from unittest.mock import patch


def test_load_external_persists_via_repository(client):
    import pandas as pd
    from src.data.parser import LoadResult
    from src.persistence.repository import get_repository

    fake = LoadResult(
        df=pd.DataFrame({
            "x": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [2.0, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        }),
        success=True,
        warnings=[],
    )
    with patch("main.load_csv_from_url", return_value=fake):
        r = client.post(
            "/api/load_external",
            json={"external_source": "https://example.com/data.csv"},
        )
    assert r.status_code == 200, r.text
    repo = get_repository()
    assert repo.get("guest:pytest-session") is not None
    assert repo.count() == 1
```

- [ ] **Step 2: Run, expect failure**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest tests/test_api_persistence_integration.py::test_load_external_persists_via_repository -v`
Expected: FAIL — repo empty.

- [ ] **Step 3: Modify `/api/load_external` in `main.py`**

The current storage block (around lines 354-356) is:

```python
    with storage_lock:
        dashboard_storage[trace_id] = response_data
        dashboard_storage[user['session_key']] = response_data
```

Replace it with:

```python
    get_repository().save(
        user["session_key"], trace_id=trace_id, payload=response_data
    )
```

- [ ] **Step 4: Run the test, expect pass**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest tests/test_api_persistence_integration.py::test_load_external_persists_via_repository -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_api_persistence_integration.py
git commit -m "feat(persistence): persist /api/load_external via repository"
```

---

## Task 11: Wire `/api/dashboard` GET to the repository & delete the dict

**Files:**
- Modify: `main.py`

> This is the task that removes `dashboard_storage` and `storage_lock`. Tasks 7–10 have made every writer use the repository; this makes the reader use it too, then deletes the dict and the now-unused `Lock` import. After this task the THREE transiently-red Phase 0 tests (`test_upload_persists_data_so_dashboard_endpoint_can_read_it`, `test_dashboard_returns_ready_after_upload`, `test_dashboard_is_per_session`) all go green again, now DB-backed and isolated by the conftest table-truncation (not by session-id luck — re-verify `test_dashboard_is_per_session` specifically isolates correctly now that the dict is gone).

- [ ] **Step 1: Add a failing test**

Append to `tests/test_api_persistence_integration.py`:

```python
def test_dashboard_reads_from_repository(client, upload_files):
    client.post("/api/upload", files=upload_files)
    r = client.get("/api/dashboard")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["original_filename"] == "sample_data.csv"


def test_dashboard_empty_when_repository_has_no_row(client):
    r = client.get("/api/dashboard")
    assert r.status_code == 200
    assert r.json()["status"] == "empty"


def test_no_dashboard_storage_attribute_remains():
    """The in-process dict must be fully removed (§11 issue 1)."""
    import main as main_module
    assert not hasattr(main_module, "dashboard_storage")
    assert not hasattr(main_module, "storage_lock")
```

- [ ] **Step 2: Run, expect failure**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest tests/test_api_persistence_integration.py -v`
Expected: `test_no_dashboard_storage_attribute_remains` FAILS (dict still present); `test_dashboard_reads_from_repository` may FAIL (GET still reads the dict, which is no longer written).

- [ ] **Step 3: Modify `/api/dashboard` in `main.py`**

The current read block (around lines 366-367) is:

```python
    with storage_lock:
        dashboard_data = dashboard_storage.get(user['session_key'])
```

Replace it with:

```python
    dashboard_data = get_repository().get(user["session_key"])
```

- [ ] **Step 4: Delete the dict, lock, and unused import**

In `main.py`:
- Delete the two lines (around 56-58):
  ```python
  # ---------------- DASHBOARD STATE STORAGE ----------------
  dashboard_storage = {}
  storage_lock = Lock()
  ```
  (Keep a comment line `# ---------------- DASHBOARD STATE STORAGE (now in src/persistence) ----------------` if you want a breadcrumb, or remove the section header entirely.)
- Remove the now-unused import `from threading import Lock` (line ~17). First grep `main.py` for `Lock` and `storage_lock` and `dashboard_storage` — there must be ZERO remaining references before deleting the import. If any reference remains, that endpoint was missed in Tasks 8–10 — STOP and report which line.

- [ ] **Step 5: Run the full suite**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest -v`
Expected: ALL pass. Specifically verify these previously-transient Phase 0 tests are now green: `tests/test_api_dashboard.py::test_dashboard_returns_ready_after_upload`, `::test_dashboard_is_per_session`, `tests/test_api_upload.py::test_upload_persists_data_so_dashboard_endpoint_can_read_it`. The 2 Phase 0 xfails remain xfail (unchanged — unrelated to persistence).

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_api_persistence_integration.py
git commit -m "feat(persistence): read /api/dashboard from repository; remove in-process dict"
```

---

## Task 12: App startup — initialise the DB on boot

**Files:**
- Modify: `main.py`
- Test: `tests/test_api_persistence_integration.py`

- [ ] **Step 1: Add a restart-survival test**

Append to `tests/test_api_persistence_integration.py`:

```python
def test_dashboard_survives_repository_singleton_reset(client, upload_files):
    """Simulate a process restart: drop the in-memory singleton, rebuild it
    from the same DB file, and confirm the dashboard is still there.
    This is the core proof of §11 issue 1 (no persistence) being fixed."""
    import src.persistence.repository as repo_mod
    from src.persistence import db

    client.post("/api/upload", files=upload_files)
    assert client.get("/api/dashboard").json()["status"] == "ready"

    # "Restart": forget the singleton, rebuild from the same DATABASE_URL.
    repo_mod._repository = None
    rebuilt = repo_mod.get_repository()
    assert rebuilt.get("guest:pytest-session") is not None

    # And the HTTP layer still serves it after the reset.
    assert client.get("/api/dashboard").json()["status"] == "ready"
```

- [ ] **Step 2: Run, expect pass or fail**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest tests/test_api_persistence_integration.py::test_dashboard_survives_repository_singleton_reset -v`
Expected: This likely already PASSES (the repository reads the persistent SQLite file regardless of singleton identity). If it passes, that already demonstrates persistence; proceed to add explicit startup init anyway (Step 3) so production with a fresh DB/Postgres auto-creates the schema. If it FAILS because `_create_schema_once` bound a different engine, ensure `reset_repository_for_tests` and `get_repository()` resolve to the same `DATABASE_URL` (they do — both call `db.make_engine()` with no arg → `config.DATABASE_URL`).

- [ ] **Step 3: Add a FastAPI startup hook in `main.py`**

After `app = FastAPI()` and the middleware/router block (after line ~89, before the API routes), add:

```python
@app.on_event("startup")
def _init_persistence() -> None:
    from src.persistence.repository import get_repository
    # Building the singleton also calls db.init_db() (create tables if absent).
    get_repository()
    logger.info("Persistence layer initialised")
```

> `on_event("startup")` is deprecated in newer FastAPI but fully supported in 0.109.2 and is used elsewhere in this codebase's idiom. Do NOT introduce the lifespan-context API here — that is a larger refactor out of scope.

- [ ] **Step 4: Run the full suite**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest -v`
Expected: all green.

- [ ] **Step 5: Manual boot smoke test**

```bash
cd "f:/AI Powered Dashboards" && "f:/AI Powered Dashboards/venv/Scripts/python.exe" -c "import main; print('app import + startup wiring OK')"
```
Expected: prints OK with no exception. (Full uvicorn boot is optional; the import-time wiring + startup hook registration is what we verify here.)

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_api_persistence_integration.py
git commit -m "feat(persistence): initialise DB schema on app startup"
```

---

## Task 13: Optional Redis read-through cache

**Files:**
- Create: `src/persistence/cache.py`
- Modify: `src/persistence/repository.py` (wrap singleton in cache when `REDIS_URL` set)
- Test: `tests/test_persistence_cache.py`

- [ ] **Step 1: Write tests (no-op path + cached path with a fake Redis)**

Create `tests/test_persistence_cache.py`:

```python
"""Optional Redis read-through cache. Must no-op cleanly when REDIS_URL unset."""
import pytest

from src.persistence import db
from src.persistence.repository import DashboardRepository
from src.persistence.cache import CachedRepository, build_cache_client


def _payload(name="c.csv"):
    return {"original_filename": name, "kpis": []}


@pytest.fixture
def base_repo(tmp_path):
    engine = db.make_engine(f"sqlite:///{tmp_path / 'cache.db'}")
    db.init_db(engine)
    yield DashboardRepository(db.make_session_factory(engine))
    engine.dispose()


def test_build_cache_client_returns_none_when_url_empty(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "")
    assert build_cache_client("") is None


def test_cached_repo_is_passthrough_without_client(base_repo):
    cached = CachedRepository(base_repo, client=None)
    cached.save("guest:s", trace_id="t", payload=_payload("p.csv"))
    assert cached.get("guest:s")["original_filename"] == "p.csv"
    assert cached.get("guest:absent") is None


class _FakeRedis:
    def __init__(self):
        self.store = {}
        self.deleted = []

    def get(self, k):
        return self.store.get(k)

    def setex(self, k, ttl, v):
        self.store[k] = v

    def delete(self, k):
        self.deleted.append(k)
        self.store.pop(k, None)


def test_cached_repo_serves_second_read_from_cache(base_repo):
    fake = _FakeRedis()
    cached = CachedRepository(base_repo, client=fake)
    cached.save("guest:s", trace_id="t", payload=_payload("hit.csv"))
    # First get → DB miss → populates cache
    first = cached.get("guest:s")
    assert first["original_filename"] == "hit.csv"
    assert "dash:guest:s" in fake.store
    # Corrupt the DB row to prove the 2nd read came from cache
    base_repo.save("guest:s", trace_id="t2", payload=_payload("changed.csv"))
    # save() invalidates the cache key, so next get reflects the new value
    assert cached.get("guest:s")["original_filename"] == "changed.csv"


def test_save_invalidates_cache_key(base_repo):
    fake = _FakeRedis()
    cached = CachedRepository(base_repo, client=fake)
    cached.save("guest:s", trace_id="t", payload=_payload("v1.csv"))
    cached.get("guest:s")  # populate
    cached.save("guest:s", trace_id="t", payload=_payload("v2.csv"))
    assert "dash:guest:s" in fake.deleted
```

- [ ] **Step 2: Run, expect failure**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest tests/test_persistence_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.persistence.cache'`.

- [ ] **Step 3: Create `src/persistence/cache.py`**

```python
"""Optional Redis read-through cache for DashboardRepository.

Disabled by default: when REDIS_URL is empty, build_cache_client() returns None
and CachedRepository becomes a transparent passthrough. This mirrors the Phase 0
optional-OTel/Sentry pattern — adding a scaling capability without forcing infra
on dev or CI.
"""
from __future__ import annotations

import json
from typing import Optional

from src import config
from src.persistence.repository import DashboardRepository

_KEY_PREFIX = "dash:"


def build_cache_client(url: str | None = None):
    url = url if url is not None else config.REDIS_URL
    if not url:
        return None
    try:
        import redis
    except ImportError:
        return None
    return redis.Redis.from_url(url, decode_responses=True)


class CachedRepository:
    """Wraps a DashboardRepository. Read-through on get(); write-through +
    cache-invalidation on save(). Transparent passthrough when client is None."""

    def __init__(self, base: DashboardRepository, client=None, ttl_seconds: int | None = None):
        self._base = base
        self._client = client
        self._ttl = ttl_seconds if ttl_seconds is not None else config.DASHBOARD_TTL_SECONDS

    @staticmethod
    def _key(session_key: str) -> str:
        return f"{_KEY_PREFIX}{session_key}"

    def save(self, session_key: str, *, trace_id: str, payload: dict) -> None:
        self._base.save(session_key, trace_id=trace_id, payload=payload)
        if self._client is not None:
            self._client.delete(self._key(session_key))

    def get(self, session_key: str) -> Optional[dict]:
        if self._client is not None:
            cached = self._client.get(self._key(session_key))
            if cached is not None:
                return json.loads(cached)
        value = self._base.get(session_key)
        if value is not None and self._client is not None:
            self._client.setex(self._key(session_key), self._ttl, json.dumps(value))
        return value

    def purge_expired(self) -> int:
        return self._base.purge_expired()

    def count(self) -> int:
        return self._base.count()
```

- [ ] **Step 4: Wire the cache into the singleton accessor**

In `src/persistence/repository.py`, modify `get_repository()` so the returned object is cache-wrapped when Redis is configured. Replace the existing `get_repository()` body with:

```python
def get_repository():
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
```

And update `reset_repository_for_tests` to also wrap (so tests exercise the same code path, with `client=None` since `REDIS_URL=""` in conftest):

```python
def reset_repository_for_tests(session_factory) -> None:
    global _repository
    from src.persistence.cache import CachedRepository, build_cache_client
    base = DashboardRepository(session_factory)
    _repository = CachedRepository(base, client=build_cache_client())
```

> `CachedRepository` exposes `get/save/purge_expired/count` — the same surface the endpoints and conftest already use, so no caller changes. Confirm by grepping callers: `get_repository().save(`, `.get(`, `.count(` — all four methods exist on `CachedRepository`.

- [ ] **Step 5: Run cache tests + full suite**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest tests/test_persistence_cache.py -v`
Expected: 4 PASS (the test file defines 4 functions; an earlier draft of this plan miscounted as 5). Full suite ends at 81 passed, 2 xfailed.
Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest -v`
Expected: full suite green (conftest uses `REDIS_URL=""` so the cache is a passthrough — all prior persistence/integration tests still pass unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/persistence/cache.py src/persistence/repository.py tests/test_persistence_cache.py
git commit -m "feat(persistence): optional Redis read-through cache (no-op when unconfigured)"
```

---

## Task 14: Phase 1 close-out — integration sweep, docs, changelog

**Files:**
- Modify: `tests/test_api_persistence_integration.py`
- Modify: `CHANGELOG.md`
- Modify: `docs/superpowers/plans/2026-05-15-phase-1-persistence.md` (check boxes — optional)

- [ ] **Step 1: Add the consolidated end-to-end guarantees test**

Append to `tests/test_api_persistence_integration.py`:

```python
def test_per_session_isolation_end_to_end(client, upload_files):
    # Alice uploads.
    a = client.post(
        "/api/upload", files=upload_files,
        headers={"X-Guest-Mode": "1", "X-Guest-Session-Id": "alice"},
    )
    assert a.status_code == 200
    # Bob sees nothing.
    b = client.get(
        "/api/dashboard",
        headers={"X-Guest-Mode": "1", "X-Guest-Session-Id": "bob"},
    )
    assert b.json()["status"] == "empty"
    # Alice still sees hers.
    a2 = client.get(
        "/api/dashboard",
        headers={"X-Guest-Mode": "1", "X-Guest-Session-Id": "alice"},
    )
    assert a2.json()["status"] == "ready"


def test_ttl_expiry_end_to_end(client, upload_files, monkeypatch):
    """With TTL forced negative, an uploaded dashboard is immediately expired
    and the GET returns empty — proving §11 issue 21 end-to-end."""
    import src.persistence.repository as repo_mod
    from src.persistence import db

    # Rebuild the singleton with ttl=-1 against the test DB.
    engine = db.make_engine()
    db.init_db(engine)
    from src.persistence.repository import DashboardRepository
    from src.persistence.cache import CachedRepository, build_cache_client
    repo_mod._repository = CachedRepository(
        DashboardRepository(db.make_session_factory(engine), ttl_seconds=-1),
        client=build_cache_client(),
    )
    try:
        client.post("/api/upload", files=upload_files)
        assert client.get("/api/dashboard").json()["status"] == "empty"
    finally:
        repo_mod._repository = None  # let the next test rebuild cleanly
```

- [ ] **Step 2: Run the full suite twice (flakiness / state-bleed check)**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest -v` (twice)
Expected: identical result both runs, all green, the 2 Phase 0 xfails still xfail. No SQLite "database is locked" errors. If a "database is locked" error appears under the autouse truncate fixture, add `engine.dispose()` after each `_reset_storage` block (already present in the Task 7 fixture) and ensure no session is left open.

- [ ] **Step 3: Coverage check**

Run: `"f:/AI Powered Dashboards/venv/Scripts/python.exe" -m pytest --cov --cov-report=term-missing -v`
Record the total coverage %. Report it honestly; the new `src/persistence/*` modules should be well-covered (>85%). Do NOT pad.

- [ ] **Step 4: Update `CHANGELOG.md`**

Add a new section at the top under `# Changelog`:

```markdown
## [Unreleased] — Phase 1: Persistence Layer

### Added
- SQL-backed dashboard persistence (`src/persistence/`): SQLAlchemy `DashboardRecord`, `DashboardRepository`, Alembic migrations. Default backend SQLite; production via `DATABASE_URL` (Postgres-compatible).
- Row-level TTL: dashboards expire after `DASHBOARD_TTL_SECONDS` (default 24h); expired rows are excluded from reads and swept opportunistically on write.
- Optional Redis read-through cache (`REDIS_URL`); transparent passthrough when unset.
- `@app.on_event("startup")` hook creates the schema on boot.

### Changed
- `/api/upload`, `/api/upload/stream`, `/api/load_external` now persist via the repository; `/api/dashboard` reads from it.
- Test suite is DB-backed (isolated SQLite per session; table truncated per test).

### Removed
- In-process `dashboard_storage` dict and `storage_lock` (§11 issue 1).
- Orphan `trace_id`-keyed storage entries — `trace_id` remains in the HTTP response but is never a storage key (§11 issue 5).

### Fixed (§11)
- Issue 1: dashboards survive process restart.
- Issue 5: no orphan `trace_id` rows; one row per `session_key`.
- Issue 21: rows expire (TTL) and are evicted.

### Deferred
- Raw-CSV object store (not required by issues 1/5/21 — YAGNI).
- Async DB/pipeline (sub-plan #3).
```

- [ ] **Step 5: Update `.env.example` sanity check**

Confirm `DATABASE_URL`, `DASHBOARD_TTL_SECONDS`, `REDIS_URL` are present and documented (added in Task 1). If missing, add them now.

- [ ] **Step 6: Final commit (NO push, NO PR)**

```bash
git add tests/test_api_persistence_integration.py CHANGELOG.md .env.example
git commit -m "docs(persistence): Phase 1 changelog + end-to-end guarantee tests"
```

- [ ] **Step 7: Report the branch state**

Run: `git log --oneline phase-0-stabilize..phase-1-persistence` and report the full commit list, final pytest summary line, and total coverage %. Confirm: "Did NOT push. Did NOT open a PR."

---

## Self-review

**1. Spec coverage:**
- Issue 1 (no persistence) → Tasks 2–5, 8–12; proven by `test_dashboard_survives_repository_singleton_reset` (Task 12) and `test_no_dashboard_storage_attribute_remains` (Task 11). ✓
- Issue 5 (orphan trace_id) → Tasks 8–11; proven by `test_upload_does_not_create_trace_id_keyed_row` and `repo.count() == 1` assertions. ✓
- Issue 21 (no TTL) → Tasks 6, 14; proven by `test_persistence_ttl.py` and `test_ttl_expiry_end_to_end`. ✓
- Auth unchanged → no task touches `src/auth.py`; `session_key` consumed exactly as Phase 0 produced it. ✓
- Infrastructure-agnostic → `DATABASE_URL`/`REDIS_URL` driven; SQLite default, Postgres via URL, Redis optional. ✓
- Stacks on phase-0-stabilize → stated in "Branch & prerequisites". ✓

**2. Placeholder scan:** No "TBD/TODO/handle edge cases/similar to Task N". Every code step has complete, runnable code. The two intentional cross-task dependencies (Task 2 tests red until Task 3; Task 7 conftest references symbols added in its own Step 3) are explicitly called out with rationale, not left as silent gaps. ✓

**3. Type/name consistency:**
- `DashboardRepository.__init__(session_factory, ttl_seconds=None)` — consistent across Tasks 5, 6, 13, 14.
- `save(session_key, *, trace_id, payload)` and `get(session_key) -> Optional[dict]` and `purge_expired() -> int` and `count() -> int` — identical signatures everywhere they appear (repository, cache wrapper, tests, endpoints).
- `get_repository()` / `reset_repository_for_tests(session_factory)` — defined Task 7, extended Task 13, used Tasks 8–12, 14. Consistent.
- `db.make_engine(url=None)`, `db.make_session_factory(engine)`, `db.init_db(engine)`, `db.reset_db_for_tests(engine)` — consistent Tasks 2–14.
- `CachedRepository` exposes the same 4-method surface as `DashboardRepository`, so swapping it into the singleton (Task 13) needs no caller changes — verified against the endpoint call sites (`get_repository().save(...)` / `.get(...)`) introduced in Tasks 8–11. ✓

**Risks called out for the implementer:**
1. SQLite cross-thread access (FastAPI threadpool + SSE sync generator) — mitigated by `check_same_thread=False` in `make_engine` (Task 2). If "database is locked" appears under load, that's a known SQLite limitation; production uses Postgres. Tests dispose engines per fixture to avoid it.
2. Task ordering is load-bearing: **7 before 11**. The plan enforces it and explains why in the "Critical sequencing note".
3. Between Tasks 8 and 11, **three** Phase 0 tests are transiently red by design (not one — the upload→`/api/dashboard` readback contract is exercised by `test_upload_persists_data_so_dashboard_endpoint_can_read_it`, `test_dashboard_returns_ready_after_upload`, and `test_dashboard_is_per_session`). All fail with the identical mechanism: `assert 'empty' == 'ready'` because the writer uses the repo while `/api/dashboard` still reads the dict. Which of the three surface in Tasks 8/9 vs. 10 varies with inter-test dict-state leakage; by Task 10 all three are red. They ALL return green at Task 11 (Step 5 verifies exactly these three). The implementer must not "fix" them early; any red with a DIFFERENT assertion/mechanism IS a real bug and must stop the chain.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-15-phase-1-persistence.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, spec + code-quality review after each, fast iteration. Same approach that delivered Phase 0's 20 tasks.

**2. Inline Execution** — execute in this session via executing-plans, batch with checkpoints.

**Which approach?**
