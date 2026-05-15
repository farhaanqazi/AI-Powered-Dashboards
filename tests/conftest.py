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
