"""Persistence config + engine tests."""


def test_config_exposes_persistence_knobs():
    from src import config
    assert isinstance(config.DATABASE_URL, str)
    assert config.DATABASE_URL  # non-empty default
    assert isinstance(config.DASHBOARD_TTL_SECONDS, int)
    assert config.DASHBOARD_TTL_SECONDS > 0
    # REDIS_URL may be empty string (= cache disabled), but must be a str
    assert isinstance(config.REDIS_URL, str)


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
