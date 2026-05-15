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
