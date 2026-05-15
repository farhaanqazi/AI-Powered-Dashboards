"""Persistence config + engine tests."""


def test_config_exposes_persistence_knobs():
    from src import config
    assert isinstance(config.DATABASE_URL, str)
    assert config.DATABASE_URL  # non-empty default
    assert isinstance(config.DASHBOARD_TTL_SECONDS, int)
    assert config.DASHBOARD_TTL_SECONDS > 0
    # REDIS_URL may be empty string (= cache disabled), but must be a str
    assert isinstance(config.REDIS_URL, str)
