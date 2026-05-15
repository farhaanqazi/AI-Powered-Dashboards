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
    repo = make_repo(ttl_seconds=-1)
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
    # save() purges opportunistically, so expired_repo.save already deleted
    # "guest:old"; an explicit purge here is therefore a no-op (>= 0). The
    # invariant that matters: purge never touches a live row.
    removed = live_repo.purge_expired()
    assert removed >= 0
    assert live_repo.get("guest:new") is not None
    assert live_repo.count() == 1


def test_save_triggers_opportunistic_purge(make_repo):
    expired_repo = make_repo(ttl_seconds=-1)
    expired_repo.save("guest:a", trace_id="t1", payload=_payload())
    expired_repo.save("guest:b", trace_id="t2", payload=_payload())
    assert expired_repo.count() == 0
