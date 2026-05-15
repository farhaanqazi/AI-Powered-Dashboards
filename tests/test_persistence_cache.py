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
    first = cached.get("guest:s")
    assert first["original_filename"] == "hit.csv"
    assert "dash:guest:s" in fake.store
    cached.save("guest:s", trace_id="t2", payload=_payload("changed.csv"))
    assert cached.get("guest:s")["original_filename"] == "changed.csv"


def test_save_invalidates_cache_key(base_repo):
    fake = _FakeRedis()
    cached = CachedRepository(base_repo, client=fake)
    cached.save("guest:s", trace_id="t", payload=_payload("v1.csv"))
    cached.get("guest:s")
    cached.save("guest:s", trace_id="t", payload=_payload("v2.csv"))
    assert "dash:guest:s" in fake.deleted
