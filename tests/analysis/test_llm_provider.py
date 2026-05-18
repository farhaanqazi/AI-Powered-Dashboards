"""Phase 9 (S9.1) — provider abstraction + ground-truth response cache.

Asserts: the interface contract, config-driven selection, fail-safe
degradation (no key ⇒ NullProvider ⇒ heuristic fallback, never an exception),
and that identical ground truth is served from cache without a second
(paid) call.
"""
from __future__ import annotations

import sys
import types

import pytest

from src import config
from src.analysis.llm import LLMUnavailable, get_llm_provider
from src.analysis.llm.cache import (
    LLMResponseCache,
    ground_truth_key,
    reset_response_cache_for_tests,
)
from src.analysis.llm.factory import (
    NullProvider,
    reset_llm_provider_for_tests,
)
from src.analysis.llm.groq_provider import GroqProvider


@pytest.fixture(autouse=True)
def _reset_singletons():
    reset_llm_provider_for_tests()
    reset_response_cache_for_tests()
    yield
    reset_llm_provider_for_tests()
    reset_response_cache_for_tests()


# --- factory / interface --------------------------------------------------

def test_factory_returns_groq_for_groq_provider(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    reset_llm_provider_for_tests()
    p = get_llm_provider()
    assert isinstance(p, GroqProvider)
    assert get_llm_provider() is p  # singleton


def test_factory_falls_back_to_null_for_unknown_provider(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "totally-unknown")
    reset_llm_provider_for_tests()
    p = get_llm_provider()
    assert isinstance(p, NullProvider)
    assert p.available() is False
    with pytest.raises(LLMUnavailable):
        p.complete_json(system="s", user="u")


def test_groq_provider_unavailable_without_key(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    p = GroqProvider()
    assert p.available() is False
    with pytest.raises(LLMUnavailable):
        p.complete_json(system="s", user="u")


# --- ground-truth cache ---------------------------------------------------

def test_ground_truth_key_is_deterministic_and_input_sensitive():
    base = dict(provider="groq", model="m", temperature=0.2, system="S", user="U")
    assert ground_truth_key(**base) == ground_truth_key(**base)
    assert ground_truth_key(**{**base, "user": "U2"}) != ground_truth_key(**base)
    assert ground_truth_key(**{**base, "temperature": 0.0}) != ground_truth_key(**base)


def test_cache_roundtrip_ttl_and_lru(monkeypatch):
    monkeypatch.setattr(config, "LLM_RESPONSE_CACHE_ENABLED", True)
    c = LLMResponseCache(ttl_seconds=1000, max_entries=2)
    c.put("k1", {"a": 1})
    assert c.get("k1") == {"a": 1}
    # Returned value is a copy — mutating it must not poison the cache.
    got = c.get("k1")
    got["a"] = 999
    assert c.get("k1") == {"a": 1}
    # LRU eviction at capacity.
    c.put("k2", {"b": 2})
    c.get("k1")  # touch k1 so k2 is the least-recently-used
    c.put("k3", {"c": 3})
    assert c.get("k2") is None
    assert c.get("k1") == {"a": 1}


def test_cache_disabled_is_transparent(monkeypatch):
    monkeypatch.setattr(config, "LLM_RESPONSE_CACHE_ENABLED", False)
    c = LLMResponseCache()
    c.put("k", {"x": 1})
    assert c.get("k") is None


def test_cache_expiry(monkeypatch):
    monkeypatch.setattr(config, "LLM_RESPONSE_CACHE_ENABLED", True)
    c = LLMResponseCache(ttl_seconds=1000)
    c.put("k", {"x": 1})
    t = {"now": 0.0}
    monkeypatch.setattr("src.analysis.llm.cache.time.monotonic", lambda: t["now"])
    c.put("k", {"x": 1})
    assert c.get("k") == {"x": 1}
    t["now"] = 2000.0
    assert c.get("k") is None


# --- GroqProvider + cache integration (stubbed SDK) -----------------------

class _Counter:
    n = 0


def _install_fake_groq(monkeypatch, payload='{"ok": true}'):
    """Inject a fake `groq` module that counts API calls."""
    _Counter.n = 0

    class _Msg:
        def __init__(self, content):
            self.message = types.SimpleNamespace(content=content)

    class _Resp:
        def __init__(self, content):
            self.choices = [_Msg(content)]

    class _Completions:
        def create(self, **kw):
            _Counter.n += 1
            return _Resp(payload)

    class _Chat:
        completions = _Completions()

    class Groq:
        def __init__(self, **kw):
            self.chat = _Chat()

    fake = types.ModuleType("groq")
    fake.Groq = Groq
    monkeypatch.setitem(sys.modules, "groq", fake)


def test_groq_provider_caches_identical_ground_truth(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(config, "LLM_RESPONSE_CACHE_ENABLED", True)
    _install_fake_groq(monkeypatch, '{"narrative": "hello"}')
    reset_response_cache_for_tests()

    p = GroqProvider()
    r1 = p.complete_json(system="S", user="GROUND TRUTH A", temperature=0.2)
    r2 = p.complete_json(system="S", user="GROUND TRUTH A", temperature=0.2)
    assert r1 == r2 == {"narrative": "hello"}
    assert _Counter.n == 1  # second call served from cache, no paid call

    p.complete_json(system="S", user="DIFFERENT GROUND TRUTH", temperature=0.2)
    assert _Counter.n == 2  # different ground truth ⇒ real call


def test_groq_provider_wraps_sdk_failure_as_unavailable(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")

    class Groq:
        def __init__(self, **kw):
            raise RuntimeError("network down")

    fake = types.ModuleType("groq")
    fake.Groq = Groq
    monkeypatch.setitem(sys.modules, "groq", fake)

    with pytest.raises(LLMUnavailable):
        GroqProvider().complete_json(system="s", user="u")


def test_run_ai_analyst_falls_back_when_provider_unavailable(monkeypatch):
    """End-to-end: no provider ⇒ heuristic fallback, never an exception."""
    monkeypatch.setattr(config, "AI_ANALYST_ENABLED", True)
    monkeypatch.setattr(config, "LLM_PROVIDER", "null")
    reset_llm_provider_for_tests()
    from src.analysis.llm_analyst import run_ai_analyst

    out = run_ai_analyst(
        {"col": object()},  # truthy profiles; provider is what's unavailable
        [],
        {},
        fallback_kpis=[{"label": "K"}],
        fallback_specs=[{"intent": "category_count", "x_field": "col"}],
    )
    assert out["kpis"] == [{"label": "K"}]
    assert out["chart_specs"] == [{"intent": "category_count", "x_field": "col"}]
