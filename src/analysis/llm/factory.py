"""Config-driven provider selection.

`config.LLM_PROVIDER` picks the concrete implementation. An unknown/disabled
provider yields a :class:`NullProvider` whose ``available()`` is False, so the
pipeline degrades to the heuristic layer instead of raising — the same
fail-safe contract the rest of the AI layer guarantees.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src import config
from src.analysis.llm.base import LLMProvider, LLMUnavailable


class NullProvider(LLMProvider):
    """No-op provider: never available, never calls out."""

    name = "null"

    def available(self) -> bool:
        return False

    def complete_json(
        self, *, system: str, user: str, temperature: float = 0.2
    ) -> Dict[str, Any]:
        raise LLMUnavailable("No LLM provider configured")


_singleton: Optional[LLMProvider] = None


def _build() -> LLMProvider:
    provider = (config.LLM_PROVIDER or "").strip().lower()
    if provider == "groq":
        from src.analysis.llm.groq_provider import GroqProvider

        return GroqProvider()
    return NullProvider()


def get_llm_provider() -> LLMProvider:
    global _singleton
    if _singleton is None:
        _singleton = _build()
    return _singleton


def reset_llm_provider_for_tests() -> None:
    global _singleton
    _singleton = None
