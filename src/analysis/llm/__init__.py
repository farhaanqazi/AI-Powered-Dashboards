"""Phase 9 (S9.1) — provider-agnostic AI.

`src.analysis.llm_analyst` depends ONLY on the abstractions exported here, never
on a concrete SDK. The concrete provider + model are chosen from `config.py`.

Public surface:
    LLMProvider          — the interface analysis code programs against.
    LLMUnavailable       — raised when no usable provider is configured.
    get_llm_provider()   — config-driven singleton factory.
"""
from src.analysis.llm.base import LLMProvider, LLMUnavailable
from src.analysis.llm.factory import get_llm_provider, reset_llm_provider_for_tests

__all__ = [
    "LLMProvider",
    "LLMUnavailable",
    "get_llm_provider",
    "reset_llm_provider_for_tests",
]
