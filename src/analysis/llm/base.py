"""The LLM provider interface analysis code programs against.

Hard invariant (unchanged from before the abstraction): the model only selects
and labels — it never supplies a number. This interface deliberately exposes a
single capability: "given a system + user prompt, return parsed JSON". That is
all `llm_analyst` ever needed from Groq; nothing provider-specific leaks out.
"""
from __future__ import annotations

import abc
from typing import Any, Dict


class LLMUnavailable(RuntimeError):
    """No usable provider (missing key, SDK absent, or call failed).

    Callers treat this exactly like the old bare-`Exception` path: log and
    fall back to the deterministic heuristic Layer 4. The app never breaks
    because the AI layer is unavailable.
    """


class LLMProvider(abc.ABC):
    """A JSON-returning chat completion, abstracted away from any SDK."""

    #: Stable short name (``"groq"``, ``"null"`` …) — used in cache keys + logs.
    name: str = "base"

    @abc.abstractmethod
    def available(self) -> bool:
        """True iff a real call could succeed (key present, SDK importable)."""

    @abc.abstractmethod
    def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """Return the model's response parsed as a JSON object.

        Raises :class:`LLMUnavailable` on any failure (unconfigured, SDK
        missing, network/timeout, non-JSON output). Implementations must never
        leak a provider-specific exception type to the caller.
        """

    def model(self) -> str:
        """The concrete model id, for cache-key disambiguation + logs."""
        return "unknown"
