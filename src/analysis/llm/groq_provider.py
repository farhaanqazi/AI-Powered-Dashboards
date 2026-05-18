"""Groq-backed :class:`LLMProvider`.

The ONLY module that imports the ``groq`` SDK. Wraps a JSON chat completion
and the ground-truth response cache. Any SDK/network/parse failure surfaces as
:class:`LLMUnavailable` so callers fall back to the heuristic layer exactly as
before the abstraction existed.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from src import config
from src.analysis.llm.base import LLMProvider, LLMUnavailable
from src.analysis.llm.cache import get_response_cache, ground_truth_key

try:
    from src.logger import get_logger
    logger = get_logger(__name__)
except Exception:  # pragma: no cover - logging is best-effort
    import logging
    logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    name = "groq"

    def model(self) -> str:
        return config.GROQ_MODEL

    def available(self) -> bool:
        if not config.GROQ_API_KEY:
            return False
        try:
            import groq  # noqa: F401
        except Exception:
            return False
        return True

    def complete_json(
        self, *, system: str, user: str, temperature: float = 0.2
    ) -> Dict[str, Any]:
        if not self.available():
            raise LLMUnavailable("Groq provider unavailable (no key or SDK)")

        cache = get_response_cache()
        key = ground_truth_key(
            provider=self.name,
            model=self.model(),
            temperature=temperature,
            system=system,
            user=user,
        )
        cached = cache.get(key)
        if cached is not None:
            logger.info("LLM response cache hit (%s)", key[:12])
            return cached

        try:
            from groq import Groq

            client = Groq(
                api_key=config.GROQ_API_KEY,
                timeout=config.GROQ_TIMEOUT_SECONDS,
            )
            resp = client.chat.completions.create(
                model=config.GROQ_MODEL,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            parsed = json.loads(resp.choices[0].message.content)
        except Exception as exc:  # SDK missing, network, timeout, bad JSON
            raise LLMUnavailable(str(exc)) from exc

        if not isinstance(parsed, dict):
            raise LLMUnavailable("LLM returned non-object JSON")

        cache.put(key, parsed)
        return parsed
