"""Phase 11 — "Ask Your Data".

The conversational follow-up is governed by the same invariant as the rest of
the system: **the LLM never produces a number.** It only proposes which
deterministic tool to run (from a fixed, contract-guarded catalog) and then
narrates the numbers the backend computed. Every figure carries a provenance
token, and the agent is hard-bounded so it always terminates.
"""
from src.analysis.ask.agent import run_ask
from src.analysis.ask.interact import (
    apply_filters,
    run_interaction,
    run_interaction_cached,
)
from src.analysis.ask.tools import TOOLS, ToolError

__all__ = [
    "run_ask",
    "apply_filters",
    "run_interaction",
    "run_interaction_cached",
    "TOOLS",
    "ToolError",
]
