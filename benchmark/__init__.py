"""Isolated benchmark harness (Document A scaffold).

This package drives the existing engine (``src/``) as a *library*, read-only
with respect to engine source. It exists to stand up the divergence baseline;
the benchmark itself (authority knob, metrics, datasets, plots) is Document B.

ISOLATION RULE: ``benchmark/`` imports ``src`` — never the reverse. Nothing in
the app may import anything from ``benchmark/``.
"""
