"""Document B benchmark harness (additive, flag-gated, isolated).

Drives the FROZEN engine (`src/`) as a library. Nothing here is imported by the
app. Import order matters: call `env.setup_scratch_env()` BEFORE importing any
`src` module (config reads env at import time), then use `ingest`.
"""
