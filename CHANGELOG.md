# Changelog

## [Unreleased] — Phase 0: Stabilize

### Added
- pytest test suite (55 passing, 2 documented xfail) covering all HTTP endpoints, per-pipeline-layer behaviour, and the observability surface.
- GitHub Actions CI: backend tests + frontend build on every PR and main push (read-only token, 15-min job timeouts).
- Pydantic response models for `/api/upload`, `/api/load_external`, `/api/validate_external`, `/api/dashboard`.
- Observability:
  - `structlog` JSON stdout logging with request-id propagation.
  - `X-Request-ID` middleware (contextvar-based, echoes/generates the header).
  - `/healthz` (liveness) and `/readyz` (readiness, extensible via `register_readiness_check`).
  - Prometheus `/metrics` endpoint: HTTP request counter/histogram/in-flight gauge + per-pipeline-layer duration histogram.
  - Optional OpenTelemetry tracing (enable via `OTEL_EXPORTER_OTLP_ENDPOINT`).
  - Optional Sentry error reporting (enable via `SENTRY_DSN`).
- `.env.example` documenting every runtime env var.
- `pyproject.toml` with pytest + coverage configuration.

### Changed
- `CORSMiddleware` registered (was imported but unused). Allowlist via `CORS_ALLOW_ORIGINS`.
- Layer 3 correlation threshold reads `src.config.MIN_CORRELATION` (was hard-coded `0.5`).
- Logging migrated to 12-factor stdout JSON; legacy file-rotation handlers available behind `LOG_FILE_HANDLERS=true`.

### Removed
- `src/viz/eda_visualizer.py` — dead code (no Python callers).
- Duplicate `serve_dynamic_assets` route + duplicate `import re` + redundant `app.mount("/assets", ...)` in `main.py`.

### Known issues surfaced by the new test suite (deferred to later phases)
- **`/api/upload` & `/api/upload/stream` return HTTP 500 instead of 400/422 on empty multipart filename.** Root cause: the global `RequestValidationError` handler in `main.py` calls `JSONResponse(content={..., "errors": exc.errors()})`; when a validation error carries a non-serializable `ctx.error` (e.g. a raw `ValueError`), `json.dumps` fails and the 422 is never rendered. Affects ANY validation error with a `ctx.error`, not just filenames. Tracked by `xfail(strict=True)` test `test_upload_rejects_empty_filename`. Belongs in the Security Hardening / API Hygiene sub-plan.
- **Layer 2 classifier misclassifies short unique-date object columns as `identifier` instead of `datetime`.** Root cause: in `src/analysis/layer_2_classifier.py::run_semantic_classification`, the identifier-confidence check runs before the object-column datetime parse, so a low-row column of unique ISO dates trips the high-cardinality identifier signal. Tracked by `xfail(strict=True)` test `test_layer2_detects_datetime_role`. Belongs in the Pipeline Algorithms sub-plan.

### Known limitations
- **Backend coverage is 57% (src/ + main.py), below the Phase 0 target of ≥60%.** The shortfall is concentrated in modules with low test exposure: `src/data/parser.py` (25%), `src/viz/utils.py` (24%), `src/viz/plotly_renderer.py` (42%), `src/diagnostics/tracer.py` (47%, plus a duplicate `src/diagnostics/diagnostics/tracer.py` at 0%), and `src/auth.py` (50%). HTTP endpoints, the four pipeline layers, observability, and config are well covered (74%–100%). Closing the gap (parser and viz utility tests, removing the duplicate tracer module) is deferred to a later phase rather than padded with filler tests.

### Deferred to later phases
See `updrage-plan.txt` §11 — issues 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 21, 22.

### Phase 0 test coverage
Backend coverage: 57% (src/ + main.py) — below the ≥60% Phase 0 target; see Known limitations.
