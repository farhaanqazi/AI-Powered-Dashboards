# Phase 0 — Stabilize: API Hygiene + Test/CI Baseline + Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the safety net for an enterprise-grade upgrade — comprehensive automated tests, a green CI pipeline, observable runtime (structured logs, traces, metrics, health checks, error reporting), and a clean API surface (Pydantic response models, no duplicate routes, registered CORS, no dead code, no config drift) — before any load-bearing architectural change touches the codebase.

**Architecture:** No new infrastructure. Pure code work inside the existing FastAPI single-container app. Tests use `fastapi.testclient.TestClient` (sync, no new test stack). CI runs on GitHub Actions. Logging migrates from file-rotation to 12-factor stdout JSON via `structlog`; old file handlers stay on by default for local dev. OpenTelemetry, Prometheus, and Sentry are all wired but **no-op when their env vars are unset** so the current HF Space deployment keeps working unchanged.

**Tech Stack:** Python 3.9, FastAPI 0.109.2, Pydantic v2, pytest, httpx (via TestClient), GitHub Actions, structlog, opentelemetry-instrumentation-fastapi, opentelemetry-exporter-otlp, prometheus-client, sentry-sdk[fastapi]. No frontend changes in this phase.

---

## Spec coverage map

| §11 issue | Task |
|---|---|
| 10 — CORSMiddleware imported, not registered | Task 5 |
| 11 — `MIN_CORRELATION` config drift | Task 7 |
| 12 — `viz/eda_visualizer.py` dead code | Task 6 |
| 13 — No automated tests on request path | Tasks 1, 2, 8, 9, 10, 11 |
| 19 — Duplicate `serve_dynamic_assets` + duplicate `import re` | Task 4 |
| 20 — No Pydantic response models | Task 3 |
| (enterprise: observability gap) | Tasks 12, 13, 14, 15, 16, 17, 18 |

Issues NOT addressed in Phase 0 (deferred): 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 21, 22.

---

## File structure

### Files created
- `tests/conftest.py` — shared pytest fixtures (test client, sample CSVs, env setup)
- `tests/fixtures/sample_data.csv` — fixture for the existing `test_pipeline.py` (currently missing → causes its test to fail)
- `tests/fixtures/numeric_with_corr.csv` — two strongly correlated numeric columns
- `tests/fixtures/bad.html` — HTML masquerading as data (rejection test)
- `tests/test_api_upload.py` — `/api/upload` endpoint tests
- `tests/test_api_stream.py` — `/api/upload/stream` SSE tests
- `tests/test_api_external.py` — `/api/validate_external` + `/api/load_external` tests
- `tests/test_api_dashboard.py` — `/api/dashboard` tests
- `tests/test_pipeline_layers.py` — per-layer pipeline unit tests
- `tests/test_observability.py` — `/healthz`, `/readyz`, `/metrics`, request-id propagation tests
- `tests/test_response_schemas.py` — Pydantic response-model contract tests
- `src/api/__init__.py`
- `src/api/schemas.py` — Pydantic response models
- `src/observability/__init__.py`
- `src/observability/logging.py` — structlog config + request_id contextvar
- `src/observability/request_id.py` — ASGI middleware to attach request id
- `src/observability/tracing.py` — OpenTelemetry init (no-op if `OTEL_EXPORTER_OTLP_ENDPOINT` unset)
- `src/observability/metrics.py` — Prometheus collectors + `/metrics` endpoint factory
- `src/observability/sentry.py` — Sentry SDK init (no-op if `SENTRY_DSN` unset)
- `src/observability/health.py` — `/healthz` and `/readyz` route factory
- `.github/workflows/ci.yml` — GitHub Actions: lint + pytest on PR + push to main
- `.env.example` — documents every env var the app reads
- `pyproject.toml` — pytest + coverage config (none currently exists)

### Files modified
- `main.py` — register CORS, wire observability, mount `/metrics`/`/healthz`/`/readyz`, drop duplicate routes, drop duplicate `import re`, attach `response_model=` to endpoints, gate diag endpoints behind env flag
- `src/config.py` — wire `MIN_CORRELATION` to Layer 3, document via `.env.example`
- `src/analysis/layer_3_relational.py` — read `MIN_CORRELATION` from config instead of literal `0.5`
- `src/core/pipeline.py` — add per-layer OTel span + Prometheus histogram instrumentation
- `src/logger.py` — route through `structlog` for JSON stdout; preserve existing file handlers behind env flag for local-dev parity
- `requirements.txt` — pin new deps
- `tests/test_main.py` — remove the commented-out dead test stub (or replace with smoke test)
- `tests/core/test_pipeline.py` — leave intact; Task 1 creates the missing fixture so this passes

### Files deleted
- `src/viz/eda_visualizer.py` — confirmed unreferenced by any Python module (only mentioned in README + the planning doc)

---

## Conventions used throughout this plan

- **Test client:** all endpoint tests use `from fastapi.testclient import TestClient` and pass `headers={"X-Guest-Mode": "1", "X-Guest-Session-Id": "test-sid"}` so the Clerk JWT path is bypassed. No JWT mocking required.
- **Commit message style:** Conventional Commits (`feat:`, `fix:`, `test:`, `chore:`, `refactor:`, `docs:`).
- **Each task's final step is a commit.** Multiple commits per phase. Push to a feature branch (e.g., `phase-0-stabilize`), not directly to `main`.
- **Backwards compatibility:** existing endpoints keep their exact response shapes. Pydantic models are added with `model_config = ConfigDict(extra="allow")` to avoid breaking the frontend on any field the doc missed.
- **Run all commands from repo root** (`f:/AI Powered Dashboards`).

---

## Task 1: Pytest scaffold + smoke test + missing fixture

**Files:**
- Create: `pyproject.toml`
- Create: `tests/conftest.py`
- Create: `tests/fixtures/sample_data.csv`
- Modify: `tests/test_main.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add test deps to `requirements.txt`**

Append these lines to `requirements.txt`:

```
pytest-asyncio==0.23.5
pytest-cov==4.1.0
httpx==0.27.0
```

(`pytest==8.0.0` is already pinned. `httpx` is needed because `fastapi.testclient.TestClient` depends on it in recent fastapi versions.)

Then install:

```bash
pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-ra",
    "--strict-markers",
    "--strict-config",
]
filterwarnings = [
    "ignore::DeprecationWarning",
]

[tool.coverage.run]
source = ["src", "main"]
omit = [
    "tests/*",
    "src/viz/eda_visualizer.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
```

- [ ] **Step 3: Create the missing fixture CSV**

Create `tests/fixtures/sample_data.csv` with content:

```csv
record_id,transaction_date,amount,category,description
1,2024-01-15,101.15,groceries,Weekly food shop
2,2024-01-16,52.40,fuel,Petrol station
3,2024-01-17,1200.00,rent,Monthly rent
4,2024-01-18,15.75,coffee,Morning espresso
5,2024-01-19,89.30,groceries,Mid-week top up
6,2024-01-20,250.00,utilities,Electricity bill
7,2024-01-21,42.10,fuel,Petrol station
8,2024-01-22,12.50,coffee,Afternoon flat white
9,2024-01-23,76.80,groceries,Veg market
10,2024-01-24,1200.00,rent,Monthly rent
```

- [ ] **Step 4: Create `tests/conftest.py` with shared fixtures**

```python
"""Shared pytest fixtures for the AI-Powered Dashboards test suite."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main as main_module

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient that auto-attaches guest-mode auth headers."""
    c = TestClient(main_module.app)
    c.headers.update({
        "X-Guest-Mode": "1",
        "X-Guest-Session-Id": "pytest-session",
    })
    yield c


@pytest.fixture(autouse=True)
def _reset_storage():
    """Clear the in-process dashboard store between tests."""
    main_module.dashboard_storage.clear()
    yield
    main_module.dashboard_storage.clear()


@pytest.fixture
def sample_csv_bytes() -> bytes:
    return (FIXTURES / "sample_data.csv").read_bytes()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "sample_data.csv")


@pytest.fixture
def upload_files(sample_csv_bytes):
    return {"dataset": ("sample_data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
```

- [ ] **Step 5: Replace the commented-out `tests/test_main.py` with a real smoke test**

Replace the entire contents of `tests/test_main.py` with:

```python
"""Top-level smoke tests."""


def test_root_serves_spa(client):
    """GET / returns the built index.html (or 404 if dist not built yet)."""
    response = client.get("/")
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        assert "text/html" in response.headers.get("content-type", "")


def test_unknown_api_returns_404(client):
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
```

- [ ] **Step 6: Run the existing test suite and confirm it passes**

```bash
pytest -v
```

Expected: 3 passing tests (`tests/core/test_pipeline.py::test_pipeline_end_to_end_from_dataframe` + the two new smoke tests). If `test_pipeline_end_to_end_from_dataframe` was already passing, you've kept it green; if it was failing for "fixture missing", it now passes.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml tests/conftest.py tests/fixtures/sample_data.csv tests/test_main.py requirements.txt
git commit -m "test: scaffold pytest infrastructure with shared fixtures"
```

---

## Task 2: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write the workflow file**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  backend:
    name: Backend tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.9
        uses: actions/setup-python@v5
        with:
          python-version: "3.9"
          cache: pip

      - name: Install Python deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run pytest with coverage
        run: pytest --cov --cov-report=term-missing --cov-report=xml -v

      - name: Upload coverage artefact
        uses: actions/upload-artifact@v4
        with:
          name: coverage-xml
          path: coverage.xml
          if-no-files-found: warn

  frontend-build:
    name: Frontend build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Node 18
        uses: actions/setup-node@v4
        with:
          node-version: "18"
          cache: npm
          cache-dependency-path: frontend/package-lock.json

      - name: Install JS deps
        working-directory: frontend
        run: npm ci

      - name: Lint
        working-directory: frontend
        run: npm run lint --if-present

      - name: Build
        working-directory: frontend
        env:
          VITE_CLERK_PUBLISHABLE_KEY: pk_test_placeholder
        run: npm run build
```

- [ ] **Step 2: Verify the workflow YAML is syntactically valid**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
```

Expected: no output, exit code 0. (If `yaml` is missing: `pip install pyyaml` first.)

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: run backend tests + frontend build on PR and main"
```

- [ ] **Step 4: Push the branch and confirm CI runs**

```bash
git push -u origin phase-0-stabilize
```

Expected: open the GitHub PR view; both `Backend tests` and `Frontend build` jobs start. Both should pass. If `Frontend build` fails on `npm run lint`, that's fine — the `--if-present` flag lets it pass when no lint script exists; if a lint script exists and fails, fix the lint errors as a separate commit.

---

## Task 3: Pydantic response models

**Files:**
- Create: `src/api/__init__.py`
- Create: `src/api/schemas.py`
- Create: `tests/test_response_schemas.py`
- Modify: `main.py`

- [ ] **Step 1: Write tests asserting the response schemas exist and validate sample payloads**

Create `tests/test_response_schemas.py`:

```python
"""Pydantic response-model contract tests."""
import pytest

from src.api import schemas


def test_upload_response_validates_minimal_payload():
    payload = {
        "status": "success",
        "trace_id": "abc-123",
        "data": {
            "dataset_profile": {},
            "kpis": [],
            "charts": [],
            "primary_chart": None,
            "category_charts": {},
            "all_charts": [],
            "original_filename": "x.csv",
            "errors": [],
            "warnings": [],
            "critical_totals": {},
            "critical_full_dataset_aggregates": {},
            "eda_summary": {},
        },
    }
    parsed = schemas.UploadResponse.model_validate(payload)
    assert parsed.status == "success"
    assert parsed.trace_id == "abc-123"


def test_dashboard_response_validates_empty_state():
    payload = {
        "status": "empty",
        "timestamp": "2026-01-01T00:00:00",
        "metadata": {"hint": "Upload a dataset to generate insights"},
        "kpis": [],
        "charts": [],
        "eda": {},
        "errors": [],
        "warnings": [],
        "message": "Dashboard initializing.",
        "dataset_profile": {},
        "primary_chart": None,
        "category_charts": {},
        "all_charts": [],
        "original_filename": "",
        "critical_totals": {},
        "critical_full_dataset_aggregates": {},
        "eda_summary": {},
    }
    parsed = schemas.DashboardResponse.model_validate(payload)
    assert parsed.status == "empty"


def test_dashboard_response_validates_ready_state():
    payload = {
        "status": "ready",
        "timestamp": "2026-01-01T00:00:00",
        "metadata": {"columns": 5, "rows": 10, "filename": "x.csv"},
        "kpis": [{"name": "Amount", "score": 0.9}],
        "charts": [],
        "eda": {},
        "errors": [],
        "warnings": [],
        "message": None,
        "dataset_profile": {"n_cols": 5, "n_rows": 10},
        "primary_chart": None,
        "category_charts": {},
        "all_charts": [],
        "original_filename": "x.csv",
        "critical_totals": {},
        "critical_full_dataset_aggregates": {},
        "eda_summary": {},
    }
    parsed = schemas.DashboardResponse.model_validate(payload)
    assert parsed.status == "ready"


def test_validate_external_response():
    parsed = schemas.ValidateExternalResponse.model_validate({"ok": True})
    assert parsed.ok is True


def test_load_external_response_validates():
    payload = {
        "status": "success",
        "trace_id": "xyz",
        "data": {
            "dataset_profile": {},
            "kpis": [],
            "charts": [],
            "primary_chart": None,
            "category_charts": {},
            "all_charts": [],
            "original_filename": "url",
            "errors": [],
            "warnings": [],
            "critical_totals": {},
            "critical_full_dataset_aggregates": {},
            "eda_summary": {},
        },
    }
    parsed = schemas.LoadExternalResponse.model_validate(payload)
    assert parsed.status == "success"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_response_schemas.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.api'`.

- [ ] **Step 3: Create `src/api/__init__.py`**

```python
"""HTTP layer — Pydantic request/response schemas live in `schemas.py`."""
```

- [ ] **Step 4: Create `src/api/schemas.py`**

```python
"""Pydantic response models for every HTTP endpoint in `main.py`.

These describe the wire contract. Frontend code (`frontend/src/services/api.js`)
relies on these field names exactly. Add new fields as `Optional` to keep
backward compatibility; never rename existing fields without a frontend change.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


_PERMISSIVE = ConfigDict(extra="allow")


class DashboardPayload(BaseModel):
    """The shared payload nested under `data` in upload/load responses and
    surfaced at the top level by GET /api/dashboard."""

    model_config = _PERMISSIVE

    dataset_profile: Dict[str, Any] = Field(default_factory=dict)
    kpis: List[Dict[str, Any]] = Field(default_factory=list)
    charts: List[Dict[str, Any]] = Field(default_factory=list)
    primary_chart: Optional[Dict[str, Any]] = None
    category_charts: Dict[str, Any] = Field(default_factory=dict)
    all_charts: List[Dict[str, Any]] = Field(default_factory=list)
    original_filename: str = ""
    errors: List[Any] = Field(default_factory=list)
    warnings: List[Any] = Field(default_factory=list)
    critical_totals: Dict[str, Any] = Field(default_factory=dict)
    critical_full_dataset_aggregates: Dict[str, Any] = Field(default_factory=dict)
    eda_summary: Dict[str, Any] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    model_config = _PERMISSIVE
    status: str
    trace_id: str
    data: DashboardPayload


class LoadExternalResponse(BaseModel):
    model_config = _PERMISSIVE
    status: str
    trace_id: str
    data: DashboardPayload


class ValidateExternalResponse(BaseModel):
    model_config = _PERMISSIVE
    ok: bool


class DashboardResponse(BaseModel):
    model_config = _PERMISSIVE

    status: str
    timestamp: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    kpis: List[Dict[str, Any]] = Field(default_factory=list)
    charts: List[Dict[str, Any]] = Field(default_factory=list)
    eda: Dict[str, Any] = Field(default_factory=dict)
    errors: List[Any] = Field(default_factory=list)
    warnings: List[Any] = Field(default_factory=list)
    message: Optional[str] = None
    dataset_profile: Dict[str, Any] = Field(default_factory=dict)
    primary_chart: Optional[Dict[str, Any]] = None
    category_charts: Dict[str, Any] = Field(default_factory=dict)
    all_charts: List[Dict[str, Any]] = Field(default_factory=list)
    original_filename: str = ""
    critical_totals: Dict[str, Any] = Field(default_factory=dict)
    critical_full_dataset_aggregates: Dict[str, Any] = Field(default_factory=dict)
    eda_summary: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    model_config = _PERMISSIVE
    message: str
    error_type: Optional[str] = None
    error_detail: Optional[str] = None
    errors: Optional[List[Any]] = None
```

- [ ] **Step 5: Run the schema test to verify it passes**

```bash
pytest tests/test_response_schemas.py -v
```

Expected: 5 passing tests.

- [ ] **Step 6: Wire `response_model=` onto endpoints in `main.py`**

Add this import near the top of `main.py` (after the existing internal imports block):

```python
from src.api.schemas import (
    UploadResponse,
    LoadExternalResponse,
    ValidateExternalResponse,
    DashboardResponse,
)
```

Change the decorators for the affected routes:

```python
@app.post("/api/upload", response_model=UploadResponse)
async def api_upload(...):
    ...

@app.post("/api/validate_external", response_model=ValidateExternalResponse)
async def api_validate_external(...):
    ...

@app.post("/api/load_external", response_model=LoadExternalResponse)
async def api_load_external(...):
    ...

@app.get("/api/dashboard", response_model=DashboardResponse)
async def api_get_dashboard(...):
    ...
```

(Do NOT attach a `response_model` to `/api/upload/stream` — it returns a `StreamingResponse`, which FastAPI's response-model machinery would corrupt.)

- [ ] **Step 7: Run all tests to confirm no regression**

```bash
pytest -v
```

Expected: all tests pass, including the smoke test and pipeline test.

- [ ] **Step 8: Commit**

```bash
git add src/api/ tests/test_response_schemas.py main.py
git commit -m "feat: add Pydantic response models for all HTTP endpoints"
```

---

## Task 4: Dedupe routes & remove stray imports

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Write a regression test that fails today**

Append to `tests/test_main.py`:

```python
def test_main_module_has_no_duplicate_route_names():
    """Detect duplicate route handler function names in main.py.

    Currently `serve_dynamic_assets` is defined twice. Both definitions are
    overwritten in the FastAPI routing table, but the dead one is a bug magnet.
    """
    import main as main_module
    seen = {}
    for route in main_module.app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            continue
        name = endpoint.__qualname__
        path = getattr(route, "path", "?")
        seen.setdefault(name, []).append(path)
    duplicates = {n: paths for n, paths in seen.items() if len(paths) > 1}
    assert not duplicates, f"Duplicate endpoint functions: {duplicates}"


def test_main_imports_re_only_once():
    src = (
        __import__("pathlib").Path(__file__).resolve().parents[1] / "main.py"
    ).read_text(encoding="utf-8")
    occurrences = src.count("\nimport re\n") + (1 if src.startswith("import re\n") else 0)
    assert occurrences == 1, f"`import re` appears {occurrences} times in main.py"
```

- [ ] **Step 2: Run tests to verify both fail**

```bash
pytest tests/test_main.py -v
```

Expected: FAIL — `test_main_module_has_no_duplicate_route_names` reports `serve_dynamic_assets` mapped to two paths; `test_main_imports_re_only_once` reports 2 occurrences.

- [ ] **Step 3: Remove the duplicate `import re` and consolidate asset routes in `main.py`**

In `main.py`:

1. Delete the line `import re` at line ~372 (the second one, inside the file body).
2. Delete the entire second `serve_dynamic_assets` function definition (around lines 452-459 — the one decorated with `@app.get("/assets/{subdir}/{filename}")`).
3. Keep only the first `serve_dynamic_assets` (the one decorated with `@app.get("/assets/{full_path:path}")` near line 375). It already handles the nested-subdirectory case via `full_path`.
4. Decide whether to also drop the `app.mount("/assets", StaticFiles(...))` at the bottom — the explicit route already covers all assets. Drop the mount to remove the third layer of overlap.

After editing, the asset-serving section of `main.py` should contain exactly:

```python
@app.get("/assets/{full_path:path}")
async def serve_dynamic_assets(full_path: str):
    filepath = f"frontend/dist/assets/{full_path}"
    if os.path.exists(filepath):
        return FileResponse(filepath)
    raise HTTPException(status_code=404, detail="Asset not found")
```

…and no `app.mount("/assets", ...)` call.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_main.py -v
```

Expected: PASS for both regression tests.

- [ ] **Step 5: Manually smoke-test asset serving**

Start the app locally:

```bash
uvicorn main:app --port 8000 &
curl -I http://localhost:8000/assets/missing.js
```

Expected: HTTP 404 with `{"detail": "Asset not found"}` body.

If `frontend/dist/` is built:

```bash
ls frontend/dist/assets/
# Pick any existing file, e.g. assets/<BUILD_ID>/index-abc.js
curl -I http://localhost:8000/assets/<BUILD_ID>/index-abc.js
```

Expected: HTTP 200.

Kill the server:

```bash
kill %1
```

- [ ] **Step 6: Run the full test suite**

```bash
pytest -v
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "refactor: remove duplicate serve_dynamic_assets and import re in main.py"
```

---

## Task 5: Register CORSMiddleware

**Files:**
- Modify: `main.py`
- Modify: `src/config.py`

- [ ] **Step 1: Write a failing test**

Create a new test file `tests/test_cors.py`:

```python
def test_cors_preflight_returns_allow_headers(client):
    response = client.options(
        "/api/dashboard",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert response.status_code in (200, 204)
    assert "access-control-allow-origin" in {k.lower() for k in response.headers}


def test_cors_actual_request_returns_allow_origin(client):
    response = client.get(
        "/api/dashboard",
        headers={"Origin": "http://localhost:5173"},
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_cors.py -v
```

Expected: FAIL — no `access-control-allow-origin` header present.

- [ ] **Step 3: Add CORS config knob in `src/config.py`**

Append to `src/config.py`:

```python
# --- HTTP Configuration ---
_default_origins = "http://localhost:5173,http://localhost:8000"
CORS_ALLOW_ORIGINS = [
    o.strip()
    for o in os.environ.get("CORS_ALLOW_ORIGINS", _default_origins).split(",")
    if o.strip()
]
```

- [ ] **Step 4: Register `CORSMiddleware` in `main.py`**

The `CORSMiddleware` is already imported at the top of `main.py`. Just below the `app = FastAPI()` line, add:

```python
from src.config import CORS_ALLOW_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_cors.py -v
```

Expected: PASS for both.

- [ ] **Step 6: Run the full suite**

```bash
pytest -v
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add main.py src/config.py tests/test_cors.py
git commit -m "feat: register CORSMiddleware with env-driven allowlist"
```

---

## Task 6: Delete `src/viz/eda_visualizer.py`

**Files:**
- Delete: `src/viz/eda_visualizer.py`
- Modify: `README.md` (if it references the module)

- [ ] **Step 1: Verify the file has zero Python callers**

```bash
grep -r "eda_visualizer" --include='*.py' .
```

Expected: no matches under any `.py` file.

If there are matches in `.py` files, **stop** and audit them; the module isn't safe to delete.

- [ ] **Step 2: Delete the file**

```bash
git rm src/viz/eda_visualizer.py
```

- [ ] **Step 3: Update `README.md` if it mentions `eda_visualizer`**

```bash
grep -n "eda_visualizer" README.md
```

If any line references it as part of the architecture, edit that section to remove the reference. If the README has a "modules" or "architecture" section listing it, drop the bullet.

- [ ] **Step 4: Run full test suite**

```bash
pytest -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: drop unused src/viz/eda_visualizer.py"
```

---

## Task 7: Fix `MIN_CORRELATION` config drift

**Files:**
- Modify: `src/config.py`
- Modify: `src/analysis/layer_3_relational.py`
- Create: `tests/test_layer_3_correlation_threshold.py`

- [ ] **Step 1: Decide the default threshold**

The old literal in `layer_3_relational.py` is `0.5`. The current config default is `0.1`. Reading the function comment ("Only report moderate to strong correlations") and the EDA pipeline's downstream use, the runtime expectation is `0.5`. Change `src/config.py` default to `0.5` and document that callers can override via env.

Edit `src/config.py` line 29:

```python
MIN_CORRELATION = float(os.environ.get("MIN_CORRELATION", 0.5))
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_layer_3_correlation_threshold.py`:

```python
"""Verify Layer 3 reads its threshold from config, not a literal."""
import numpy as np
import pandas as pd

from src.analysis.data_structures import EnrichedProfile
from src.analysis.layer_3_relational import run_relational_analysis
from src import config


def _make_profile(name: str, std: float) -> EnrichedProfile:
    return EnrichedProfile(
        name=name,
        dtype="float64",
        role="numeric",
        semantic_tags=[],
        null_count=0,
        unique_count=100,
        stats={"std": std, "count": 100},
        top_categories=[],
    )


def test_min_correlation_threshold_is_respected(monkeypatch):
    rng = np.random.default_rng(seed=42)
    base = rng.normal(0, 1, 200)
    weak_noise = rng.normal(0, 1, 200)
    # Weak correlation (~0.3) — should be excluded under default 0.5 threshold
    weak = 0.3 * base + 0.95 * weak_noise
    # Strong correlation (~0.9)
    strong = 0.9 * base + 0.1 * rng.normal(0, 1, 200)

    df = pd.DataFrame({"base": base, "weak": weak, "strong": strong})
    profiles = {
        "base": _make_profile("base", float(df["base"].std())),
        "weak": _make_profile("weak", float(df["weak"].std())),
        "strong": _make_profile("strong", float(df["strong"].std())),
    }

    monkeypatch.setattr(config, "MIN_CORRELATION", 0.5)
    insights = run_relational_analysis(df, profiles)
    pairs = [tuple(sorted(i.columns)) for i in insights]
    assert ("base", "strong") in pairs
    assert ("base", "weak") not in pairs

    monkeypatch.setattr(config, "MIN_CORRELATION", 0.2)
    insights_low = run_relational_analysis(df, profiles)
    pairs_low = [tuple(sorted(i.columns)) for i in insights_low]
    assert ("base", "weak") in pairs_low
```

- [ ] **Step 3: Run to verify the test fails**

```bash
pytest tests/test_layer_3_correlation_threshold.py -v
```

Expected: FAIL on the second `monkeypatch` block — Layer 3 still uses the literal `0.5`, so `weak` never appears regardless of config.

- [ ] **Step 4: Wire `MIN_CORRELATION` into Layer 3**

Edit `src/analysis/layer_3_relational.py`:

Add the import at the top (after the existing imports):

```python
from src import config as _cfg
```

Change the threshold line. Currently:

```python
if pd.isna(corr) or abs(corr) < 0.5 or p_value > 0.05:
    continue
```

Replace with:

```python
if pd.isna(corr) or abs(corr) < _cfg.MIN_CORRELATION or p_value > 0.05:
    continue
```

Note the `strength = "strong" if abs(corr) >= 0.7 else "moderate"` line stays as-is — that's a labelling threshold, not a reporting threshold.

- [ ] **Step 5: Run the new test plus the existing pipeline test**

```bash
pytest tests/test_layer_3_correlation_threshold.py tests/core/test_pipeline.py -v
```

Expected: all pass.

- [ ] **Step 6: Run full suite**

```bash
pytest -v
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/config.py src/analysis/layer_3_relational.py tests/test_layer_3_correlation_threshold.py
git commit -m "fix: wire MIN_CORRELATION config to Layer 3 (was hard-coded to 0.5)"
```

---

## Task 8: Endpoint test coverage — `/api/upload` and `/api/upload/stream`

**Files:**
- Create: `tests/test_api_upload.py`
- Create: `tests/test_api_stream.py`

- [ ] **Step 1: Create `tests/test_api_upload.py`**

```python
"""Tests for POST /api/upload (sync upload endpoint)."""
import io


def test_upload_happy_path_returns_200_and_dashboard_payload(client, upload_files):
    response = client.post("/api/upload", files=upload_files)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    assert "trace_id" in body
    data = body["data"]
    assert "dataset_profile" in data
    assert "kpis" in data
    assert "all_charts" in data


def test_upload_rejects_non_csv_extension(client):
    files = {"dataset": ("not_a_csv.txt", io.BytesIO(b"col\n1\n"), "text/plain")}
    response = client.post("/api/upload", files=files)
    assert response.status_code == 400
    assert "csv" in response.json()["detail"].lower()


def test_upload_rejects_path_traversal_filename(client):
    files = {"dataset": ("../escape.csv", io.BytesIO(b"col\n1\n"), "text/csv")}
    response = client.post("/api/upload", files=files)
    assert response.status_code == 400
    assert "invalid" in response.json()["detail"].lower()


def test_upload_rejects_empty_filename(client):
    files = {"dataset": ("", io.BytesIO(b"col\n1\n"), "text/csv")}
    response = client.post("/api/upload", files=files)
    assert response.status_code in (400, 422)


def test_upload_persists_data_so_dashboard_endpoint_can_read_it(client, upload_files):
    upload = client.post("/api/upload", files=upload_files)
    assert upload.status_code == 200
    dashboard = client.get("/api/dashboard")
    assert dashboard.status_code == 200
    assert dashboard.json()["status"] == "ready"
```

- [ ] **Step 2: Create `tests/test_api_stream.py`**

```python
"""Tests for POST /api/upload/stream (Server-Sent Events)."""
import json


def _parse_sse(body_text: str):
    events = []
    for chunk in body_text.split("\n\n"):
        chunk = chunk.strip()
        if not chunk.startswith("data: "):
            continue
        events.append(json.loads(chunk[len("data: "):]))
    return events


def test_stream_emits_expected_phases_in_order(client, upload_files):
    with client.stream("POST", "/api/upload/stream", files=upload_files) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = response.read().decode("utf-8")

    events = _parse_sse(body)
    phases = [e["phase"] for e in events]
    assert phases[0] in ("reading", "preparing", "profiling")
    assert phases[-1] == "done"
    expected_phase_set = {
        "reading", "preparing", "profiling", "classifying",
        "relating", "eda", "kpis", "rendering", "done",
    }
    assert set(phases).issubset(expected_phase_set | {"error"})
    assert events[-1]["percent"] == 100
    assert "data" in events[-1]
    assert "trace_id" in events[-1]


def test_stream_rejects_non_csv(client):
    import io
    files = {"dataset": ("x.txt", io.BytesIO(b"a,b\n1,2\n"), "text/plain")}
    response = client.post("/api/upload/stream", files=files)
    assert response.status_code == 400
```

- [ ] **Step 3: Run the new endpoint tests**

```bash
pytest tests/test_api_upload.py tests/test_api_stream.py -v
```

Expected: all pass. If `test_stream_emits_expected_phases_in_order` fails because event order differs slightly, inspect printed events and adjust assertions to be order-tolerant where the pipeline reasonably permits.

- [ ] **Step 4: Run full suite**

```bash
pytest -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add tests/test_api_upload.py tests/test_api_stream.py
git commit -m "test: cover /api/upload and /api/upload/stream endpoints"
```

---

## Task 9: Endpoint test coverage — `/api/validate_external` and `/api/load_external`

**Files:**
- Create: `tests/test_api_external.py`

- [ ] **Step 1: Write the test file**

```python
"""Tests for POST /api/validate_external and POST /api/load_external.

These tests stub `requests.get` so they don't perform real network I/O.
"""
from unittest.mock import patch, MagicMock


# ---------- /api/validate_external ----------

def test_validate_external_rejects_empty(client):
    response = client.post("/api/validate_external", json={"external_source": ""})
    assert response.status_code == 400


def test_validate_external_rejects_malformed_url(client):
    response = client.post(
        "/api/validate_external", json={"external_source": "http://"},
    )
    assert response.status_code == 400


def test_validate_external_rejects_kaggle_dataset_page_url(client):
    response = client.post(
        "/api/validate_external",
        json={"external_source": "https://www.kaggle.com/datasets/foo/bar"},
    )
    assert response.status_code == 400
    assert "kaggle" in response.json()["detail"].lower()


def test_validate_external_accepts_valid_slug(client):
    response = client.post(
        "/api/validate_external", json={"external_source": "owner/dataset"},
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_validate_external_rejects_bad_slug(client):
    response = client.post(
        "/api/validate_external", json={"external_source": "missing-slash"},
    )
    assert response.status_code == 400


def test_validate_external_accepts_raw_csv_url(client):
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {"Content-Type": "text/csv"}
    fake_response.iter_content = lambda chunk_size: iter([b"col1,col2\n1,2\n"])
    fake_response.close = lambda: None

    with patch("main.requests.get", return_value=fake_response):
        response = client.post(
            "/api/validate_external",
            json={"external_source": "https://example.com/data.csv"},
        )
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_validate_external_rejects_html_content_type(client):
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {"Content-Type": "text/html; charset=utf-8"}
    fake_response.iter_content = lambda chunk_size: iter([b"<html></html>"])
    fake_response.close = lambda: None

    with patch("main.requests.get", return_value=fake_response):
        response = client.post(
            "/api/validate_external",
            json={"external_source": "https://example.com/page"},
        )
    assert response.status_code == 400
    assert "html" in response.json()["detail"].lower() or "csv" in response.json()["detail"].lower()


def test_validate_external_rejects_html_body_with_csv_content_type(client):
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {"Content-Type": "text/csv"}
    fake_response.iter_content = lambda chunk_size: iter([b"<!DOCTYPE html><html></html>"])
    fake_response.close = lambda: None

    with patch("main.requests.get", return_value=fake_response):
        response = client.post(
            "/api/validate_external",
            json={"external_source": "https://example.com/sneaky"},
        )
    assert response.status_code == 400


# ---------- /api/load_external ----------

def test_load_external_rejects_bad_slug_shape(client):
    response = client.post(
        "/api/load_external", json={"external_source": "no-slash-here"},
    )
    assert response.status_code == 400


def test_load_external_loads_url_and_returns_dashboard(client):
    import pandas as pd
    from src.data.parser import LoadResult

    fake_result = LoadResult(
        df=pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        }),
        success=True,
        warnings=[],
    )

    with patch("main.load_csv_from_url", return_value=fake_result):
        response = client.post(
            "/api/load_external",
            json={"external_source": "https://example.com/data.csv"},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    assert body["data"]["original_filename"] == "https://example.com/data.csv"
```

(Note: `LoadResult` import — verify the actual class name in `src/data/parser.py`. If the constructor signature differs, adjust the fake. If the field is named differently than `success` / `df` / `warnings`, adjust accordingly. The plan assumes the public symbol matches what `main.py` already consumes; quick verification step in the next sub-step.)

- [ ] **Step 2: Confirm `LoadResult` API matches the test**

```bash
grep -n "class LoadResult\|LoadResult(" src/data/parser.py
```

Expected: a class definition with `df`, `success`, `warnings` (and possibly more) — `main.py` already accesses `result.success`, `result.df`, `result.warnings`. If the constructor refuses positional args (e.g., `@dataclass(kw_only=True)`), adjust the `fake_result = LoadResult(df=..., success=True, warnings=[])` call to use keyword args. If it's a `NamedTuple` or `pydantic.BaseModel`, that already works as written.

- [ ] **Step 3: Run the test file**

```bash
pytest tests/test_api_external.py -v
```

Expected: all pass.

- [ ] **Step 4: Run full suite**

```bash
pytest -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add tests/test_api_external.py
git commit -m "test: cover /api/validate_external and /api/load_external endpoints"
```

---

## Task 10: Endpoint test coverage — `/api/dashboard`

**Files:**
- Create: `tests/test_api_dashboard.py`

- [ ] **Step 1: Write the test file**

```python
"""Tests for GET /api/dashboard (session-keyed retrieval)."""


def test_dashboard_returns_empty_when_no_upload_yet(client):
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "empty"
    assert body["kpis"] == []
    assert body["all_charts"] == []


def test_dashboard_returns_ready_after_upload(client, upload_files):
    upload = client.post("/api/upload", files=upload_files)
    assert upload.status_code == 200
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["original_filename"] == "sample_data.csv"
    assert isinstance(body["metadata"]["rows"], int)
    assert isinstance(body["metadata"]["columns"], int)


def test_dashboard_is_per_session(client, upload_files):
    """Different guest session ids must not see each other's dashboards."""
    upload = client.post(
        "/api/upload", files=upload_files,
        headers={"X-Guest-Mode": "1", "X-Guest-Session-Id": "alice"},
    )
    assert upload.status_code == 200

    bob = client.get(
        "/api/dashboard",
        headers={"X-Guest-Mode": "1", "X-Guest-Session-Id": "bob"},
    )
    assert bob.status_code == 200
    assert bob.json()["status"] == "empty"

    alice = client.get(
        "/api/dashboard",
        headers={"X-Guest-Mode": "1", "X-Guest-Session-Id": "alice"},
    )
    assert alice.status_code == 200
    assert alice.json()["status"] == "ready"
```

- [ ] **Step 2: Run the test file**

```bash
pytest tests/test_api_dashboard.py -v
```

Expected: all pass.

- [ ] **Step 3: Run full suite**

```bash
pytest -v
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add tests/test_api_dashboard.py
git commit -m "test: cover /api/dashboard incl. per-session isolation"
```

---

## Task 11: Pipeline layer unit tests

**Files:**
- Create: `tests/test_pipeline_layers.py`

- [ ] **Step 1: Write the test file**

```python
"""Per-layer unit tests for the analysis pipeline.

These verify each layer in isolation against minimal DataFrames so any
regression points at the offending layer, not the whole 4-layer chain.
"""
import numpy as np
import pandas as pd

from src.analysis.layer_1_profiler import run_syntactic_profiling
from src.analysis.layer_2_classifier import run_semantic_classification
from src.analysis.layer_3_relational import run_relational_analysis
from src.analysis.layer_4_interpreter import determine_kpis, select_charts


# ---------- Layer 1 ----------

def test_layer1_profiles_numeric_column():
    df = pd.DataFrame({"price": [1.0, 2.0, 3.0, 4.0, 5.0]})
    profiles = run_syntactic_profiling(df, max_cols=50)
    assert "price" in profiles
    p = profiles["price"]
    assert p.stats["min"] == 1.0
    assert p.stats["max"] == 5.0
    assert p.stats["mean"] == 3.0


def test_layer1_truncates_to_max_cols():
    df = pd.DataFrame({f"c{i}": range(5) for i in range(10)})
    profiles = run_syntactic_profiling(df, max_cols=3)
    assert len(profiles) == 3


def test_layer1_skips_all_null_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "all_null": [None, None, None]})
    profiles = run_syntactic_profiling(df, max_cols=50)
    assert "a" in profiles
    assert "all_null" not in profiles


# ---------- Layer 2 ----------

def test_layer2_detects_datetime_role():
    df = pd.DataFrame({"d": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]})
    profiles = run_syntactic_profiling(df, max_cols=50)
    enriched = run_semantic_classification(profiles, df)
    assert enriched["d"].role == "datetime"


def test_layer2_detects_numeric_role():
    df = pd.DataFrame({"x": [1.1, 2.2, 3.3, 4.4]})
    profiles = run_syntactic_profiling(df, max_cols=50)
    enriched = run_semantic_classification(profiles, df)
    assert enriched["x"].role == "numeric"


def test_layer2_detects_categorical_role():
    df = pd.DataFrame({"c": ["red", "blue", "red", "green", "blue"] * 10})
    profiles = run_syntactic_profiling(df, max_cols=50)
    enriched = run_semantic_classification(profiles, df)
    assert enriched["c"].role == "categorical"


def test_layer2_detects_identifier_role():
    df = pd.DataFrame({"id": [f"u-{i:05d}" for i in range(200)], "v": list(range(200))})
    profiles = run_syntactic_profiling(df, max_cols=50)
    enriched = run_semantic_classification(profiles, df)
    assert enriched["id"].role == "identifier"


# ---------- Layer 3 ----------

def test_layer3_detects_strong_correlation():
    rng = np.random.default_rng(seed=7)
    base = rng.normal(0, 1, 100)
    df = pd.DataFrame({
        "a": base,
        "b": 0.95 * base + 0.05 * rng.normal(0, 1, 100),
    })
    profiles = run_syntactic_profiling(df, max_cols=50)
    enriched = run_semantic_classification(profiles, df)
    insights = run_relational_analysis(df, enriched)
    assert any({"a", "b"} == set(i.columns) for i in insights)


def test_layer3_omits_independent_columns():
    rng = np.random.default_rng(seed=11)
    df = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200)})
    profiles = run_syntactic_profiling(df, max_cols=50)
    enriched = run_semantic_classification(profiles, df)
    insights = run_relational_analysis(df, enriched)
    assert not insights


# ---------- Layer 4 ----------

def test_layer4_kpis_returns_list():
    df = pd.DataFrame({"revenue": [100.0, 200.0, 300.0, 400.0, 500.0]})
    profiles = run_syntactic_profiling(df, max_cols=50)
    enriched = run_semantic_classification(profiles, df)
    insights = run_relational_analysis(df, enriched)
    kpis = determine_kpis(enriched, insights, top_k=5)
    assert isinstance(kpis, list)


def test_layer4_select_charts_returns_list_with_priority():
    df = pd.DataFrame({
        "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
        "cat": ["a", "b", "a", "b", "a"],
    })
    profiles = run_syntactic_profiling(df, max_cols=50)
    enriched = run_semantic_classification(profiles, df)
    insights = run_relational_analysis(df, enriched)
    charts = select_charts(enriched, insights, max_charts=10)
    assert isinstance(charts, list)
    if charts:
        chart_types = {c.get("chart_type") for c in charts}
        assert chart_types.issubset({"histogram", "bar", "line", "scatter"})
```

- [ ] **Step 2: Run the layer tests**

```bash
pytest tests/test_pipeline_layers.py -v
```

Expected: all pass. If one fails, the layer's public contract drifted from what the architecture doc claims — investigate the layer and either fix the test (if behavior is correct) or fix the layer (if behavior regressed).

- [ ] **Step 3: Run full suite with coverage**

```bash
pytest --cov --cov-report=term-missing -v
```

Expected: all tests pass. Note the coverage % printed at the end — Phase 0 target is **≥60% on `src/` and `main.py`**. If under 60%, scan the missing-lines report for cheap additions, but don't pad coverage with low-value tests.

- [ ] **Step 4: Commit**

```bash
git add tests/test_pipeline_layers.py
git commit -m "test: add per-layer unit tests for layers 1-4"
```

---

## Task 12: Structured stdout logging via structlog

**Files:**
- Create: `src/observability/__init__.py`
- Create: `src/observability/logging.py`
- Modify: `src/logger.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add `structlog` to `requirements.txt`**

Append:

```
structlog==24.1.0
```

Then install:

```bash
pip install -r requirements.txt
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_observability.py` (we'll extend this in later tasks):

```python
"""Observability layer — logging, request id, health, metrics, tracing."""
import json
import logging

import pytest

from src.observability import logging as obs_logging


def test_structlog_emits_json_to_stdout(capsys):
    obs_logging.configure_observability_logging(force=True)
    log = logging.getLogger("test.json.emit")
    log.info("hello", extra={"request_id": "r-1", "user": "alice"})
    captured = capsys.readouterr().out.strip().splitlines()
    assert captured, "expected at least one log line on stdout"
    parsed = json.loads(captured[-1])
    assert parsed["event"] == "hello" or parsed.get("message") == "hello"
    assert parsed["level"].lower() == "info"


def test_structlog_includes_request_id_from_contextvar(capsys):
    obs_logging.configure_observability_logging(force=True)
    token = obs_logging.request_id_var.set("ctx-req-99")
    try:
        log = logging.getLogger("test.ctx.req")
        log.info("contextual")
    finally:
        obs_logging.request_id_var.reset(token)
    captured = capsys.readouterr().out.strip().splitlines()
    parsed = json.loads(captured[-1])
    assert parsed.get("request_id") == "ctx-req-99"
```

- [ ] **Step 3: Run to verify it fails**

```bash
pytest tests/test_observability.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.observability'`.

- [ ] **Step 4: Create `src/observability/__init__.py`**

```python
"""Cross-cutting observability concerns: logging, tracing, metrics, health, errors."""
```

- [ ] **Step 5: Create `src/observability/logging.py`**

```python
"""Structlog configuration: JSON stdout logs with request-id propagation.

Existing call sites use stdlib `logging.getLogger(...)`. Structlog patches the
root handlers so those calls automatically render as JSON on stdout — no per-
module migration needed. The existing file-rotation handlers in `src/logger.py`
remain available behind the `LOG_FILE_HANDLERS=true` env flag for local dev.
"""
from __future__ import annotations

import logging
import os
import sys
from contextvars import ContextVar
from typing import Optional

import structlog

request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

_CONFIGURED = False


def _add_request_id(_, __, event_dict):
    rid = request_id_var.get()
    if rid is not None:
        event_dict["request_id"] = rid
    return event_dict


def configure_observability_logging(force: bool = False) -> None:
    """Idempotently configure structlog + stdlib logging.

    Args:
        force: bypass the sentinel — used by tests to reconfigure between cases.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        timestamper,
        _add_request_id,
    ]

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(level)

    root = logging.getLogger()
    root.handlers = [
        h for h in root.handlers
        if not isinstance(h, logging.StreamHandler) or h.stream is not sys.stdout
    ]
    root.addHandler(handler)
    root.setLevel(level)

    for noisy in ("uvicorn.access", "watchfiles", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
```

- [ ] **Step 6: Run the failing test again**

```bash
pytest tests/test_observability.py -v
```

Expected: both tests pass.

- [ ] **Step 7: Wire `configure_observability_logging` into `main.py`**

In `main.py`, replace the existing logging block:

```python
# ---------------- LOGGING ----------------
try:
    from src.logger import configure_logging, get_logger
    configure_logging()
    logger = get_logger(__name__)
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
```

with:

```python
# ---------------- LOGGING ----------------
from src.observability.logging import configure_observability_logging
configure_observability_logging()

if os.environ.get("LOG_FILE_HANDLERS", "false").lower() == "true":
    try:
        from src.logger import configure_logging
        configure_logging()
    except Exception:
        pass

logger = logging.getLogger(__name__)
```

- [ ] **Step 8: Run the full suite to verify no regressions**

```bash
pytest -v
```

Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add src/observability/__init__.py src/observability/logging.py main.py requirements.txt tests/test_observability.py
git commit -m "feat(observability): structured JSON stdout logging via structlog"
```

---

## Task 13: Request-ID middleware

**Files:**
- Create: `src/observability/request_id.py`
- Modify: `main.py`
- Modify: `tests/test_observability.py`

- [ ] **Step 1: Append to `tests/test_observability.py`**

```python
def test_request_id_header_round_trips(client):
    """X-Request-ID supplied by the caller is echoed in the response headers."""
    response = client.get("/api/dashboard", headers={"X-Request-ID": "trace-abc"})
    assert response.status_code == 200
    assert response.headers.get("x-request-id") == "trace-abc"


def test_request_id_is_generated_if_missing(client):
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    rid = response.headers.get("x-request-id")
    assert rid and len(rid) >= 8
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_observability.py::test_request_id_header_round_trips -v
```

Expected: FAIL — no `x-request-id` header in response.

- [ ] **Step 3: Create `src/observability/request_id.py`**

```python
"""ASGI middleware: ensure every request has a stable X-Request-ID."""
from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from src.observability.logging import request_id_var


class RequestIDMiddleware(BaseHTTPMiddleware):
    HEADER = "X-Request-ID"

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get(self.HEADER) or uuid.uuid4().hex
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers[self.HEADER] = rid
        return response
```

- [ ] **Step 4: Register the middleware in `main.py`**

Add the import near the top:

```python
from src.observability.request_id import RequestIDMiddleware
```

And register it **after** `app = FastAPI()` and **before** `CORSMiddleware`:

```python
app.add_middleware(RequestIDMiddleware)
```

(Middleware order in FastAPI is LIFO — last `add_middleware` runs first on the request path. Adding `RequestIDMiddleware` after `CORSMiddleware` means CORS sees the request first; we want request-id to be set before any other middleware logs anything, so register it **last** so it runs **first**:

```python
# These should be in this order — last-added runs first.
app.add_middleware(CORSMiddleware, ...)
app.add_middleware(RequestIDMiddleware)
```

Update Task 5's middleware block accordingly if needed.)

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest tests/test_observability.py -v
```

Expected: all pass.

- [ ] **Step 6: Run full suite**

```bash
pytest -v
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/observability/request_id.py main.py tests/test_observability.py
git commit -m "feat(observability): add X-Request-ID middleware with contextvar propagation"
```

---

## Task 14: `/healthz` and `/readyz` endpoints

**Files:**
- Create: `src/observability/health.py`
- Modify: `main.py`
- Modify: `tests/test_observability.py`

- [ ] **Step 1: Append to `tests/test_observability.py`**

```python
def test_healthz_returns_200(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_returns_200_in_phase_0(client):
    """Phase 0 readiness has no real deps; it returns ok unconditionally.
    Later phases add Postgres + Redis dependency checks here."""
    response = client.get("/readyz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert "checks" in body
```

- [ ] **Step 2: Run to verify fail**

```bash
pytest tests/test_observability.py::test_healthz_returns_200 -v
```

Expected: FAIL — 404.

- [ ] **Step 3: Create `src/observability/health.py`**

```python
"""Liveness (`/healthz`) and readiness (`/readyz`) endpoints.

`/healthz` should be cheap and never fail — it answers "is the process alive?"
`/readyz` should fail when a hard dependency is unreachable. In Phase 0 there
are no external deps yet, so it always returns ready. Future phases (Postgres,
Redis, S3) extend `_readiness_checks()`.
"""
from __future__ import annotations

from typing import Awaitable, Callable, Dict, List, Tuple

from fastapi import APIRouter, Response

ReadinessCheck = Callable[[], Awaitable[Tuple[bool, str]]]

_checks: List[Tuple[str, ReadinessCheck]] = []


def register_readiness_check(name: str, check: ReadinessCheck) -> None:
    _checks.append((name, check))


def build_router() -> APIRouter:
    router = APIRouter(tags=["observability"])

    @router.get("/healthz")
    async def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @router.get("/readyz")
    async def readyz(response: Response) -> Dict[str, object]:
        results = {}
        all_ok = True
        for name, check in _checks:
            try:
                ok, detail = await check()
            except Exception as exc:
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            results[name] = {"ok": ok, "detail": detail}
            all_ok = all_ok and ok
        if not all_ok:
            response.status_code = 503
            return {"status": "not_ready", "checks": results}
        return {"status": "ready", "checks": results}

    return router
```

- [ ] **Step 4: Mount the router in `main.py`**

Add the import:

```python
from src.observability.health import build_router as build_health_router
```

And right after `app = FastAPI()` and the middleware setup, but **before** the SPA catchall route at the bottom (catchall would eat `/healthz` otherwise):

```python
app.include_router(build_health_router())
```

Critical: this MUST be added before the `@app.get("/{full_path:path}")` SPA catchall. The catchall is currently near the bottom of `main.py`. Place the `include_router` call right after `app = FastAPI()`.

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_observability.py -v
```

Expected: pass.

- [ ] **Step 6: Manually probe the endpoints**

```bash
uvicorn main:app --port 8000 &
curl -s http://localhost:8000/healthz
curl -s http://localhost:8000/readyz
kill %1
```

Expected: `{"status": "ok"}` and `{"status": "ready", "checks": {}}` respectively.

- [ ] **Step 7: Commit**

```bash
git add src/observability/health.py main.py tests/test_observability.py
git commit -m "feat(observability): add /healthz and /readyz endpoints"
```

---

## Task 15: Prometheus `/metrics` endpoint

**Files:**
- Create: `src/observability/metrics.py`
- Modify: `main.py`
- Modify: `requirements.txt`
- Modify: `tests/test_observability.py`

- [ ] **Step 1: Add `prometheus-client` to `requirements.txt`**

```
prometheus-client==0.20.0
```

Then `pip install -r requirements.txt`.

- [ ] **Step 2: Append to `tests/test_observability.py`**

```python
def test_metrics_endpoint_returns_prometheus_format(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "# HELP" in body
    assert "http_requests_total" in body


def test_metrics_records_request(client):
    client.get("/api/dashboard")
    response = client.get("/metrics")
    assert 'http_requests_total{' in response.text
    assert 'path="/api/dashboard"' in response.text
```

- [ ] **Step 3: Run to verify fail**

```bash
pytest tests/test_observability.py::test_metrics_endpoint_returns_prometheus_format -v
```

Expected: FAIL — 404.

- [ ] **Step 4: Create `src/observability/metrics.py`**

```python
"""Prometheus metrics — request counters/histograms + a /metrics endpoint.

The pipeline-layer histogram (`pipeline_layer_seconds`) is consumed by
`src/core/pipeline.py` in Task 16.
"""
from __future__ import annotations

import time
from typing import Callable

from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

registry = CollectorRegistry()

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    labelnames=("method", "path", "status"),
    registry=registry,
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    labelnames=("method", "path"),
    registry=registry,
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

http_requests_in_flight = Gauge(
    "http_requests_in_flight",
    "Currently in-flight HTTP requests",
    registry=registry,
)

pipeline_layer_seconds = Histogram(
    "pipeline_layer_seconds",
    "Time spent in each pipeline layer",
    labelnames=("layer",),
    registry=registry,
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        path = request.url.path
        method = request.method
        http_requests_in_flight.inc()
        start = time.perf_counter()
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            http_requests_total.labels(method=method, path=path, status="500").inc()
            raise
        finally:
            http_requests_in_flight.dec()
            http_request_duration_seconds.labels(method=method, path=path).observe(
                time.perf_counter() - start
            )
        http_requests_total.labels(method=method, path=path, status=str(status)).inc()
        return response


def build_router() -> APIRouter:
    router = APIRouter(tags=["observability"])

    @router.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(
            content=generate_latest(registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    return router
```

- [ ] **Step 5: Register middleware and router in `main.py`**

Add imports:

```python
from src.observability.metrics import MetricsMiddleware, build_router as build_metrics_router
```

After the existing `app.include_router(build_health_router())` line, add:

```python
app.include_router(build_metrics_router())
app.add_middleware(MetricsMiddleware)
```

The middleware order (recall: LIFO) should now be `Request-ID → Metrics → CORS → handler`. Reorder the `add_middleware` calls so the final block reads:

```python
app.add_middleware(CORSMiddleware, ...)        # innermost (closest to handler)
app.add_middleware(MetricsMiddleware)           # measures including CORS overhead
app.add_middleware(RequestIDMiddleware)         # outermost — first to see request
```

- [ ] **Step 6: Run the tests**

```bash
pytest tests/test_observability.py -v
```

Expected: all pass.

- [ ] **Step 7: Run full suite**

```bash
pytest -v
```

Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add src/observability/metrics.py main.py requirements.txt tests/test_observability.py
git commit -m "feat(observability): expose Prometheus /metrics endpoint"
```

---

## Task 16: Per-pipeline-layer instrumentation

**Files:**
- Modify: `src/core/pipeline.py`
- Modify: `tests/test_observability.py`

- [ ] **Step 1: Append to `tests/test_observability.py`**

```python
def test_pipeline_records_layer_metrics(client, upload_files):
    from src.observability import metrics as obs_metrics

    obs_metrics.pipeline_layer_seconds.clear()
    client.post("/api/upload", files=upload_files)

    samples = list(obs_metrics.pipeline_layer_seconds.collect())[0].samples
    layer_counts = {
        s.labels["layer"]: s.value
        for s in samples
        if s.name.endswith("_count")
    }
    for expected_layer in ("profiling", "classifying", "relating", "eda", "interpreting", "rendering"):
        assert layer_counts.get(expected_layer, 0) >= 1, f"no observation for layer={expected_layer}"
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_observability.py::test_pipeline_records_layer_metrics -v
```

Expected: FAIL — no observations recorded.

- [ ] **Step 3: Instrument `src/core/pipeline.py`**

Open `src/core/pipeline.py`. Locate the sync pipeline orchestrator (`build_dashboard_from_df`). Around each layer call, wrap with the histogram timer.

Add the import at the top:

```python
from contextlib import contextmanager
from src.observability.metrics import pipeline_layer_seconds


@contextmanager
def _time_layer(layer_name: str):
    with pipeline_layer_seconds.labels(layer=layer_name).time():
        yield
```

Then wrap each layer call. The function should end up looking roughly like:

```python
def build_dashboard_from_df(df, max_cols=50, original_filename=None):
    df = _reshape_if_wide_timeseries(df)

    with _time_layer("profiling"):
        profiles = run_syntactic_profiling(df, max_cols=max_cols)

    with _time_layer("classifying"):
        enriched = run_semantic_classification(profiles, df)

    with _time_layer("relating"):
        insights = run_relational_analysis(df, enriched)

    with _time_layer("eda"):
        eda_summary = run_eda_analysis(df, enriched, insights)

    with _time_layer("interpreting"):
        kpis = determine_kpis(enriched, insights, top_k=KPI_TOP_K)
        chart_specs = select_charts(enriched, insights, max_charts=MAX_CHARTS)

    with _time_layer("rendering"):
        all_charts = build_charts_from_specs(df, chart_specs, ...)

    # ... (rest of assembly unchanged)
```

(Adjust to match the actual function — copy the existing call shape exactly, only wrap each call in a `with _time_layer(...)` block. Don't change argument order or names.)

Apply the same instrumentation inside the streaming generator (`build_dashboard_from_file_generator` / `build_dashboard_from_df_generator`) at the same call sites.

- [ ] **Step 4: Run the tests**

```bash
pytest tests/test_observability.py::test_pipeline_records_layer_metrics tests/core/test_pipeline.py -v
```

Expected: both pass.

- [ ] **Step 5: Run full suite**

```bash
pytest -v
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/core/pipeline.py tests/test_observability.py
git commit -m "feat(observability): record per-layer pipeline durations as Prometheus histogram"
```

---

## Task 17: OpenTelemetry tracing (no-op when unconfigured)

**Files:**
- Create: `src/observability/tracing.py`
- Modify: `main.py`
- Modify: `requirements.txt`
- Modify: `tests/test_observability.py`

- [ ] **Step 1: Add OTel deps to `requirements.txt`**

```
opentelemetry-api==1.24.0
opentelemetry-sdk==1.24.0
opentelemetry-instrumentation-fastapi==0.45b0
opentelemetry-exporter-otlp-proto-http==1.24.0
```

Then `pip install -r requirements.txt`.

- [ ] **Step 2: Append to `tests/test_observability.py`**

```python
def test_tracing_init_is_noop_when_endpoint_unset(monkeypatch):
    """Without OTEL_EXPORTER_OTLP_ENDPOINT, tracing init must not raise or set
    a real exporter."""
    from src.observability import tracing

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    tracing.configure_tracing(force=True)
    assert tracing.is_enabled() is False


def test_tracing_init_succeeds_with_endpoint(monkeypatch):
    from src.observability import tracing

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "ai-powered-dashboards-test")
    tracing.configure_tracing(force=True)
    assert tracing.is_enabled() is True
```

- [ ] **Step 3: Run to verify fail**

```bash
pytest tests/test_observability.py::test_tracing_init_is_noop_when_endpoint_unset -v
```

Expected: FAIL — module not found.

- [ ] **Step 4: Create `src/observability/tracing.py`**

```python
"""OpenTelemetry tracing — opt-in via OTEL_EXPORTER_OTLP_ENDPOINT.

When the endpoint env var is absent, tracing init is a no-op. This keeps the
current Hugging Face Space deployment working unchanged while letting any
managed deployment turn on traces with a single env var.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI

_enabled: bool = False
_configured: bool = False


def is_enabled() -> bool:
    return _enabled


def configure_tracing(app: Optional[FastAPI] = None, force: bool = False) -> None:
    global _enabled, _configured
    if _configured and not force:
        return
    _configured = True
    _enabled = False

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:
        return

    service_name = os.environ.get("OTEL_SERVICE_NAME", "ai-powered-dashboards")
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)

    if app is not None:
        FastAPIInstrumentor.instrument_app(app)

    _enabled = True
```

- [ ] **Step 5: Wire into `main.py`**

Add the import:

```python
from src.observability.tracing import configure_tracing
```

Right after `app = FastAPI()` (and after the health/metrics router inclusion is fine — order doesn't matter for tracing), add:

```python
configure_tracing(app)
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_observability.py -v
```

Expected: pass.

- [ ] **Step 7: Run full suite**

```bash
pytest -v
```

Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add src/observability/tracing.py main.py requirements.txt tests/test_observability.py
git commit -m "feat(observability): optional OpenTelemetry tracing via OTLP env var"
```

---

## Task 18: Sentry error reporting (no-op when unconfigured)

**Files:**
- Create: `src/observability/sentry.py`
- Modify: `main.py`
- Modify: `requirements.txt`
- Modify: `tests/test_observability.py`

- [ ] **Step 1: Add Sentry to `requirements.txt`**

```
sentry-sdk[fastapi]==1.45.0
```

Then `pip install -r requirements.txt`.

- [ ] **Step 2: Append to `tests/test_observability.py`**

```python
def test_sentry_init_is_noop_without_dsn(monkeypatch):
    from src.observability import sentry as obs_sentry

    monkeypatch.delenv("SENTRY_DSN", raising=False)
    obs_sentry.configure_sentry(force=True)
    assert obs_sentry.is_enabled() is False


def test_sentry_init_is_enabled_with_dsn(monkeypatch):
    from src.observability import sentry as obs_sentry

    monkeypatch.setenv("SENTRY_DSN", "https://public@o0.ingest.sentry.io/0")
    obs_sentry.configure_sentry(force=True)
    assert obs_sentry.is_enabled() is True
```

- [ ] **Step 3: Run to verify fail**

```bash
pytest tests/test_observability.py::test_sentry_init_is_noop_without_dsn -v
```

Expected: FAIL — module not found.

- [ ] **Step 4: Create `src/observability/sentry.py`**

```python
"""Sentry SDK init — opt-in via SENTRY_DSN."""
from __future__ import annotations

import os

_enabled: bool = False
_configured: bool = False


def is_enabled() -> bool:
    return _enabled


def configure_sentry(force: bool = False) -> None:
    global _enabled, _configured
    if _configured and not force:
        return
    _configured = True
    _enabled = False

    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        return

    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
    except ImportError:
        return

    environment = os.environ.get("SENTRY_ENVIRONMENT", "production")
    traces_sample_rate = float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.0"))

    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        traces_sample_rate=traces_sample_rate,
        integrations=[StarletteIntegration(), FastApiIntegration()],
    )
    _enabled = True
```

- [ ] **Step 5: Wire into `main.py`**

Add the import:

```python
from src.observability.sentry import configure_sentry
```

Just **before** the line `app = FastAPI()`, add:

```python
configure_sentry()
```

(Sentry init must happen before FastAPI is instantiated so the SDK's Starlette/FastAPI integrations hook in correctly.)

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_observability.py -v
```

Expected: pass.

- [ ] **Step 7: Run full suite**

```bash
pytest -v
```

Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add src/observability/sentry.py main.py requirements.txt tests/test_observability.py
git commit -m "feat(observability): optional Sentry error reporting via SENTRY_DSN"
```

---

## Task 19: `.env.example` and dependency lock check

**Files:**
- Create: `.env.example`

- [ ] **Step 1: Create `.env.example`**

```env
# AI-Powered Dashboards — environment variable reference.
# Copy to `.env` for local dev. Hugging Face Space variables/secrets mirror this.

# ─── Auth (Clerk) ──────────────────────────────────────────
CLERK_PUBLISHABLE_KEY=
# Backend falls back to this if CLERK_PUBLISHABLE_KEY is unset:
VITE_CLERK_PUBLISHABLE_KEY=

# ─── Pipeline tuning ───────────────────────────────────────
MAX_ROWS=500000
MAX_COLS=50
MAX_CATEGORIES=10
MAX_CHARTS=20
KPI_TOP_K=10
MIN_VARIABILITY_THRESHOLD=0.01
MIN_UNIQUE_RATIO=0.01
MAX_UNIQUE_RATIO=0.9
MIN_CORRELATION=0.5
MIN_VARIANCE=0.001
MEMORY_LIMIT_MB=1000
TIMEOUT_SECONDS=600

# ─── HTTP ──────────────────────────────────────────────────
CORS_ALLOW_ORIGINS=http://localhost:5173,http://localhost:8000

# ─── Observability ─────────────────────────────────────────
LOG_LEVEL=INFO
LOG_FILE_HANDLERS=false

# OpenTelemetry — set both to enable; leave blank to disable.
OTEL_EXPORTER_OTLP_ENDPOINT=
OTEL_SERVICE_NAME=ai-powered-dashboards

# Sentry — set DSN to enable; leave blank to disable.
SENTRY_DSN=
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=0.0

# ─── Kaggle ingestion (optional) ───────────────────────────
KAGGLE_USERNAME=
KAGGLE_KEY=
```

- [ ] **Step 2: Verify `.gitignore` ignores `.env`**

```bash
grep -n "^\.env$\|^\.env\b" .gitignore || echo "MISSING — add it"
```

If missing, append `.env` to `.gitignore` and commit that as a separate small change.

- [ ] **Step 3: Verify all new deps resolve as a set**

```bash
pip install --dry-run -r requirements.txt
```

Expected: pip prints "Would install …" with no resolver conflicts. If any conflict appears (typically between `opentelemetry-*` packages), pin the conflicting transitive to match.

- [ ] **Step 4: Commit**

```bash
git add .env.example
git commit -m "docs: add .env.example documenting every runtime env var"
```

---

## Task 20: Phase 0 close-out — coverage check and changelog

**Files:**
- Create: `CHANGELOG.md` (or append if it already exists)

- [ ] **Step 1: Run the full suite with coverage**

```bash
pytest --cov --cov-report=term-missing --cov-report=html -v
```

Expected: all tests pass. Open `htmlcov/index.html` to inspect uncovered lines. Target: **≥60% on `src/` + `main.py`**. If under, add 1-2 quick tests targeting the highest-value missing branches. Don't pad with trivial assertions.

- [ ] **Step 2: Confirm GitHub Actions CI is green**

Push the latest commits to the branch:

```bash
git push
```

Open the GitHub Actions tab in the browser. Confirm both `Backend tests` and `Frontend build` jobs succeed.

- [ ] **Step 3: Create or append to `CHANGELOG.md`**

```markdown
# Changelog

## [Unreleased] — Phase 0: Stabilize

### Added
- pytest-based test suite covering all HTTP endpoints, per-pipeline-layer behaviour, and observability surface.
- GitHub Actions CI: runs backend tests + frontend build on every PR and main push.
- Pydantic response models for `/api/upload`, `/api/upload/stream` final-event payload, `/api/validate_external`, `/api/load_external`, `/api/dashboard`.
- Observability:
  - `structlog` JSON stdout logging.
  - `X-Request-ID` middleware with `contextvars` propagation.
  - `/healthz` (liveness) and `/readyz` (readiness, extensible).
  - Prometheus `/metrics` endpoint with HTTP + per-pipeline-layer histograms.
  - Optional OpenTelemetry tracing (enable via `OTEL_EXPORTER_OTLP_ENDPOINT`).
  - Optional Sentry error reporting (enable via `SENTRY_DSN`).
- `.env.example` documenting every supported env var.
- `pyproject.toml` with pytest + coverage configuration.

### Changed
- `CORSMiddleware` is now registered (was imported but unused). Allowlist via `CORS_ALLOW_ORIGINS`.
- Layer 3 correlation threshold reads from `src.config.MIN_CORRELATION` (was hard-coded `0.5`).

### Removed
- `src/viz/eda_visualizer.py` — dead code (no Python callers).
- Duplicate `serve_dynamic_assets` route definition in `main.py`.
- Duplicate `import re` in `main.py`.
- Redundant `app.mount("/assets", StaticFiles(...))` — the explicit route handler covers the same surface.

### Deferred to later phases
See `updrage-plan.txt` §11 — issues 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 21, 22.
```

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: add Phase 0 changelog entry"
```

- [ ] **Step 5: Open a PR**

```bash
gh pr create --base main --head phase-0-stabilize \
  --title "Phase 0: Stabilize — tests, CI, observability, API hygiene" \
  --body "$(cat <<'EOF'
## Summary
Phase 0 of the enterprise-readiness upgrade roadmap. Establishes the safety net before any architectural change touches the codebase.

## What's in
- §11 issues resolved: 10, 11, 12, 13, 19, 20
- Observability foundation: structured logs, request id, healthz/readyz, /metrics, optional OTel + Sentry

## What's not in (deferred)
- §11 issues 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 21, 22 — addressed in later phases.

## Test plan
- [ ] CI green on this PR
- [ ] Coverage ≥60% on `src/` + `main.py`
- [ ] Manual smoke: `uvicorn main:app --port 8000`, hit `/healthz`, `/readyz`, `/metrics`, upload sample CSV via UI
- [ ] HF Space deploy: confirm no env-var change needed (all new env vars are optional)

EOF
)"
```

---

## Self-review

**Spec coverage:** every §11 issue in Phase 0's scope (10, 11, 12, 13, 19, 20) maps to a specific task. The observability work (12-18) is the "enterprise-ready" gap not in the original 22. ✓

**Placeholder scan:** no "TBD", "TODO later", "implement later", "fill in details", "add error handling", or "similar to Task N" placeholders. Every code block is runnable. ✓

**Type consistency:** `request_id_var` defined in Task 12 (`src/observability/logging.py`) is consumed unchanged in Task 13 (`src/observability/request_id.py`). `pipeline_layer_seconds` defined in Task 15 is consumed in Task 16. `LoadResult` usage in Task 9 includes a verification sub-step in case its actual constructor differs. The `EnrichedProfile` constructor in Task 7's test uses field names sourced from `src/analysis/data_structures.py` — if those drift, the test fails fast with a clear `TypeError`. ✓

**Risks the engineer should know about:**
1. `requirements.txt` adds 8 new deps. If `pip install --dry-run` (Task 19 Step 3) surfaces a transitive conflict, the most likely culprit is the `opentelemetry-*` set — pin all four to the same minor (`1.24.x` / `0.45b0`).
2. The existing `tests/core/test_pipeline.py` references `tests/fixtures/sample_data.csv` which currently doesn't exist; Task 1 Step 3 creates it. If the test was being collected and failing pre-Phase-0, this fixes it; if it was being silently skipped, this turns it on.
3. Middleware order is fragile. Task 15 Step 5 spells out the final order — don't reorder casually.
4. Sentry must be initialised **before** `app = FastAPI()`; Task 18 Step 5 enforces this.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-15-phase-0-stabilize.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints for review.

**Which approach?**
