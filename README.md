---
title: AI Powered Dashboards
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# AI Powered Dashboards

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![React 18](https://img.shields.io/badge/react-18-61dafb.svg)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest%20%2B%20vitest-0a9edc.svg)](tests/)

A FastAPI + React analytics platform that turns a raw tabular file (CSV / Excel /
Parquet / JSON / NDJSON — uploaded, URL-fetched, or pulled from Kaggle) into an
interactive Plotly dashboard with auto-selected KPIs, curated chart sections,
deterministic advanced statistics, machine-learning insights, and a
conversational *Ask-Your-Data* layer — where **every reported number is traceable
back to a deterministic computation, never invented by an LLM.**

The spine is a **Semantic Contract Layer**. Ingest is gated and cleaned, columns
are profiled and classified with confidence, an invariant critic vetoes
nonsensical roles, and a frozen per-dataset contract then governs which
aggregations, correlations, and charts are even *allowed*. Low-confidence schemas
halt for human-in-the-loop review before any analysis runs. The LLM proposes;
the backend computes; an output validator rejects any number whose provenance
doesn't resolve.

Everything beyond the core deterministic pipeline is **config-gated and degrades
gracefully**: no `GROQ_API_KEY` → AI layers turn off; no `REDIS_URL` → the queue
and caches run in-process; no `DATABASE_URL` → persistence falls back to SQLite;
no scikit-learn/statsmodels → the ML and stats blocks return `{"available":
false}` instead of failing. A bare `pip install + uvicorn` already runs the full
dashboard pipeline.

## What it does

- **Multi-format streaming ingest** — `POST /api/upload/stream` drives an
  SSE progress bar with phase labels (`preparing → ingest_gate → profiling →
  classifying → relating → eda → kpis → rendering → done`). CSV, Excel, Parquet,
  JSON, and NDJSON sit behind one format-detection + parser seam
  ([src/data/formats.py](src/data/formats.py), [src/data/parser.py](src/data/parser.py)),
  with a DuckDB-based delimiter/dialect sniffer and a pandas/Polars/DuckDB engine
  seam ([src/data/engine.py](src/data/engine.py)).
- **Semantic Contract Layer** ([src/contract/](src/contract)) — an ingest gate
  (currency/thousands coercion, sentinel→NA, null-row drop, Presidio/regex PII
  detection), a frozen `DatasetContract` with a canonical schema fingerprint, a
  role-aware router that decides what may be summed/correlated/collapsed, and an
  invariant critic that vetoes nonsensical classifications (unique-numeric-as-id,
  ratio summation, leakage, extreme dispersion).
- **Human-in-the-loop schema review** — datasets below the confidence bar
  (`AUTO_ACCEPT_CONFIDENCE`, per-field floor) halt before Layer 3. The user edits
  roles in an editable contract table; `PATCH /api/dashboard/{id}/registry`
  re-locks the contract (version + 1) and recomputes the dashboard **without
  re-running the LLM**.
- **4-layer deterministic analysis** ([src/analysis/](src/analysis)) —
  profiler → classifier → relational → interpreter, producing roles, correlations,
  KPIs, and chart specs from contract-allowed operations only.
- **Statistical depth** ([src/analysis/statistical_depth.py](src/analysis/statistical_depth.py))
  — Spearman / MI / Cramér's V / η, Mann-Kendall + STL trend tests,
  IsolationForest / LOF outliers, KMeans / HDBSCAN clustering, skew / kurtosis /
  normality. Seeded and row-capped for determinism; each block degrades to `{}`.
- **Machine-learning insights** ([src/analysis/ml/](src/analysis/ml)) — auto-target
  driver analysis (HistGradientBoosting vs. linear/logistic baseline, cross-validated,
  permutation importance, leakage guard), KMeans segmentation with silhouette
  search, IsolationForest anomalies, Holt-Winters forecasting, and a no-retrain
  **what-if predictor** that scores user-supplied feature values against a cached
  fitted model.
- **Provenance-validated AI** — the LLM narrates; it never produces figures. Every
  KPI carries a provenance token, and an output validator rejects any number whose
  provenance doesn't resolve, falling back to a logged heuristic. PII-bearing
  datasets block LLM egress entirely until the user grants explicit consent
  (`POST /api/dashboard/{id}/ai-consent`).
- **Ask-Your-Data** ([src/analysis/ask/](src/analysis/ask)) — `POST /api/ask` runs a
  bounded agent over a fixed, contract-guarded tool catalogue (`column_stat`,
  `aggregate`, `top_categories`, `correlation`, `filter`). The LLM plans steps
  (hard-capped by `ASK_MAX_ITERATIONS`), the backend executes them deterministically,
  and narration may use *only* the computed numbers.
- **Interactive dashboard (no LLM, no WASM)** — `POST /api/interact` runs
  structured, deterministic interactions (filter / aggregate / predict / re-segment)
  reusing the Ask tool catalogue, with a chained provenance token and an LRU result
  cache keyed on `sha256(fingerprint + canonical spec)`. The frontend also applies
  pure client-side filters/highlights ([frontend/src/lib/clientFilter.js](frontend/src/lib/clientFilter.js))
  without a round-trip where the data is already shipped.
- **Adaptive charting** ([src/viz/](src/viz)) — scatter >10k points → density;
  long time series → span-aware resample; skewed Y axes → log scale; value labels
  only on ≤12 bars; mean-reference lines on category comparisons and least-squares
  trendlines on series. Charts are sectioned (Breakdowns / Distributions / Trends /
  Relationships) and rank-curated.
- **Async at scale** ([src/jobs/](src/jobs)) — `POST /api/jobs/upload` runs large
  uploads on an Arq/Redis queue (idempotent on data hash), reusing the same SSE
  event shape; degrades to an in-process asyncio task when no worker/Redis is
  configured. Auth happens at submit time so long pipelines hold no live token.
- **Accounts & history** — Clerk auth with org multi-tenancy and optional signed
  guest sessions; analyses are persisted and re-openable per owner
  (`GET /api/history`, scoped org > user > guest).
- **Server-side PDF export** — `GET /api/dashboard/export.pdf` renders via ReportLab
  ([src/reporting/pdf_report.py](src/reporting/pdf_report.py)) instead of a flaky
  browser screenshot.

## Run it

### Local

```bash
# 1. Python backend
python -m venv venv
venv\Scripts\activate              # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt

# 2. Frontend (one-time build OR dev server with HMR)
cd frontend
npm ci
npm run build                       # writes frontend/dist/ served by FastAPI
# -- or, for hot reload during development:
npm run dev                         # Vite dev server on :5173, proxies /api to the backend

# 3. Backend (serves the built SPA + API)
cd ..
python -m uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

Open <http://localhost:7860>. See [docs/LOCAL_DEV.md](docs/LOCAL_DEV.md) for the
two local modes (in-process default vs. production-correct out-of-process Arq +
Redis), env vars, and optional Postgres/PII extras.

### Docker

```bash
docker build -t ai-powered-dashboards .
docker run -p 7860:7860 ai-powered-dashboards
```

Single-stage image on `python:3.9-slim` with Node 18 for the frontend build,
running as a non-root user on port 7860. `docker/entrypoint.sh` launches uvicorn
and — when `JOB_QUEUE_ENABLED=true` and `RUN_WORKER` is not `false` — an Arq
worker alongside it.

### Full async stack

[docker-compose.yml](docker-compose.yml) brings up the production topology:
**api** + a separate **worker** container (`RUN_WORKER=false` on the api,
dedicated worker service), **redis** (cache + Arq broker), and **postgres**
(shared dashboard DB so the worker's results are visible to the web process),
with a shared spool volume for file handoff and healthchecks gating startup.

### Deployment

- **Hugging Face Spaces** (live demo) — the `hf` remote auto-deploys via the Docker
  SDK (frontmatter at the top of this file). Push with `git push hf main`. HF runs
  the **single-container degraded profile** (`JOB_QUEUE_ENABLED=false`, in-process
  queue/caches, no external Redis/Postgres). `VITE_CLERK_PUBLISHABLE_KEY` must be
  set at build time (Vite inlines it); secrets map to HF Space variables.
- **Full async stack** (out-of-process worker + Redis + Postgres) — HF can't host
  the worker/Redis/DB, so the full target is documented in
  [docs/DEPLOY-GCP.md](docs/DEPLOY-GCP.md) (Google Cloud Always-Free `e2-micro`).

## API

JSON endpoints under `/api` (line numbers reference [main.py](main.py)):

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/upload` | Blocking upload — returns the final dashboard payload. |
| POST | `/api/upload/stream` | **Streaming upload (SSE).** Emits `{phase, message, percent}`; final event carries the dashboard state. |
| POST | `/api/jobs/upload` | **Async upload.** Returns a `job_id` (202); idempotent on data hash. |
| GET | `/api/jobs/{id}` · `/api/jobs/{id}/events` | Job status / SSE event stream. |
| POST | `/api/jobs/{id}/cancel` | Cancel an in-flight job. |
| POST | `/api/load_external` · `/api/validate_external` | Load / pre-validate a CSV by URL or Kaggle slug (`username/dataset`). |
| GET | `/api/dashboard` | Fetch the session's current dashboard payload. |
| GET | `/api/dashboard/export.pdf` | Server-side ReportLab PDF export of the current dashboard. |
| PATCH | `/api/dashboard/{id}/registry` | Submit a HITL schema-review override → re-lock contract → recompute. |
| POST | `/api/dashboard/{id}/ai-consent` | Grant consent to run the AI analyst on a PII-bearing dataset. |
| POST | `/api/ask` | **Ask-Your-Data** — bounded, provenance-tracked Q&A. |
| POST | `/api/interact` | Contract-guarded interactive filtering / aggregate / predict / re-segment (no LLM). |
| GET | `/api/history` · `/api/history/{trace_id}` | List / reopen the owner's past analyses. |

Operational endpoints: `/healthz`, `/readyz`, `/metrics` (Prometheus).

The SSE stream uses plain `data: {json}\n\n` frames. Example client (see
[frontend/src/services/api.js](frontend/src/services/api.js)):

```js
const jobId = await submitUploadJob(file);
await streamJobEvents(jobId, (evt) => {
  console.log(evt.phase, evt.percent, evt.message);
});
```

## Architecture

```
.
├── main.py                              FastAPI app, endpoints, SSE wiring, auth/rate-limit
├── src/
│   ├── config.py                        Central config + feature flags (graceful degradation)
│   ├── auth.py                          Clerk verification, owner-key scoping, signed guest ids
│   ├── core/
│   │   ├── pipeline.py                  Contract-gated orchestrator (sync + SSE generator)
│   │   └── state_payload.py             DashboardState dataclass
│   ├── contract/                        Semantic Contract Layer
│   │   ├── ingest_gate.py               Cleaning + PII detection (regex tier; Presidio optional)
│   │   ├── compiler.py                  Frozen contract + schema fingerprint
│   │   ├── role_router.py               What may be summed / correlated / collapsed
│   │   ├── invariant_critic.py          Vetoes nonsensical classifications
│   │   ├── dq_report.py                 Data-quality verdict + auto-accept criteria
│   │   ├── registry_patch.py            HITL override → re-lock → recompute (no LLM)
│   │   ├── ai_consent.py                PII consent unlock for the AI layer
│   │   ├── df_cache.py                  Transient + durable (Parquet) cleaned-frame cache
│   │   └── cache.py / rebuild.py / models.py
│   ├── analysis/
│   │   ├── layer_1..4_*.py              Profiler → classifier → relational → interpreter
│   │   ├── statistical_depth.py         Deterministic advanced statistics
│   │   ├── eda_analyzer.py              Indicators, outliers, use cases, recommendations
│   │   ├── llm_analyst.py               Provenance-validated AI narration
│   │   ├── llm/                         Provider-agnostic LLM (interface + Groq + response cache)
│   │   ├── ask/                         Bounded Ask-Your-Data agent + interact engine + tools
│   │   └── ml/                          Supervised drivers · segments · anomalies · forecast · what-if + model cache
│   ├── data/
│   │   ├── formats.py                   CSV/Excel/Parquet/JSON detection + normalization
│   │   ├── parser.py                    SSRF-hardened file/URL/Kaggle loaders + encoding sniff
│   │   └── engine.py                    DataFrame engine seam (pandas · Polars · DuckDB)
│   ├── jobs/                            Arq queue · runner · worker · store (Redis or in-process)
│   ├── persistence/                     SQLAlchemy models · repository · Redis cache · history
│   ├── observability/                   OTel tracing · Prometheus metrics · Sentry · health · request-id · logging
│   ├── reporting/pdf_report.py          ReportLab dashboard PDF
│   ├── viz/                             Plotly renderer + per-chart builders + sectioning
│   └── diagnostics/tracer.py            Per-request pipeline tracing
├── observability/                       SLOs, Prometheus alerts, Grafana board, Locust load test
├── alembic/                             DB migrations
├── docker/entrypoint.sh                 web + optional Arq worker
├── docker-compose.yml                   api · worker · redis · postgres (full async stack)
├── docs/                                LOCAL_DEV · DEPLOY-GCP · DEPENDENCY_POLICY · research-framing
├── tests/                               analysis · contract · core · data · eval (golden datasets) + API/integration
└── frontend/
    ├── package.json                     React 18 · Vite · Tailwind + DaisyUI · Clerk · plotly.js-basic
    └── src/
        ├── App.jsx                      Routes + ErrorBoundary
        ├── dashboardStore.js            Zustand store (data · theme · interactions · schema review · AI consent)
        ├── services/api.js              axios client + SSE jobs + ask/interact/history/export
        ├── lib/clientFilter.js          Pure client-side filter / highlight primitives
        └── components/
            ├── charts/                  ChartRenderer · LazyMount (IntersectionObserver) · ChartModal
            └── dashboard/               DataQuality · Overview · EDA · Visualizations · Predictions (what-if) · Columns
```

## Pipeline

`build_dashboard_from_df` (sync) and `build_dashboard_from_df_generator` (SSE)
in [src/core/pipeline.py](src/core/pipeline.py) chain the same deterministic stages:

`ingest gate → L1 profiling → L2 classification → contract compile + critique +
cache → [schema-review gate] → L3 relational → EDA (+ statistical depth + ML
insights, each defensive) → L4 KPIs/charts → AI narration (optional, PII-gated)
→ Plotly render → DashboardState`.

A high-confidence, non-PII dataset auto-accepts and runs straight through; a
low-confidence one halts at the schema-review gate and returns a review payload
for the HITL flow.

## Stack

- **Backend**: Python 3.9 (runtime) · FastAPI 0.109 · uvicorn 0.27 · pandas 2.2 ·
  pyarrow 21 · openpyxl 3.1 · DuckDB <1.5 (3.9-compatible) · plotly 5.18 · scipy
  1.12 · scikit-learn 1.6 · statsmodels 0.14 · SQLAlchemy 2 + Alembic + psycopg2
  (Postgres) · Redis 4.6 + Arq 0.26 (job queue) · Groq SDK · PyJWT (Clerk) ·
  slowapi (rate limiting) · structlog · OpenTelemetry · prometheus-client ·
  sentry-sdk · reportlab.
  *(CI/dev run on Python 3.12; optional PII tier via
  [requirements-pii.txt](requirements-pii.txt) — Presidio/spaCy needs ≥3.10, so
  the default image uses the regex PII tier. Exact pins in
  [requirements.lock.txt](requirements.lock.txt).)*
- **Frontend**: React 18 · Vite 4 · Tailwind CSS 3 + DaisyUI · react-plotly.js over
  `plotly.js-basic-dist` (bar/pie/scatter + derived line/histogram/box/heatmap) ·
  Clerk · axios · zustand · react-router-dom · Vitest + Testing Library +
  vitest-axe + Playwright.
- **Container**: Single-stage Docker on `python:3.9-slim` with Node 18 for the
  frontend build; `docker-compose` for the full api/worker/redis/postgres topology.

## Testing & CI

- **Local gate**: `pytest` — contract property tests, frozen snapshot mappings,
  backward-compat, pipeline wiring, API/integration, and a golden-dataset AI eval
  (`tests/eval/`, must pass 100%). This is the merge gate today.
- **Frontend**: `npm test` (Vitest + Testing Library + axe), `npm run test:e2e`
  (Playwright smoke), `npm run lint` (ESLint, zero-warnings), `npm run build`.
- **CI**: [.github/workflows/ci.yml](.github/workflows/ci.yml) runs backend
  pytest + AI-eval and frontend lint + test + build, every job gated on
  `vars.CI_ENABLED == 'true'` (a no-op until billing is armed — flip one repo
  variable to enable). Dependabot + an SBOM script + a quarterly React/Vite
  major-bump policy ([docs/DEPENDENCY_POLICY.md](docs/DEPENDENCY_POLICY.md)).
- **Reliability**: [observability/](observability) holds five SLOs (availability,
  fast-path/submit latency, analysis success, pipeline duration) with error
  budgets, multi-window burn-rate Prometheus alerts, a Grafana board, and a Locust
  load test.

## Contributing

1. Fork, branch (`git checkout -b feat/short-description`).
2. Run `pytest` locally — it's the gate.
3. Commit with a focused message (one logical change); push and open a PR.

## License

Released under the [MIT License](LICENSE). © 2026 Farhaan Qazi.
