---
title: ML Dashboard Generator
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# ML Dashboard Generator

A FastAPI + React analytics platform that ingests tabular data (CSV / Excel /
Parquet / JSON, uploaded, URL-fetched, or pulled from Kaggle), runs it through a
**contract-governed analysis pipeline**, and renders an interactive Plotly
dashboard with auto-selected KPIs, curated chart sections, statistical depth, and
a conversational *Ask-Your-Data* layer — where **every reported number is
traceable back to a deterministic computation**.

It is no longer a one-shot CSV → chart tool. The spine is a **Semantic Contract
Layer**: ingest is gated and cleaned, columns are classified with confidence and
arbitrated by an invariant critic, a frozen per-dataset contract governs which
aggregations and charts are even *allowed*, and the LLM is constrained to narrate
validated ground-truth numbers (never to invent them). Low-confidence schemas
halt for human-in-the-loop review before any analysis runs.

## Project status

| | |
|---|---|
| **Maturity** | Production-hardened; security, observability, CI gate, and contract test net in place |
| **Roadmap** | Phases 0–13 ✅ complete · Phase 14 (server-side interactive dashboard) 🚧 in progress |
| **Source of truth** | [UPGRADE-PLAN.md](UPGRADE-PLAN.md) — dated, per-step status |

Delivered so far: security hardening (Phase 0), the 8-phase Semantic Contract
Layer + HITL schema review (Phases 1–8), provider-agnostic AI + statistical depth
+ an AI eval harness (Phase 9), async job queue + multi-format ingest + per-user
history (Phase 10), Ask-Your-Data (Phase 11), reliability/observability + frontend
test net + a CI/CD merge gate (Phase 12), and dashboard composition/curation
(Phase 13). Phase 14 wires a no-WASM, server-side interactive-filter engine into
the live charts.

## What it does

- **Multi-format streaming ingest** ([`POST /api/upload/stream`](main.py#L380)) —
  SSE-driven progress bar with phase labels (Reading → Profiling → Classifying →
  Relating → EDA → KPIs → Rendering → Done). Accepts CSV, Excel, Parquet, and
  NDJSON behind one parser seam.
- **Semantic Contract Layer** ([src/contract](src/contract)) — ingest gate
  (currency/thousands coercion, sentinel→NA, PII detection), a frozen
  `DatasetContract` with schema fingerprint + grain detection, a role-aware router
  that decides what may be summed/correlated, and an invariant critic that vetoes
  nonsensical classifications (e.g. unique-numeric-as-identifier, ratio summation).
- **Human-in-the-loop schema review** — datasets below the confidence threshold
  halt before analysis; the user edits roles in an editable contract table, which
  re-locks the contract and recomputes the dashboard without re-running the LLM.
- **Provenance-validated AI** — the LLM proposes; the backend computes. Every KPI
  carries a provenance token, and an output validator rejects any number whose
  provenance doesn't resolve, falling back to a logged heuristic.
- **Statistical depth** ([`src/analysis/statistical_depth.py`](src/analysis/statistical_depth.py))
  — Spearman/MI/Cramér's V/η, Mann-Kendall + STL trend tests, IsolationForest/LOF
  outliers, KMeans/HDBSCAN clustering, normality, and RandomForest driver analysis;
  each block degrades to `{}` rather than failing.
- **Ask-Your-Data** ([`POST /api/ask`](main.py#L662)) — conversational follow-up
  over a fixed, contract-guarded tool catalogue (`column_stat`, `aggregate`,
  `top_categories`, `correlation`, `filter_count`); the LLM plans steps, the
  backend executes them deterministically, and narration may use *only* the
  computed numbers.
- **Adaptive charting** — scatter >10k points → 2D-histogram density; time series
  >10k rows → span-aware resample (`W`/`D`/`H`/`5min`/`30s`); skewed Y axes →
  log scale when `max/median > 100`. Charts are sectioned (Distributions /
  Breakdowns / Trends / Relationships) and rank-curated.
- **Async at scale** ([`POST /api/jobs/upload`](main.py#L858)) — large uploads run
  on an Arq/Redis job queue (idempotent on data hash), reusing the same SSE event
  shape; degrades to an in-process task when no worker/Redis is configured.
- **Accounts & history** — Clerk auth with org multi-tenancy; analyses are
  persisted and re-openable per owner ([`GET /api/history`](main.py#L637)).
- **React SPA** ([frontend/](frontend)) — Vite + Tailwind + react-plotly.js, with
  Data Quality / Overview / EDA / Visualizations / Columns tabs, an error
  boundary, and a Vitest + Playwright + axe test net.

## Run it

### Local

```bash
# 1. Python backend
python -m venv venv
venv\Scripts\activate              # Windows  (use: source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

# 2. Frontend (one-time build OR dev server with HMR)
cd frontend
npm ci
npm run build                       # writes frontend/dist/ served by FastAPI
# -- or, for hot reload during development:
npm run dev                         # Vite dev server on :5173, proxies /api to :7860

# 3. Backend (serves the built SPA + API)
cd ..
python -m uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

Open <http://localhost:7860>. See [docs/LOCAL_DEV.md](docs/LOCAL_DEV.md) for the
full dev setup (env vars, optional Redis/Postgres, PII extras).

Everything beyond the core pipeline is **config-gated and degrades gracefully**:
with no `GROQ_API_KEY` the AI layers turn off, with no `REDIS_URL` the queue runs
in-process, with no `DATABASE_URL` persistence falls back. So a bare
`pip install + uvicorn` already runs the full dashboard pipeline.

### Docker

```bash
docker build -t ml-dashboard .
docker run -p 7860:7860 ml-dashboard
```

Single-stage image on `python:3.9-slim` with Node 18 for the frontend build; the
`docker/entrypoint.sh` optionally launches the Arq worker alongside the web
process when `JOB_QUEUE_ENABLED=true`.

### Deployment

- **Hugging Face Spaces** (live demo) — `main` on the `hf` remote auto-deploys via
  the Docker SDK (frontmatter at the top of this file). Push with `git push hf main`.
  HF runs the **single-container degraded profile** (in-process queue, no external
  Redis/Postgres worker).
- **Full async stack** (out-of-process worker + Redis + Postgres) — HF Spaces can't
  host the worker/Redis/DB, so the full-stack target is documented in
  [docs/DEPLOY-ORACLE.md](docs/DEPLOY-ORACLE.md) (Oracle Always Free).

## API

JSON endpoints under `/api`:

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/upload` | Blocking upload — returns the final dashboard payload. |
| POST | `/api/upload/stream` | **Streaming upload (SSE).** Emits `{phase, message, percent}`; final event carries `trace_id` + payload. |
| POST | `/api/jobs/upload` | **Async upload.** Returns a `job_id`; idempotent on data hash. |
| GET | `/api/jobs/{id}` · `/api/jobs/{id}/events` | Job status / SSE event stream. |
| POST | `/api/jobs/{id}/cancel` | Cancel an in-flight job. |
| POST | `/api/load_external` · `/api/validate_external` | Load / pre-validate a CSV by URL or Kaggle slug (`username/dataset`). |
| GET | `/api/dashboard` | Fetch a dashboard payload by `trace_id` (or `most_recent`). |
| PATCH | `/api/dashboard/{id}/registry` | Submit a HITL schema-review override → re-lock contract → recompute. |
| POST | `/api/dashboard/{id}/ai-consent` | Grant/record consent for the AI analyst on a dataset. |
| POST | `/api/ask` | **Ask-Your-Data** — deterministic, provenance-tracked Q&A. |
| POST | `/api/interact` | Contract-guarded interactive filtering (no LLM). |
| GET | `/api/history` · `/api/history/{trace_id}` | List / reopen the owner's past analyses. |

Operational endpoints: `/healthz`, `/readyz`, `/metrics` (Prometheus).

The SSE stream format is plain `data: {json}\n\n` frames. Example client (see
[`frontend/src/services/api.js`](frontend/src/services/api.js)):

```js
await uploadFileStream(file, (evt) => {
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
│   ├── core/pipeline.py                 Contract-gated orchestrator (sync + SSE generator)
│   ├── contract/                        Semantic Contract Layer
│   │   ├── ingest_gate.py               Cleaning + PII detection (regex tier; Presidio optional)
│   │   ├── compiler.py                  Schema fingerprint, grain, agg/chart allow-lists
│   │   ├── role_router.py               What may be summed / correlated / collapsed
│   │   ├── invariant_critic.py          Vetoes nonsensical classifications
│   │   ├── registry_patch.py / rebuild.py   HITL override → re-lock → recompute (no LLM)
│   │   ├── df_cache.py                  Transient + durable (Parquet) cleaned-frame cache
│   │   └── cache.py / dq_report.py / models.py
│   ├── analysis/
│   │   ├── layer_1..4_*.py              Profiler → classifier → relational → interpreter
│   │   ├── statistical_depth.py         Deterministic advanced statistics
│   │   ├── eda_analyzer.py              Indicators, outliers, use cases, recommendations
│   │   ├── llm_analyst.py               Provenance-validated AI narration
│   │   └── llm/                         Provider-agnostic LLM (interface + Groq + response cache)
│   ├── data/
│   │   ├── parser.py                    SSRF-hardened CSV/URL/Kaggle loaders + encoding sniff
│   │   ├── formats.py                   CSV/Excel/Parquet/JSON detection + normalization
│   │   └── engine.py                    DataFrameEngine seam (pandas · Polars · DuckDB)
│   ├── jobs/                            Arq queue · runner · store (Redis or in-process)
│   ├── persistence/                     SQLAlchemy models · repository · history
│   ├── observability/                   OTel tracing · Prometheus metrics · Sentry · health · logging
│   ├── viz/                             Plotly renderer + per-chart builders + sectioning
│   └── diagnostics/tracer.py            Per-request pipeline tracing
├── observability/                       SLOs, Prometheus alerts, Grafana board, Locust load test
├── alembic/                             DB migrations
├── docker/                              entrypoint.sh (web + optional worker)
├── docs/                                LOCAL_DEV · DEPLOY-ORACLE · DEPENDENCY_POLICY
├── tests/                               analysis · contract · core · data · eval (golden datasets)
└── frontend/
    ├── package.json                     React 18 · Vite · Tailwind · Clerk · plotly.js-basic
    └── src/
        ├── App.jsx                      Routes + ErrorBoundary
        ├── dashboardStore.js            Zustand store (incl. interaction slice)
        ├── services/api.js              axios client + SSE upload + ask/interact/history
        ├── lib/clientFilter.js          Pure client-side filter primitives (Phase 14)
        ├── components/
        │   ├── charts/ChartRenderer.jsx Type-aware Plotly wrapper
        │   └── dashboard/               DataQuality / Overview / EDA / Visualizations / Columns tabs
        └── test/                        Vitest + RTL + axe
```

## Stack

- **Backend**: Python 3.9 (runtime) · FastAPI 0.109 · pandas 2.2 · plotly 5.18 ·
  scipy 1.12 · scikit-learn 1.6 · statsmodels 0.14 · SQLAlchemy 2 + Alembic
  (Postgres) · Redis + Arq (job queue) · Groq SDK · slowapi (rate limiting) ·
  structlog · OpenTelemetry · Prometheus · Sentry · uvicorn (standard).
  *(CI/dev run on Python 3.12; PII via optional [requirements-pii.txt](requirements-pii.txt).)*
- **Frontend**: React 18 · Vite 4 · Tailwind CSS 3 · react-plotly.js (over
  plotly.js-basic-dist) · Clerk · axios · zustand · react-router-dom · Vitest +
  Playwright + vitest-axe.
- **Container**: Single-stage Docker on `python:3.9-slim` with Node 18 for the
  frontend build.

## Testing & CI

- **Local gate**: `pytest` (contract property tests, frozen snapshot mappings,
  backward-compat, golden-dataset AI eval) — this is the merge gate today.
- **Frontend**: `npm test` (Vitest + RTL + axe), `npm run test:e2e` (Playwright smoke).
- **CI**: [.github/workflows/ci.yml](.github/workflows/ci.yml) runs lint + pytest +
  AI-eval + frontend build, every job gated on `vars.CI_ENABLED == 'true'` (a no-op
  until billing is armed — flip one repo variable to enable). Dependabot + an SBOM
  script + a quarterly React/Vite major-bump policy ([docs/DEPENDENCY_POLICY.md](docs/DEPENDENCY_POLICY.md)).

## Contributing

1. Fork, branch (`git checkout -b feat/short-description`).
2. Run `pytest` locally — it's the gate.
3. Commit with a focused message (one logical change); push and open a PR.

## License

MIT.
