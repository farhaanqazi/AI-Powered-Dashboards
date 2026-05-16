# AI Powered Dashboards — Strategic Upgrade Plan (2030-Ready)

> Goal: industry-standard, production-grade, and still relevant/usable in 2030.
> Grounded against the working tree (FastAPI + SQLAlchemy/Alembic + Redis +
> OTel/Prometheus/Sentry + Groq LLM; React 18 + Vite 4 + Clerk + Plotly).
> This is a roadmap, not a rewrite — most infra primitives already exist.

---

## 0. Honest current-state assessment

| Area | Today | 2030 risk |
|---|---|---|
| Analytics core | Pearson + IQR + regex heuristics | **High** — shallow; "advanced algorithms" is oversold |
| LLM layer | Hardwired to Groq (`groq==1.0.0`) | **High** — model lock-in; SOTA model in 2030 ≠ Groq |
| Data | CSV only, pandas in-process, on request thread | **High** — won't scale; no Parquet/Excel/large data |
| Ingestion security | Arbitrary URL + Kaggle fetch | **Critical** — SSRF / unbounded fetch surface |
| Reliability infra | OTel/Prom/Sentry/structlog present | Medium — wired but no SLOs/alerts/load tests |
| Persistence | SQLAlchemy + Redis cache (Phase 1 done) | Low — solid foundation |
| Frontend | React 18 / Vite 4, zero component tests | Medium — framework drift + no test safety net |
| Delivery | CI billing-locked, manual `git push hf main` | Medium — no automated gate, no eval harness |

Two issues are **not optional** to fix: (1) **SSRF** in URL/Kaggle ingestion,
(2) **LLM provider lock-in**. Everything else is prioritized below.

---

## 1. Architectural principles (the things that keep it relevant)

1. **Provider-agnostic AI.** No file imports a vendor SDK directly. One
   `LLMProvider` interface; Groq is one impl. Swapping to the 2030 SOTA model
   is a config change, not a refactor.
2. **Deterministic numbers, narrated by AI.** Keep the existing guarantee
   (`llm_analyst.py`): every figure is ground-truth from the stats layers; the
   model only selects and explains. This is the trust moat — extend it, never
   weaken it.
3. **Analysis is a job, not a request.** Heavy work moves off the HTTP thread
   onto a queue. The request returns a job id; results stream/poll.
4. **Engines behind interfaces.** pandas today, Polars/DuckDB tomorrow, behind
   a `DataFrameEngine` seam so a 2030 swap doesn't touch analysis code.
5. **Everything testable offline.** Local `pytest` stays the gate (CI is
   billing-locked); add a golden-dataset eval harness for the AI layer.

---

## 2. Phased roadmap

### P0 — Security & correctness (do first, ~1 sprint)

- **SSRF hardening** (`src/data/parser.py`): allowlist schemes, block private/
  link-local/metadata IP ranges (169.254.x, 10/172/192, ::1), enforce
  max-size + timeout + redirect cap, content-type sniff before parse.
- **Upload limits**: hard caps on file size, row count, column count; reject
  on stream, not after load. Surface clear errors (frontend already has the
  validation path via `sniffCsvFile`).
- **PII detection pass**: flag/optionally redact emails, phone, national-id
  patterns in profiling so the LLM prompt never ships raw PII to a 3rd party.
- **Endpoint auth audit**: confirm every `/api/*` route enforces Clerk JWT or
  explicit guest path (`src/auth.py`); add per-IP/user rate limiting.
- **Guest-state cleanup**: clear `dataInsight:guestMode` on Clerk sign-in
  (the stale-flag issue already diagnosed this session).

### P1.0 — Pipeline Correctness Hardening (AI-augmented classification)

> **Why before P1 depth:** the deterministic layers feed everything. A
> misclassified column → wrong role → wrong KPIs → wrong charts, *regardless
> of how advanced the P1 algorithms are*. Adding Spearman/IsolationForest on
> top of a classifier that thinks `revenue` is an identifier just produces
> sophisticated garbage. Correctness gates the differentiator.

**Architectural guardrail (extends the `llm_analyst.py` guarantee):** the LLM
may decide **roles, semantics, intent, and descriptions** (metadata only). It
**never produces a number and never mutates a data value**. Item #6 is the
sharp edge: AI decides wide-vs-long *intent*, but the reshape stays a guarded,
surfaced transform — never silent. Every AI metadata decision has a
deterministic heuristic fallback and is validated against a whitelist.

Correctness-critical first (1 → 2 → 7 → 6), then narration/additive (3,4,5,8):

| # | Target (file:line) | Fix | Impact | Risk |
|---|---|---|---|---|
| 1 | `layer_2_classifier.py:222` `run_semantic_classification` (role cascade 232–283) | AI assigns column role/tags from sampled values + name; validate against role whitelist; heuristic cascade as fallback | **Highest** | Med |
| 2 | `utils/identifier_detector.py:9` `is_likely_identifier_with_confidence` (hard 0.65/0.95/0.98) | AI adjudicates ambiguous identifier calls; keep thresholds as fast-path, AI only on the uncertain band | High | Med |
| 7 | `layer_2_classifier.py:76` `_detect_aggregation_semantics` + `_detect_dataset_grain` | AI decides sum-vs-mean / additive-vs-rate from semantics; wrong agg → wrong chart values | Med | Med |
| 6 | `pipeline.py` `_reshape_if_wide_timeseries` | AI decides wide-vs-long *intent*; reshape becomes **guarded + surfaced to user**, never silent (§11.6) | Med | **High — mutates data** |
| 3 | `eda_analyzer.py:105` `_generate_key_indicators` | Replace templated "average of X is N" with AI narration; numbers stay ground-truth | Med | Low |
| 4 | `eda_analyzer.py:168` `_identify_patterns_and_relationships` | AI explains *why* each outlier/anomaly/trend matters (pairs with P1 trend compute) | Med | Low |
| 5 | `ColumnsTab` + new `column_descriptions` | AI writes per-column plain-English business description (data dictionary; feeds P3 semantic layer) | Med | Low — additive |
| 8 | `main.py` exception handlers / `data/parser.py` | AI turns raw 500/traceback into a friendly failure explanation | Low | Low |

- Each AI metadata call routes through the **P1 provider abstraction** — so
  build the `LLMProvider` interface (first bullet of P1) *before or alongside*
  P1.0, not after. Cache decisions keyed on column-signature hash.
- Regression guard: extend the **AI eval harness** (P1) with golden
  classification cases — assert roles/agg/identifier verdicts on known
  datasets so P1.0 can't silently regress as models change.

### P1 — Provider-agnostic AI + analytics depth (the differentiator)

- **`src/analysis/llm/` provider abstraction**: `LLMProvider.generate(...)`
  interface; `GroqProvider` impl; model/provider chosen via `config.py`.
  `llm_analyst.py` depends on the interface only. Add response caching keyed
  on a hash of the ground-truth payload (cuts cost + latency, deterministic).
- **Statistical depth (deterministic, Layer 3 / EDA):**
  - Full association matrix: Spearman, mutual information, Cramér's V
    (categorical↔categorical), correlation ratio η (categorical↔numeric).
    *Today only numeric Pearson — ~90% of relationships are invisible.*
  - Real trend + seasonality: Mann-Kendall test + STL decomposition on
    datetime×numeric. *Today `patterns['trends']` is initialized and never
    computed.*
  - ML anomaly detection: IsolationForest/LOF for multivariate outliers,
    replacing per-column IQR.
  - Auto-segmentation: KMeans/HDBSCAN with silhouette-chosen k.
  - Distribution characterization: skew/kurtosis/normality → better chart
    selection + insights.
  - Driver analysis: when a target is evident, RandomForest/GBM importances
    ("X is the strongest driver of Y").
  - Deps: add `scikit-learn`, `statsmodels` (flag image-size impact on the
    HF Space Docker build — consider a slim wheel set).
- **AI eval harness**: 5–10 golden datasets with expected
  insights/labels; assert the LLM layer's selection quality in `pytest`.
  This is what stops quality regressing as models change through 2030.

### P2 — Scale & data formats

- **Job queue**: Arq or Celery on the existing Redis. `core/pipeline.py`
  runs as a worker task; API returns job id; reuse the SSE path for progress.
  Idempotent jobs keyed on data hash → free dedupe + result cache.
- **`DataFrameEngine` seam**: wrap pandas; add a DuckDB/Polars backend for
  larger-than-memory CSV/Parquet so 2030 datasets don't OOM the container.
- **More formats**: Parquet, Excel, JSON/NDJSON ingestion behind the same
  parser interface.
- **Per-user history**: persist analyses (schema already exists via
  SQLAlchemy/Alembic); list/reopen past dashboards; Clerk org multi-tenancy.

### P3 — "Ask your data" (the 2030 table-stakes feature)

- Conversational follow-up over a **semantic layer**: LLM proposes a
  deterministic query/stat (function-calling), the backend executes it, the
  LLM narrates the *returned numbers only*. Same trust guarantee, interactive.
- Bounded analysis agent: the stats layers exposed as tools; capped
  iterations; every number traceable. (Architecturally ready because of
  principle #2.)

### P4 — Reliability & delivery maturity

- SLOs + Grafana/OTel dashboards + alerting on the metrics already emitted;
  load test (Locust/k6) the analysis path; error budgets.
- Frontend test net: Vitest + React Testing Library + Playwright smoke
  (currently zero frontend tests); error boundaries; a11y pass.
- Automated CI/CD when billing allows: lint + pytest + frontend build + AI
  eval as the merge gate; Renovate/Dependabot + SBOM + dependency scanning so
  the stack doesn't rot before 2030.
- Framework currency policy: scheduled React/Vite/dep major-bump cadence
  (React 19+, Vite 6/7 land well before 2030).

---

## 3. Sequencing & rationale

```
P0 Security ─► P1.0 Correctness ─► P1 AI depth ─► P2 Scale ─► P3 Ask-data ─► P4 Maturity
(must-fix)     (gates everything)  (differentiator) (won't-scale) (table stakes) (longevity)
```

- **P0 before anything**: SSRF + unbounded ingestion are exploitable now.
- **P1.0 before P1 depth**: classification correctness gates every downstream
  output; advanced algorithms on a wrong-role classifier produce sophisticated
  garbage. The P1 `LLMProvider` interface + eval harness are built here (P1.0
  depends on them), so P1's first bullet effectively starts in P1.0.
- **P1 next**: it's the product's actual value and what the homepage claims;
  provider abstraction is cheap now and unblocks every future model.
- **P2/P3** convert it from a single-shot tool into a platform.
- **P4** is what keeps it alive to 2030 without a rewrite.

## 4. Explicit non-goals / risks

- Don't weaken the deterministic-numbers guarantee for "smarter" AI — it's
  the trust differentiator. P1.0 extends it to metadata: AI may decide
  roles/semantics/intent/descriptions, never values, never silent mutation.
- P1.0 item #6 (`_reshape_if_wide_timeseries`) mutates data shape — highest
  risk in the whole plan. Ship it behind a guard + user-surfaced notice with
  a deterministic fallback before trusting AI intent detection.
- P1.0 adds an LLM call to the hot classification path — enforce caching
  (column-signature hash) + timeout + heuristic fallback so latency/cost and
  provider outages can't break ingestion.
- `scikit-learn`/`statsmodels` inflate the HF Docker image; budget for build
  time / consider multi-stage slimming.
- Job queue adds an ops surface (a worker process) — justified only once
  datasets/throughput exceed request-thread limits; can be deferred but the
  `DataFrameEngine` + interface seams (P1/P2) should land regardless so the
  later migration is non-invasive.
```
