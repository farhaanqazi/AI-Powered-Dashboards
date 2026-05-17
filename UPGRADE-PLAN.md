# AI Powered Dashboards — Unified Upgrade Plan (DATED)

> Single source of truth. Spine = the Semantic Contract Layer (Phases 0–8).
> Surviving items from the former 2030 roadmap (provider abstraction, analytics
> depth, scale, ask-data, maturity) are Phases 9–12. The former `UPGRADE-PLAN-2030.md`
> P1.0 (AI-augmented classification) is superseded by Phases 1–8.
> Baseline date: 2026-05-17. Every phase and step is dated.

---

## Phase 0 — Security Hardening [blocker] — 2026-05-19 → 2026-05-23 — ✅ 2026-05-19 → 2026-05-23 -- 2026-05-17 → 2026-05-17
- **S0.1** — 2026-05-19 — Verify `.env` is gitignored (`git ls-files | findstr .env` → empty); `git rm --cached .env` if tracked. — ✅ 2026-05-19 -- 2026-05-17 (verified: `.env` gitignored & untracked, only `.env.example` tracked)
- **S0.2** — 2026-05-19 — Add `SENSITIVITY_FAIL_CLOSED=True`, `PII_BLOCK_EGRESS=True` to `src/config.py`. — ✅ 2026-05-19 -- 2026-05-17
- **S0.3** — 2026-05-20 → 2026-05-21 — SSRF hardening in `src/data/parser.py`: scheme allowlist, block private/link-local/metadata IP ranges, max-size + timeout + redirect cap, content-type sniff before parse. — ✅ 2026-05-20 → 2026-05-21 -- 2026-05-17
- **S0.4** — 2026-05-21 — Upload hard caps (file size / row / column); reject on stream, not after load. — ✅ 2026-05-21 -- 2026-05-17
- **S0.5** — 2026-05-22 — Endpoint auth audit (`src/auth.py`); per-IP/user rate limiting; remove or auth-gate `/debug-build-files` and `/test-persistence`. — ✅ 2026-05-22 -- 2026-05-17
- **S0.6** — 2026-05-23 — Clear `dataInsight:guestMode` on Clerk sign-in in `frontend/src/dashboardStore.js`. — ✅ 2026-05-23 -- 2026-05-17

## Phase 1 — Ingest Contract Gate — 2026-05-25 → 2026-05-29 — ✅ 2026-05-25 → 2026-05-29 -- 2026-05-17 → 2026-05-17
- **S1.1** — 2026-05-25 — Create `src/contract/__init__.py`, `src/contract/ingest_gate.py`. — ✅ 2026-05-25 -- 2026-05-17
- **S1.2** — 2026-05-26 → 2026-05-28 — Implement `run_ingest_gate`: thousands/currency coercion, sentinel→`pd.NA`, null-row rejection, Presidio PII detection, `sensitivity`/`pii_blocked`. — ✅ 2026-05-26 → 2026-05-28 -- 2026-05-17
- **S1.3** — 2026-05-28 — `src/contract/models.py`: `CleaningManifest`, `IngestResult`. — ✅ 2026-05-28 -- 2026-05-17
- **S1.4** — 2026-05-29 — Add `presidio-analyzer`/`presidio-anonymizer` to `requirements.txt`. — ✅ 2026-05-29 -- 2026-05-17 (HF build proved Presidio→spaCy→thinc has no installable build on the HF Python; relocated to optional `requirements-pii.txt`. `ingest_gate` regex fallback is the deployed PII tier — deployment is correct without it.)

## Phase 2 — Contract Models, Compiler, Fingerprint Cache — 2026-06-01 → 2026-06-09 — ✅ 2026-06-01 → 2026-06-09 -- 2026-05-17 → 2026-05-17
- **S2.1** — 2026-06-01 → 2026-06-02 — Frozen Pydantic `FieldContract` + `DatasetContract` in `models.py`. — ✅ 2026-06-01 → 2026-06-02 -- 2026-05-17
- **S2.2** — 2026-06-03 → 2026-06-05 — `src/contract/compiler.py` `compile_contract`: schema fingerprint, grain detection, aggregate-row flagging, year/ratio classification, agg/chart allow-lists. — ✅ 2026-06-03 → 2026-06-05 -- 2026-05-17
- **S2.3** — 2026-06-08 — `layer_2_classifier.py` emits `confidence` + top-2 `alternatives`; extend `EnrichedProfile` in `data_structures.py`. — ✅ 2026-06-08 -- 2026-05-17
- **S2.4** — 2026-06-09 — `src/contract/cache.py` over `src/persistence/cache.py`; locked-hit skips recompile + LLM. — ✅ 2026-06-09 -- 2026-05-17

## Phase 3 — Role-Aware Router — 2026-06-10 → 2026-06-15 — ✅ 2026-06-10 → 2026-06-15 -- 2026-05-17 → 2026-05-17
- **S3.1** — 2026-06-10 → 2026-06-11 — `src/contract/role_router.py`: `get_allowed_aggregations`, `is_correlatable`, `collapse_to_grain`, `recompute_ratio`. — ✅ 2026-06-10 → 2026-06-11 -- 2026-05-17
- **S3.2** — 2026-06-12 — `layer_3_relational.py` filters pairs via `is_correlatable`. — ✅ 2026-06-12 -- 2026-05-17
- **S3.3** — 2026-06-15 — `layer_4_interpreter.py` + `eda_analyzer.py` use the router (ids excluded, years→min/max/range, ratio totals recomputed, panel→grain). — ✅ 2026-06-15 -- 2026-05-17 (ids excluded from corr/outliers, years→range indicator, ratios excluded from totals via `can_sum`; `collapse_to_grain`/`recompute_ratio` implemented + unit-tested in the router — layer invocation of grain-collapse activates when the contract grain is threaded in Phase 5 S5.1)

## Phase 4 — Invariant Critic — 2026-06-16 → 2026-06-19 — ✅ 2026-06-16 → 2026-06-19 -- 2026-05-17 → 2026-05-17
- **S4.1** — 2026-06-16 → 2026-06-19 — `src/contract/invariant_critic.py`: unique-numeric→identifier veto, fractional-ID veto, total-vs-components flag, share-sum flag, std≫mean flag (config-driven tolerances). — ✅ 2026-06-16 → 2026-06-19 -- 2026-05-17

## Phase 5 — Pipeline Wiring — 2026-06-22 → 2026-06-26 — ✅ 2026-06-22 → 2026-06-26 -- 2026-05-17 → 2026-05-17
- **S5.1** — 2026-06-22 → 2026-06-25 — `src/core/pipeline.py` (sync + generator): ingest gate, `pii_blocked` short-circuit, compile + vetoes + cache, `schema_review` gating before L3/L4/EDA/LLM/render, thread + persist contract. — ✅ 2026-06-22 → 2026-06-25 -- 2026-05-17 (both paths wired; rejected→DQ state; `schema_review` gate present but inert via `config.SCHEMA_REVIEW_ENABLED=False` until Phase 6 S6.3 sets the auto-accept criterion + Phase 7 ships the UI; contract+cleaning+vetoes+flags persisted in `dataset_profile`)
- **S5.2** — 2026-06-26 — `llm_analyst.py`: contract-validated aggregated payload only; never send raw `pii` rows. — ✅ 2026-06-26 -- 2026-05-17 (PII-blocked → LLM never called; sensitive columns redacted from ground truth — no top_categories, no stats)

## Phase 6 — LLM Output Validator + Graceful Degradation — 2026-06-29 → 2026-07-03 — ✅ 2026-06-29 → 2026-07-03 -- 2026-05-17 → 2026-05-17
- **S6.1** — 2026-06-29 → 2026-06-30 — `LLMOutputContract` validation + per-number provenance tokens; explicit logged fallback. — ✅ 2026-06-29 → 2026-06-30 -- 2026-05-17 (every KPI carries a `provenance` token (`column:`/`corr:` + L1/L3 token); validator rejects unknown columns/bad intents/missing provenance → logged heuristic fallback)
- **S6.2** — 2026-07-01 — `src/contract/dq_report.py` `DataQualityReport`; emitted on undetectable grain / `pii_blocked` / unlockable contract. — ✅ 2026-07-01 -- 2026-05-17 (status ok/review/blocked; threaded into `dataset_profile.data_quality.report` + `eda_summary.data_quality_report`)
- **S6.3** — 2026-07-02 → 2026-07-03 — Auto-accept rule via `config.AUTO_ACCEPT_CONFIDENCE` (no hardcoded literals); sub-threshold/pii blocks lock. — ✅ 2026-07-02 → 2026-07-03 -- 2026-05-17 (`evaluate_acceptance`: mean-conf ≥ threshold ∧ ¬pii_blocked ∧ grain → auto-lock; else unlocked + DQ "review"/"blocked". Pipeline halt still gated by `SCHEMA_REVIEW_ENABLED` until Phase 7 UI)

## Phase 7 — HITL Schema Review (API + Frontend) — 2026-07-06 → 2026-07-15 — ✅ 2026-07-06 → 2026-07-15 -- 2026-05-17 → 2026-05-17
- **S7.1** — 2026-07-06 → 2026-07-08 — `PATCH /api/dashboard/{id}/registry`: override → lock (`version+=1`) → persist by fingerprint → recompute L3→render (skip profiling + LLM). — ✅ 2026-07-06 → 2026-07-08 -- 2026-05-17 (`src/contract/registry_patch.py`; override→re-derive allow-lists→lock(version+1)→persist by session; PII stays blocked through review. **Post-deploy feedback fix 2026-05-17:** real L3→render recompute now works — `src/contract/df_cache.py` holds the cleaned frame in a transient, TTL-bound, fingerprint-keyed, config-gated cache (`CLEANED_DF_CACHE_ENABLED`); `src/contract/rebuild.py` re-runs L1-stats + contract roles → L3 → L4 → render with no LLM. Graceful fallback to contract-only drop when the cache misses/disabled. `report.recomputed` reflects which path ran.)
- **S7.2** — 2026-07-08 — Draft-contract + PATCH-body schemas in `src/api/schemas.py`. — ✅ 2026-07-08 -- 2026-05-17 (`FieldOverride`, `RegistryPatchRequest` (non-skippable `confirm`), `RegistryPatchResponse`)
- **S7.3** — 2026-07-09 → 2026-07-11 — `ColumnsTab.jsx` editable contract table (role/domain/sensitivity, confidence, alternatives, badges); non-skippable confirm. — ✅ 2026-07-09 → 2026-07-11 -- 2026-05-17 (per-row role `<select>`, pending-edit counter, mandatory confirm bar wired to the store)
- **S7.4** — 2026-07-12 → 2026-07-13 — `DataQualityTab.jsx` + register first-tab in `DashboardPage.jsx`. — ✅ 2026-07-12 → 2026-07-13 -- 2026-05-17 (cleaning/vetoes/flags/PII/status report; auto-lands on review)
- **S7.5** — 2026-07-14 → 2026-07-15 — `dashboardStore.js` + `services/api.js` handle `schema_review`/draft/PATCH/`data_quality`. — ✅ 2026-07-14 → 2026-07-15 -- 2026-05-17 (`patchRegistry`, `submitSchemaReview`, `schemaReview`/`needsSchemaReview` selectors). Backend `SCHEMA_REVIEW_ENABLED` remains False by default — the review *halt* is opt-in; the PATCH path + UI work whenever the gate is enabled. Frontend `npm run build` green; Python gate 137 passed.

## Phase 8 — CI/CD Contract Testing — 2026-07-16 → 2026-07-22
- **S8.1** — 2026-07-16 → 2026-07-22 — `tests/contract/`: hypothesis normalization/rejection, snapshot contract mappings, backward-compat load; wire into local `pytest` gate.

## Phase 9 — Provider-Agnostic AI + Analytics Depth — 2026-07-23 → 2026-08-14
*(absorbs former 2030 P1; former P1.0 superseded by Phases 1–8)*
- **S9.1** — 2026-07-23 → 2026-07-29 — `src/analysis/llm/`: `LLMProvider` interface + `GroqProvider`; `llm_analyst.py` depends on interface only; provider/model via `config.py`; response cache keyed on ground-truth hash.
- **S9.2** — 2026-07-30 → 2026-08-11 — Statistical depth (deterministic): Spearman/MI/Cramér's V/η, Mann-Kendall + STL, IsolationForest/LOF, KMeans/HDBSCAN, skew/kurtosis/normality, RandomForest driver analysis; add `scikit-learn`/`statsmodels` (budget HF Docker image size, multi-stage slim).
- **S9.3** — 2026-08-12 → 2026-08-14 — AI eval harness: 5–10 golden datasets + golden classification cases asserted in `pytest`.

## Phase 10 — Scale & Data Formats — 2026-08-17 → 2026-09-04
- **S10.1** — 2026-08-17 → 2026-08-24 — Job queue (Arq/Celery on existing Redis); API returns job id; reuse SSE; idempotent jobs keyed on data hash.
- **S10.2** — 2026-08-25 → 2026-08-31 — `DataFrameEngine` seam (pandas default; DuckDB/Polars backend for larger-than-memory).
- **S10.3** — 2026-09-01 → 2026-09-02 — Parquet / Excel / JSON-NDJSON ingestion behind the parser interface.
- **S10.4** — 2026-09-03 → 2026-09-04 — Per-user history (persist/list/reopen analyses; Clerk org multi-tenancy).

## Phase 11 — "Ask Your Data" — 2026-09-07 → 2026-09-25
- **S11.1** — 2026-09-07 → 2026-09-18 — Semantic-layer conversational follow-up: LLM proposes deterministic query/stat via function-calling; backend executes; LLM narrates returned numbers only.
- **S11.2** — 2026-09-21 → 2026-09-25 — Bounded analysis agent: stats layers exposed as tools, capped iterations, every number traceable.

## Phase 12 — Reliability & Delivery Maturity — 2026-09-28 → 2026-10-16
- **S12.1** — 2026-09-28 → 2026-10-05 — SLOs + Grafana/OTel dashboards + alerting on emitted metrics; Locust/k6 load test; error budgets.
- **S12.2** — 2026-10-06 → 2026-10-12 — Frontend test net: Vitest + React Testing Library + Playwright smoke; error boundaries; a11y pass.
- **S12.3** — 2026-10-13 → 2026-10-16 — Automated CI/CD merge gate (lint + pytest + frontend build + AI eval) when billing allows; Renovate/Dependabot + SBOM; React/Vite major-bump cadence policy.

---

## Architectural invariants (must hold across all phases)
- Deterministic numbers, AI decorative: every displayed figure traces to L1–L3/EDA via a provenance token; the LLM never computes or mutates a value.
- ~~PII is fail-closed: `pii_blocked` blocks LLM egress entirely; human approval does not unblock it.~~ **SUPERSEDED 2026-05-17 (product decision):** the deterministic dashboard sends nothing externally, so it always builds regardless of PII. The LLM/AI Insights layer is the only egress and is now gated behind **explicit user consent** (`ai_consent`, surfaced as `ai_consent_required`), not a permanent wall — sharing one's own sensitive data is the user's prerogative and responsibility. `pii_blocked` is retained meaning "PII detected → AI needs consent" (it is still never cleared by a *role* override; consent is a separate, deliberate opt-in via `POST /api/dashboard/{trace_id}/ai-consent`). On consent the AI layer re-runs full-send (no redaction) against the transiently-cached frame; no new storage backend.
- Contracts are immutable post-compile; the only mutation path is a HITL override producing a new locked version.
- No new storage backend: the contract serializes into the existing `DashboardRecord`.
- No hardcoded confidence/threshold literals in code; all live in `src/config.py`.
- Local `pytest` remains the merge gate (CI is billing-locked).
