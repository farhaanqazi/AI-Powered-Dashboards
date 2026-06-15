# INTEGRATION MAP — engine seams for the benchmark

Input to Document B. Records the exact, current locations of the seams the
benchmark drives. All references are against tag **`benchmark-baseline-v1`**
(commit `45ec5c2`). Read-only map — no engine source was modified to produce it.

> Convention: `path:line` points at the `def`/`class` as of the tagged commit.

---

## 1. Pipeline entrypoint (how the harness drives the engine)

`src/core/pipeline.py` — all headless, no FastAPI/Request context:

| Entry | Line | Signature | Use |
|---|---|---|---|
| `build_dashboard_from_df` | [pipeline.py:224](../src/core/pipeline.py#L224) | `(df, max_cols=50, original_filename=…) -> DashboardState \| None` | **Primary** harness entrypoint (used by the smoke). Synchronous, returns a `DashboardState`. |
| `build_dashboard_from_df_generator` | [pipeline.py:532](../src/core/pipeline.py#L532) | yields phase events, final `{"phase":"done","state":…}` | Streaming variant (progress events). |
| `build_dashboard_from_file_generator` | [pipeline.py:490](../src/core/pipeline.py#L490) | `(stream, original_filename, encoding=…)` | Reads raw bytes (CSV sniff etc.) then runs the generator. |

- Convert state → canonical dict with `state_to_payload(state, filename)` (`src/core/state_payload.py`).
- **Side effects to redirect** (read-only w.r.t. source, NOT filesystem): durable df-cache parquet (`config.CLEANED_DF_DURABLE_DIR`), spool (`config.JOB_SPOOL_DIR`), logs (`LOG_DIR`), sqlite (`DATABASE_URL`). The smoke sets all to a temp dir + `SCHEMA_REVIEW_ENABLED=false`. Set these env vars **before** importing `src` (config reads them at import).
- **Gate to watch:** with `SCHEMA_REVIEW_ENABLED=True` (default) low-confidence tables halt at HITL review and emit **no** charts/ML — the run early-exits. Batch eval must disable it.

---

## 2. Provenance tokens — where emitted, what fields (the "Trust" signal)

There are **three** distinct provenance sites. They do **not** share one chokepoint — this matters for Document B's comparison-arm choice.

### 2a. Dashboard KPIs — LLM-validated path only
`src/analysis/llm_analyst.py`
- Tokens attached at [llm_analyst.py:240](../src/analysis/llm_analyst.py#L240) and [:258](../src/analysis/llm_analyst.py#L258); "every KPI must carry a provenance token" backfill at [:450](../src/analysis/llm_analyst.py#L450).
- Validated against `LLMOutputContract.validate_output(...)` at [:453-455](../src/analysis/llm_analyst.py#L453) (in `src/contract/models.py`). Reject → logged heuristic fallback.
- **⚠ Gating (verified, not inferred):** `run_ai_analyst` ([:388](../src/analysis/llm_analyst.py#L388)) returns the caller's **heuristic fallback before** the provenance code when no/!enabled LLM provider. **Result: with no LLM, dashboard KPIs carry `provenance = None`.** The smoke confirmed this (KPI `amount`, `provenance=None`, 0 payload provenance fields).
- Field shape (LLM path): `{"provenance": {... "column:"/"corr:" source + L1/L3 token ...}}`.

### 2b. Ask / Interact — deterministic, **no LLM needed**
`src/analysis/ask/tools.py` — every tool stamps a token:
- `column_stat` [tools.py:63](../src/analysis/ask/tools.py#L63) → `{"source":"column:<col>","metric":…,"token":"L1.<col>.<metric>"}`
- `aggregate` [:96](../src/analysis/ask/tools.py#L96) → `token:"agg.<group>.<agg>"`
- `top_categories` [:113](../src/analysis/ask/tools.py#L113), `correlation` [:134](../src/analysis/ask/tools.py#L134) → `token:"L3.correlation.<a>|<b>"`, `filter_count` [:160](../src/analysis/ask/tools.py#L160).

`src/analysis/ask/interact.py` — wraps a tool result and asserts traceability:
- adds `verified`, `filtered`, `rows_after`, **`derived_token`** at [interact.py:151-160](../src/analysis/ask/interact.py#L151) and returns **`numbers_traceable: True`** at [:170](../src/analysis/ask/interact.py#L170).
- Smoke output (real, no LLM): `result={'column':'amount','metric':'mean','value':304.0}`, `numbers_traceable=True`, `provenance={'token':'L1.amount.mean', 'derived_token':'derived:L1.amount.mean#…', 'verified':True, …}`.

### 2c. ML what-if / re-segment (Phase 15 S15.4)
`src/analysis/ml/predict.py` — `run_predict`/`run_resegment` return `provenance.token` + `numbers_traceable: True` at [predict.py:109-114](../src/analysis/ml/predict.py#L109) and [:139-144](../src/analysis/ml/predict.py#L139).

> **Takeaway for Document B:** the provenance fence is concentrated in the
> **LLM-validated dashboard path** and the **deterministic Ask/Interact + ML
> paths**. The no-LLM dashboard build is *unprovenanced*. A "Trust" metric must
> therefore measure either (i) the LLM-validated path (needs an LLM key) or
> (ii) the Ask/Interact path. This is why fence-ablation is fragile and the
> external naive-LLM baseline (§4) is the cleaner arm.

---

## 3. The future authority-knob site (L4 select + role-router constraints)

The "authority" = how hard the engine constrains what may be computed/charted.
Enforced deterministically across these sites (none take an authority parameter
today — Document B adds one; do **not** add it here):

- **Contract allow-lists (source of truth):** `src/contract/compiler.py` — per role/domain `allowed_aggregations` + `allowed_charts` at [compiler.py:143-144](../src/contract/compiler.py#L143) (tables `_AGG_ALLOW`/`_CHART_ALLOW` at [:30](../src/contract/compiler.py#L30)).
- **Role router (the predicate layer):** `src/contract/role_router.py` — `field_view` [:51](../src/contract/role_router.py#L51), `get_allowed_aggregations` [:85](../src/contract/role_router.py#L85), `is_correlatable` [:100](../src/contract/role_router.py#L100), `can_sum` [:109](../src/contract/role_router.py#L109), `recompute_ratio` [:117](../src/contract/role_router.py#L117), `collapse_to_grain` [:140](../src/contract/role_router.py#L140).
- **L4 interpret / chart-select (consumes the router):** `src/analysis/layer_4_interpreter.py` — imports `field_view` ([:15](../src/analysis/layer_4_interpreter.py#L15)); `determine_kpis` [:75](../src/analysis/layer_4_interpreter.py#L75), `select_charts` [:136](../src/analysis/layer_4_interpreter.py#L136). **This is the primary authority-knob insertion point.**
- **Other enforcement consumers:** L3 correlation filter `is_correlatable` ([layer_3_relational.py:42](../src/analysis/layer_3_relational.py#L42)); EDA totals `can_sum` ([eda_analyzer.py:14](../src/analysis/eda_analyzer.py#L14)); invariant critic vetoes `critique`/`apply_vetoes` ([invariant_critic.py:205,215](../src/contract/invariant_critic.py#L205)).

---

## 4. LLM provider interface (the comparison-arm seam)

`src/analysis/llm/`
- ABC: `LLMProvider` at [base.py:23](../src/analysis/llm/base.py#L23); `LLMUnavailable` at [:14](../src/analysis/llm/base.py#L14).
- Factory: `get_llm_provider()` at [factory.py:42](../src/analysis/llm/factory.py#L42); `NullProvider` (fail-safe, returns nothing usable) at [factory.py:16](../src/analysis/llm/factory.py#L16).
- Concrete: `src/analysis/llm/groq_provider.py` (only module importing the SDK); response cache `src/analysis/llm/cache.py`.
- **Arm drop point (Document B):** implement the external naive-LLM (LIDA-class) baseline as an `LLMProvider` and inject via the factory. The engine depends only on the ABC, so the baseline runs **without** the contract/provenance fence while everything else stays identical.

---

## 5. Reproduction

```
venv/Scripts/python.exe -m benchmark.smoke_test            # demo fixture
venv/Scripts/python.exe -m benchmark.smoke_test <path.csv> # any table
```
Baseline tag `benchmark-baseline-v1` (`45ec5c2`); deps pinned in `requirements.lock.txt`.
