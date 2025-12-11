# ML Dashboard Generator Diagnostic (Independent Pass)

## A. Critical runtime failures
- None found via static scan/`python -m compileall`. The pipeline runs end-to-end without syntax errors. Additional runtime issues may surface with specific datasets; targeted checks are listed below.

## B. Integration / contract mismatches
1) **Extended numeric roles dropped downstream** – The analyser emits `numeric_duration`, `numeric_currency`, `numeric_percentage`, and `numeric_mixed`, but chart selection, correlations, and EDA only treated a literal `"numeric"` role. These columns were excluded from correlations, fallback chart suggestions, and numeric KPIs, causing empty or sparse outputs.
   - Fix applied: introduce `_is_numeric_role` helper and use it across the correlation engine, EDA insights, and chart selector fallback so any role beginning with `numeric` is handled as numeric. (See code changes.)

## C. Incorrect or overly strict logic
1) **Identifier filtering still skips useful numeric-like columns** – identifier heuristics are aggressive (very-high-cardinality & name-based). Consider lowering thresholds or surfacing confidence for downstream filtering so non-ID numeric columns aren’t dropped. (No code change applied in this pass.)
2) **Correlation/EDA double-computation** – `pipeline.build_dashboard_from_df` runs `analyze_correlations` once and `generate_eda_summary` calls it again internally. This doubles cost and can yield slightly different filtering. Recommend reusing the earlier result via parameter or cache.

## D. Architectural gaps from partial refactors
1) **Semantic tags unused** – semantic tags produced by the analyser are not consumed by chart selection or KPI scoring, limiting meaningful chart types (e.g., currency vs. duration) and KPI weighting.
2) **Confidence/provenance ignored** – downstream modules do not surface `confidence` or `provenance`, so UI cannot show reliability of inferred roles.

## E. Files needing redesign
- **`src/ml/chart_selector.py` / `src/ml/correlation_engine.py` / `src/eda/insights_generator.py`** – need standardized role normalization (numeric, categorical, datetime), better sharing of identifier confidence, and optional reuse of semantic tags.

## Exact code-level fixes
- **Normalize numeric roles across modules** (implemented):
  - `src/ml/correlation_engine.py`: add `_is_numeric_role` helper and use it wherever numeric columns are selected for correlations and summary stats.
  - `src/eda/insights_generator.py`: add `_is_numeric_role`; include all `numeric*` roles in numeric lists and summary counts.
  - `src/ml/chart_selector.py`: add `_is_numeric_role`; allow fallback histogram generation for `numeric*` roles.

## Refactor plan (step-by-step)
1. **Role normalization**: create a shared utility (e.g., `src/core/roles.py`) with helpers `is_numeric_role`, `is_datetime_role`, `is_categorical_role`, and apply across analyser, correlation, EDA, KPI, chart selector, and renderers.
2. **Semantic tag usage**: feed semantic tags to chart selector (choose time-series vs. currency-specific formatting) and KPI generator (monetary totals, duration aggregates).
3. **Confidence propagation**: pass `confidence` and `provenance` into EDA summaries and template contexts; surface in UI for transparency.
4. **Correlation reuse**: let `generate_eda_summary` accept precomputed correlation results to avoid recomputation; cache in pipeline state.
5. **Identifier handling**: share identifier confidence across modules; allow adjustable thresholds and logging when columns are skipped.
6. **Template alignment**: add UI fields for semantic tags and confidence per column so dashboards show why decisions were made.
7. **Testing**: add fixtures covering columns with `numeric_duration`, `numeric_percentage`, etc., ensuring correlations, KPIs, and charts include them.
