# QWEN Work Brief: ML Dashboard Generator

This condensed brief rephrases the technical audit so Pinokio/QWEN can act on it directly. Keep instructions explicit and avoid creative reinterpretation.

## Repository landmarks
- Backend entrypoints: `main.py`, FastAPI/Flask style.
- Orchestration: `src/core/pipeline.py`.
- Analyser output used everywhere: `dataset_profile` (roles, semantic tags, counts, confidence, provenance).
- Analytics modules: `src/eda/insights_generator.py`, `src/ml/correlation_engine.py`, `src/ml/kpi_generator.py`, `src/ml/chart_selector.py`, `src/ml/kpi_generator.py`.
- Rendering/templates: `src/viz/plotly_renderer.py`, `templates/` (Jinja).

## Confirmed runtime fixes already applied (do not regress)
1) `src/ml/correlation_engine.py` imports `re` so UUID-based identifier filtering works.
2) `src/eda/insights_generator.py` now builds `numeric_cols` from the analyser profile (with dtype fallback) before EDA fallbacks run.
3) `src/core/pipeline.py` runs correlation + EDA once so results/timings are not overwritten.

## Problems still to solve
- The analyser contract (`dataset_profile`) is not standardized or validated across modules.
- KPI/EDA/correlation/chart selector each re-derive roles instead of trusting the profile, leading to divergent logic.
- Identifier filtering heuristics are duplicated and inconsistent.
- Chart selection can return empty when semantic filters exclude too much.
- Templates lack a documented response shape; regressions may render empty sections silently.
- Logging/error handling is inconsistent; skipped sections are not surfaced.

## Action plan for QWEN (follow in order)
1) **Create a shared contract module** (e.g., `src/core/schema.py`):
   - Define allowed roles: `numeric`, `categorical`, `datetime`, `identifier`, `boolean`, `ordinal`, `text`.
   - Define optional fields: `semantic_tags: List[str]`, `confidence: float`, `provenance: str`.
   - Provide helper `validate_dataset_profile(profile)` that checks required keys and logs warnings.

2) **Propagate the contract**:
   - Update `kpi_generator.py`, `chart_selector.py`, `correlation_engine.py`, and `insights_generator.py` to consume `dataset_profile` roles/semantic_tags instead of recomputing dtypes.
   - Add guard clauses: if a required field is missing, log a warning and fall back to dtype inference.

3) **Centralize identifier filtering**:
   - Add a helper (e.g., `core/identifiers.py`) for detecting ID-like columns (UUIDs, monotonically increasing IDs, high-cardinality codes).
   - Replace in-place regex/length heuristics in KPI, correlation, EDA, and renderer with this helper so all modules agree.

4) **Harden chart selection**:
   - In `chart_selector.py`, ensure at least summary charts (distributions, counts) are emitted when semantic filters remove candidates.
   - Add a fallback rule: if no semantic match, pick top 1–2 numeric columns for histogram/box + categorical counts when available.

5) **EDA robustness**:
   - In `insights_generator.py`, gate trend/correlation/outlier logic with minimum sample checks but return descriptive summaries instead of empty lists when thresholds fail.
   - Make sure `numeric_cols`/`categorical_cols` always derive from the profile first, then dtype fallback.

6) **KPI scoring alignment**:
   - Use roles/semantic_tags to weight KPI importance; filter identifiers and near-constant columns early using the shared helper.

7) **Template/response contract**:
   - Document the response shape expected by Jinja templates (`kpis`, `charts`, `eda_summary`, `dataset_profile`) and add a lightweight test or schema check so empty keys surface errors instead of silently rendering nothing.

8) **Logging standard**:
   - Standardize logger usage (module-level logger, context with column name + role). Convert silent skips into structured warnings that bubble to the frontend if sections are omitted.

## Delivery checklist for Pinokio/QWEN
- Keep existing fixes intact (imports, single-pass pipeline, numeric_cols build).
- Add/modify files only within the repo—no new external dependencies.
- After coding, run at least a lightweight test (e.g., `python -m compileall src`) to catch syntax issues.
- Ensure chart/KPI/EDA outputs cannot be empty without an explicit warning message.
