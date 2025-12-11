# ML Dashboard Generator – Technical Audit

## A. Critical Runtime Failures
1. **Missing dependency import in correlation engine** – `re` was not imported even though UUID detection uses it, causing a `NameError` during correlation analysis. (src/ml/correlation_engine.py)
2. **Undefined `numeric_cols` in EDA insights** – The EDA fallback correlation logic referenced `numeric_cols` before it was defined, leading to runtime failures whenever the advanced correlation engine returned no results. (src/eda/insights_generator.py)

## B. Integration / Contract Mismatches
1. **EDA numeric column sourcing** – Pattern detection and trend analysis assumed a pre-populated `numeric_cols` list from the dataset profile but never built it, breaking the expected contract of downstream logic and preventing trend/outlier sections from running.
2. **Pipeline orchestration overwriting analytics** – The dashboard pipeline executed correlation and EDA twice, with the second pass overwriting earlier results and timings. This created mismatched data passed to KPIs/charts versus what was ultimately returned to the frontend.

## C. Incorrect or Overly Strict Logic
- Duplicate correlation/EDA execution increased runtime and risked empty EDA output when the second pass failed, even if the first succeeded.

## D. Architectural Gaps from Partial Refactor
- The analyser/KPI/EDA pipeline expects a consistent `dataset_profile` (roles, semantic tags, counts). Missing numeric column derivation and repeated pipeline stages show incomplete propagation of the new analyser contract into EDA and orchestration layers.

## E. Files Requiring Rewrite or Redesign
- **src/core/pipeline.py** – Needs single-pass orchestration with consistent artefact hand-off.
- **src/eda/insights_generator.py** – Requires reliable sourcing of typed columns from the analyser profile for all downstream checks.
- **src/ml/correlation_engine.py** – Import hygiene and dependency validation.

## Code-Level Fixes Applied
### 1) Import `re` for UUID matching
**File:** `src/ml/correlation_engine.py`
```python
import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, List, Any, Tuple, Optional
from scipy.stats import pearsonr, spearmanr
import math
```
**Why:** UUID detection relied on `re.match`; without the import, correlation analysis raised `NameError` and aborted downstream insights. Importing `re` restores the intended identifier filtering.

### 2) Derive numeric columns before EDA fallbacks
**File:** `src/eda/insights_generator.py`
```python
columns = dataset_profile.get("columns", []) if dataset_profile else []
numeric_cols = [col.get("name") for col in columns if col.get("role") == "numeric" and col.get("name") in df.columns]
if not numeric_cols:
    # Fall back to dtype-based detection if profile is missing or empty
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
```
**Why:** EDA fallback correlations, trend detection, outlier detection, and distribution analysis need a valid numeric column list. The previous code referenced `numeric_cols` before assignment, causing runtime errors and empty insights; the fix builds the list from the analyser profile with a dtype fallback.

### 3) Remove duplicate correlation/EDA execution
**File:** `src/core/pipeline.py`
- Deleted the second correlation + EDA block that re-ran analysis and overwrote earlier results.

**Why:** The duplicate pass risked returning `None`/empty EDA output even when the first pass succeeded and inflated runtime. Single-pass orchestration keeps KPI/chart generation aligned with the analysis actually returned.

## Step-by-Step Refactor Plan
1. **Standardize dataset profile contract**
   - Lock roles (`numeric`, `categorical`, `datetime`, `identifier`, `boolean`, `ordinal`, `text`) and semantic tag names in a shared constants module.
   - Ensure analyser populates `semantic_tags`, `confidence`, and `provenance` consistently for every column.
2. **Propagate profile to downstream modules**
   - Update KPI generator, chart selector, correlation engine, and EDA to consume the standard profile instead of re-deriving dtypes.
   - Add validation checks at module boundaries to log when required profile fields are missing.
3. **Harmonize identifier filtering**
   - Centralize identifier detection in a utility and import it in KPI, correlation, EDA, and rendering to avoid divergent heuristics.
4. **Chart-selection resilience**
   - Add fallbacks in `chart_selector` so at least summary charts are always produced, even when semantic rules filter many columns.
5. **EDA robustness**
   - Guard trend/correlation/outlier paths with minimum-sample checks and fallback summaries to prevent empty sections.
6. **KPI scoring alignment**
   - Use analyser roles/semantic tags to weight KPI significance; ensure identifier/near-constant columns are filtered early.
7. **Template contract verification**
   - Document the data expected by Jinja templates (`kpis`, `charts`, `eda_summary`, `dataset_profile`) and add response-shape tests to catch regressions.
8. **Logging and error handling**
   - Standardize logger usage with context (column names, roles) and convert silent passes into structured warnings surfaced in the frontend when sections are skipped.
