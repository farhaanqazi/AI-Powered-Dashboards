# Full Repository Code Listing

This document captures the current contents of all source, template, and configuration files.

## Dockerfile

```
FROM python:3.9

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

## README.md

```
---
title: ML Dashboard Generator
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# ML Dashboard Generator

A FastAPI-based dashboard generator that analyzes CSV datasets and creates interactive visualizations. The application automatically profiles datasets, identifies KPIs (important columns), suggests charts using a ChartSpec structure, and renders visualizations using Plotly.

## Features

- Upload CSV files or load from external sources (URLs, Kaggle datasets)
- Automatic dataset profiling (numeric, datetime, categorical, text detection)
- KPI generation highlighting important columns based on statistical and semantic analysis
- Multiple chart types: bar, line, scatter, pie, histogram, box plots, correlation matrix
- Interactive dashboard with dataset summary and column profiling table
- **NEW**: Enhanced Exploratory Data Analysis (EDA) with pattern recognition
- **NEW**: Automatic key indicator identification with significance scoring
- **NEW**: Dataset use case detection and recommendations
- **NEW**: Advanced visualization options with correlation heatmaps and outlier detection
- **NEW**: Tabbed interface for different analysis views (Dashboard, EDA, Visualizations)

## Usage

1. Upload a CSV file using the "Upload" button
2. Or load a dataset from a URL or Kaggle dataset by providing the source URL or Kaggle slug
3. Explore the automatically generated dashboard with visualizations and insights
4. Switch between different analysis views using the tab interface:
   - Dashboard: Traditional visualizations
   - EDA Analysis: Detailed exploratory data analysis with key indicators and use cases
   - Visualizations: Advanced visualization charts

## Technical Details

- Backend: Python/FastAPI with pandas for data processing
- Frontend: Jinja2 templates with Plotly.js for visualizations
- Responsive design with CSS grid for chart layout
- Clean architecture with separate modules for parsing, analysis, KPI generation, chart selection, and visualization
- Enhanced EDA module with statistical analysis and pattern recognition

## Architecture

- `main.py`: Main FastAPI application
- `src/core/pipeline.py`: Core dashboard builder orchestrating the entire flow
- `src/data/parser.py`: CSV loading from various sources with validation
- `src/data/analyser.py`: Dataset profiling (detects roles: numeric, datetime, categorical, text)
- `src/ml/kpi_generator.py`: Generates KPIs based on statistical and semantic analysis with enhanced identifier detection and meaningful scoring
- `src/ml/chart_selector.py`: Suggests charts for different data types and relationships
- `src/viz/plotly_renderer.py`: Plotly chart rendering functions
- `src/viz/simple_renderer.py`: Simple chart data generation for reliable frontend rendering
- `src/eda/insights_generator.py`: Enhanced EDA functionality with pattern recognition and key indicator identification
- `src/viz/eda_visualizer.py`: Advanced visualization for EDA insights
- `templates/index.html`: Upload page with CSV upload and external dataset loading
- `templates/dashboard.html`: Dashboard with dataset summary, KPIs, charts grid, and column profiling table with tabbed interface

## Deployment

### Local Development

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python main.py
   ```
5. Open your browser and go to `http://localhost:7860`

### Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t ml-dashboard .
   ```
2. Run the container:
   ```bash
   docker run -p 7860:7860 ml-dashboard
   ```

### Hugging Face Spaces Deployment

The application is configured for deployment on Hugging Face Spaces with Docker. The configuration is in the main.py file which follows the FastAPI structure for serving on Spaces.

To deploy on Hugging Face Spaces:

1. Create a new Space on Hugging Face
2. Connect your repository
3. The Space will automatically build and deploy using the Docker configuration

The Space will be accessible at `https://huggingface.co/spaces/{username}/ml-dashboard-generator`

### Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`:
  - fastapi
  - uvicorn[standard]
  - pandas
  - plotly
  - kagglehub
  - scipy
  - requests
  - python-multipart
  - jinja2

## API Documentation

### Programmatic Access

The application offers several endpoints for programmatic access:

- **POST /upload**: Upload a CSV file and generate a dashboard
- **POST /load_external**: Load a dataset from a URL or Kaggle dataset slug
- **GET /**: Main index page for file upload and external source loading

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

If you encounter any issues or have questions, please file an issue in the GitHub repository.
```

## requirements.txt

```
fastapi
uvicorn[standard]
pandas
plotly
kagglehub
scipy
requests
python-multipart
jinja2
```

## .gitignore

```
venv/
env/
env*/
venv*/
.venv
.venv*/
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
.vscode/
.idea/
*.swp
*.swo
.DS_Store
Thumbs.db
logs/
*.log
.env
.env.local
*.csv
*.xlsx
*.xls
*.parquet
*.json
```

## .gitattributes

```
*.7z filter=lfs diff=lfs merge=lfs -text
*.arrow filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.bz2 filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
*.ftz filter=lfs diff=lfs merge=lfs -text
*.gz filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
*.lfs.* filter=lfs diff=lfs merge=lfs -text
*.mlmodel filter=lfs diff=lfs merge=lfs -text
*.model filter=lfs diff=lfs merge=lfs -text
*.msgpack filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.ot filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.pb filter=lfs diff=lfs merge=lfs -text
*.pickle filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.rar filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
saved_model/**/* filter=lfs diff=lfs merge=lfs -text
*.tar.* filter=lfs diff=lfs merge=lfs -text
*.tar filter=lfs diff=lfs merge=lfs -text
*.tflite filter=lfs diff=lfs merge=lfs -text
*.tgz filter=lfs diff=lfs merge=lfs -text
*.wasm filter=lfs diff=lfs merge=lfs -text
*.xz filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
*.zst filter=lfs diff=lfs merge=lfs -text
*tfevents* filter=lfs diff=lfs merge=lfs -text

```

## main.py

```
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
import pandas as pd
import io
import os
from src.core.pipeline import build_dashboard_from_file, build_dashboard_from_df
from src.data.parser import load_csv_from_url, load_csv_from_kaggle
from starlette.responses import RedirectResponse

app = FastAPI()

# Create a custom Jinja2 environment to avoid Flask-specific functions
env = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=True
)

# Compatibility shim for old Flask-style templates that call get_flashed_messages()
def fake_get_flashed_messages(*args, **kwargs):
    # Always return an empty list so templates don't crash
    return []

env.globals["get_flashed_messages"] = fake_get_flashed_messages

# Set up templates with the custom environment
templates = Jinja2Templates(env=env)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Explicitly provide all necessary context variables
    return templates.TemplateResponse("index.html", {
        "request": request,
    })

@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, dataset: UploadFile = File(...)):
    try:
        contents = await dataset.read()

        # Create a temporary in-memory file for processing
        file_stream = io.BytesIO(contents)
        file_stream.seek(0)

        # Extract the original filename
        original_filename = dataset.filename

        # Use the central pipeline to build everything
        state = build_dashboard_from_file(file_stream, original_filename=original_filename)

        if state is None:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error_message": "Failed to read CSV file. Please ensure the file is a valid CSV.",
                "success": False
            })

        df = state.df
        dataset_profile = state.dataset_profile
        profile = state.profile
        kpis = state.kpis
        charts = state.charts
        primary_chart = state.primary_chart
        category_charts = state.category_charts
        all_charts = state.all_charts
        eda_summary = state.eda_summary
        # Pass the original filename to the template
        state.original_filename = original_filename

        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "profile": profile,
                "dataset_profile": dataset_profile,
                "kpis": kpis,
                "charts": charts,
                "primary_chart": primary_chart,
                "category_charts": category_charts,
                "all_charts": all_charts,
                "eda_summary": eda_summary,
                "original_filename": original_filename,
                "success": True
            }
        )
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error_message": f"An error occurred while processing the file: {str(e)}",
            "success": False
        })

@app.post("/load_external", response_class=HTMLResponse)
async def load_external(request: Request, external_source: str = Form(...)):
    """Load a dataset from external source (URL or Kaggle)"""
    try:
        # Extract a name from the external source
        if external_source.startswith("http://") or external_source.startswith("https://"):
            # For URLs, extract the filename from the URL path
            import urllib.parse
            parsed_path = urllib.parse.urlparse(external_source).path
            original_filename = parsed_path.split('/')[-1] or external_source
            df = load_csv_from_url(external_source)
        else:
            # Treat as Kaggle dataset slug
            original_filename = external_source.split('/')[-1].replace('-', ' ').title() or "Kaggle Dataset"
            df = load_csv_from_kaggle(external_source)

        if df is None:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error_message": "Failed to load dataset from external source. Please check the URL or Kaggle dataset slug.",
                "success": False
            })

        state = build_dashboard_from_df(df)

        if state is None:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error_message": "Failed to build dashboard from external dataset.",
                "success": False
            })

        df = state.df
        dataset_profile = state.dataset_profile
        profile = state.profile
        kpis = state.kpis
        charts = state.charts
        primary_chart = state.primary_chart
        category_charts = state.category_charts
        all_charts = state.all_charts
        eda_summary = state.eda_summary
        # Set the original filename based on the source
        state.original_filename = original_filename

        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "profile": profile,
                "dataset_profile": dataset_profile,
                "kpis": kpis,
                "charts": charts,
                "primary_chart": primary_chart,
                "category_charts": category_charts,
                "all_charts": all_charts,
                "eda_summary": eda_summary,
                "original_filename": original_filename,
                "success": True
            }
        )
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error_message": f"An error occurred while loading the external dataset: {str(e)}",
            "success": False
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
```

## AUDIT_REPORT.md

```
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

```

## AUDIT_REPORT_QWEN.md

```
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

```

## templates/index.html

```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-Powered Dashboard Generator</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdn.jsdelivr.net/npm/daisyui@4.7.2/dist/full.min.css" rel="stylesheet" type="text/css" />
</head>
<body class="bg-base-200 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <div class="hero bg-base-100 rounded-box shadow-xl mb-8">
            <div class="hero-content text-center">
                <div class="max-w-2xl">
                    <h1 class="text-4xl md:text-5xl font-bold text-primary mb-2">AI-Powered Dashboard Generator</h1>
                    <p class="text-lg italic text-gray-600">AI Engine for Effortless Dashboards | Automated Intelligence for Every Dataset</p>
                    <p class="py-4">Upload a CSV file to start building a dashboard or choose an option below.</p>
                </div>
            </div>
        </div>

        <!-- Display potential error messages passed from backend -->
        {% if error_message %}
            <div class="alert alert-error shadow-lg mb-4">
                <div>
                    <svg xmlns="http://www.w3.org/2000/svg" class="stroke-current flex-shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                    <span>{{ error_message }}</span>
                </div>
            </div>
        {% endif %}

        {% if success_message %}
            <div class="alert alert-success shadow-lg mb-4">
                <div>
                    <svg xmlns="http://www.w3.org/2000/svg" class="stroke-current flex-shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                    <span>{{ success_message }}</span>
                </div>
            </div>
        {% endif %}

        <div class="card bg-base-100 shadow-xl mb-8">
            <div class="card-body">
                <h2 class="card-title">Quick Options</h2>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div class="flex items-center">
                        <input type="radio" name="option" class="radio checked:bg-blue-500" id="upload_option" checked />
                        <label for="upload_option" class="ml-2">Upload a CSV file to analyze</label>
                    </div>
                    <div class="flex items-center">
                        <input type="radio" name="option" class="radio checked:bg-blue-500" id="url_option" />
                        <label for="url_option" class="ml-2">Load from a public URL</label>
                    </div>
                    <div class="flex items-center">
                        <input type="radio" name="option" class="radio checked:bg-blue-500" id="kaggle_option" />
                        <label for="kaggle_option" class="ml-2">Load from a Kaggle dataset</label>
                    </div>
                </div>
            </div>
        </div>

        <!-- Existing CSV upload form -->
        <div class="card bg-base-100 shadow-xl mb-8">
            <div class="card-body">
                <h2 class="card-title">Upload CSV File</h2>
                <form action="/upload" method="post" enctype="multipart/form-data" class="space-y-4">
                    <div class="form-control w-full">
                        <label class="label">
                            <span class="label-text">Select a CSV file</span>
                        </label>
                        <input type="file" name="dataset" accept=".csv" class="file-input file-input-bordered w-full max-w-xs" />
                    </div>
                    <div class="form-control mt-6">
                        <button type="submit" class="btn btn-primary">Upload CSV</button>
                    </div>
                </form>
            </div>
        </div>

        <div class="divider">OR</div>

        <!-- New: load from external source (URL or Kaggle slug) -->
        <div class="card bg-base-100 shadow-xl">
            <div class="card-body">
                <h2 class="card-title">Load Dataset from External Source</h2>
                <p class="mb-4">Paste either:</p>
                <ul class="mb-4 list-disc pl-6">
                    <li>A direct CSV URL (e.g. a GitHub raw CSV link)</li>
                    <li>A Kaggle dataset slug (e.g. <code class="bg-base-300 p-1 rounded">umitka/global-youth-unemployment-dataset</code>)</li>
                </ul>

                <form action="/load_external" method="post" class="space-y-4">
                    <div class="form-control w-full">
                        <label class="label">
                            <span class="label-text">CSV URL or Kaggle slug</span>
                        </label>
                        <input
                            type="text"
                            name="external_source"
                            placeholder="Enter URL or Kaggle dataset slug"
                            class="input input-bordered w-full max-w-md"
                        />
                    </div>
                    <div class="form-control">
                        <button type="submit" class="btn btn-accent">Load Dataset</button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</body>
</html>

```

## templates/dashboard.html

```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-Powered Dashboard Generator</title>
    <!-- Load Plotly once from CDN for all charts -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdn.jsdelivr.net/npm/daisyui@4.7.2/dist/full.min.css" rel="stylesheet" type="text/css" />
</head>
<body class="bg-base-200 min-h-screen">
    <div class="navbar bg-base-100 shadow-md mb-4">
        <div class="flex-1 items-center">
            <a class="btn btn-ghost text-xl">AI Dashboard Generator</a>
            <!-- Display Dataset Name -->
            {% if original_filename %}
                <!-- Extract the name from the filename, removing extension and replacing underscores with spaces -->
                {% set name_parts = original_filename.split('.') %}
                {% set name_without_ext = name_parts[0] if name_parts|length > 0 else original_filename %}
                {% set formatted_name = name_without_ext|replace('_', ' ')|replace('-', ' ')|title %}
                <div class="badge badge-lg badge-info capitalize ml-4">{{ formatted_name }}</div>
            {% endif %}
        </div>
        <div class="flex-none gap-2">
            <!-- Back button -->
            <a href="/" class="btn btn-outline">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" class="inline-block w-5 h-5 stroke-current mr-2"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
                Back Home
            </a>
            <!-- Upload new dataset button -->
            <a href="/" class="btn btn-primary">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" class="inline-block w-5 h-5 stroke-current mr-2"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path></svg>
                Upload New Dataset
            </a>
        </div>
    </div>

    <div class="container mx-auto px-4 py-4">
        <div class="hero bg-base-100 rounded-box shadow-xl mb-6">
            <div class="hero-content text-center">
                <div class="max-w-2xl">
                    <h1 class="text-4xl md:text-5xl font-bold text-primary mb-2">AI-Powered Dashboard Generator</h1>
                    <p class="text-lg italic text-gray-600">AI Engine for Effortless Dashboards | Automated Intelligence for Every Dataset</p>
                </div>
            </div>
        </div>

        <!-- Section Selector Tabs -->
        <div class="tabs tabs-lifted tabs-lg mb-6">
            <button class="tab tab-active" onclick="showSection('dashboard')">Dashboard</button>
            <button class="tab" onclick="showSection('eda')">EDA Analysis</button>
            <button class="tab" onclick="showSection('visualizations')">Visualizations</button>
            <button class="tab" onclick="showSection('column_profiling')">Column Profiling</button>
        </div>

        <div class="flex flex-col lg:flex-row gap-6">
            <!-- Sidebar -->
            <div class="lg:w-1/4">
                <div class="card bg-base-100 shadow-xl mb-4">
                    <div class="card-body">
                        <h2 class="card-title">Dataset Summary</h2>
                        <ul class="space-y-2">
                            <li class="flex justify-between">
                                <span><strong>Total Rows:</strong></span>
                                <span>{{ dataset_profile.n_rows }}</span>
                            </li>
                            <li class="flex justify-between">
                                <span><strong>Total Columns:</strong></span>
                                <span>{{ dataset_profile.n_cols }}</span>
                            </li>
                            {% if dataset_profile.role_counts %}
                                <li class="flex justify-between">
                                    <span><strong>Numeric Columns:</strong></span>
                                    <span class="badge badge-primary">{{ dataset_profile.role_counts.numeric }}</span>
                                </li>
                                <li class="flex justify-between">
                                    <span><strong>Datetime Columns:</strong></span>
                                    <span class="badge badge-primary">{{ dataset_profile.role_counts.datetime }}</span>
                                </li>
                                <li class="flex justify-between">
                                    <span><strong>Categorical Columns:</strong></span>
                                    <span class="badge badge-primary">{{ dataset_profile.role_counts.categorical }}</span>
                                </li>
                                <li class="flex justify-between">
                                    <span><strong>Text Columns:</strong></span>
                                    <span class="badge badge-primary">{{ dataset_profile.role_counts.text }}</span>
                                </li>
                            {% endif %}
                        </ul>
                    </div>
                </div>

                <!-- KPIs (clickable pills) -->
                <div class="card bg-base-100 shadow-xl">
                    <div class="card-body">
                        <h2 class="card-title">KPIs</h2>
                        <div class="space-y-2">
                            {% if kpis and kpis|length > 0 %}
                                {% for kpi in kpis %}
                                    {% set chart_obj = category_charts.get(kpi.label) %}
                                    <div
                                        class="badge badge-outline {% if chart_obj %}badge-primary cursor-pointer{% else %}badge-neutral opacity-50{% endif %}"
                                        data-kpi-column="{{ kpi.label }}"
                                        data-has-chart="{{ '1' if chart_obj else '0' }}"
                                        title="{% if chart_obj %}Click to view a chart for this column{% else %}No chart available yet for this KPI{% endif %}"
                                    >
                                        <strong>{{ kpi.label }}</strong>: {{ kpi.value }}
                                    </div>
                                {% endfor %}
                            {% else %}
                                <div>No KPIs generated yet.</div>
                            {% endif %}
                        </div>
                    </div>
                </div>
            </div>

            <!-- Main Content -->
            <div class="lg:w-3/4">
                <!-- Dashboard Section -->
                <div id="dashboard-section" class="analysis-section">
                    <div class="card bg-base-100 shadow-xl">
                        <div class="card-body">
                            <h2 class="card-title">Dashboard Visualizations</h2>
                            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <!-- Category Count Charts -->
                                {% if category_charts and category_charts|length > 0 %}
                                    {% for col_name, chart_data in category_charts.items() %}
                                        {% if primary_chart and chart_data.column != primary_chart.column %}
                                            <div class="card bg-base-100 shadow-md">
                                                <div class="card-body p-4">
                                                    <h4 class="text-md font-medium text-center">{{ chart_data.title or col_name|title + " Distribution" }}</h4>
                                                    <div class="chart-container h-80" id="chart-{{ col_name|replace(' ', '_')|replace('.', '_') }}"></div>
                                                </div>
                                            </div>
                                        {% elif not primary_chart %}
                                            <div class="card bg-base-100 shadow-md">
                                                <div class="card-body p-4">
                                                    <h4 class="text-md font-medium text-center">{{ chart_data.title or col_name|title + " Distribution" }}</h4>
                                                    <div class="chart-container h-80" id="chart-{{ col_name|replace(' ', '_')|replace('.', '_') }}"></div>
                                                </div>
                                            </div>
                                        {% endif %}
                                    {% endfor %}
                                {% endif %}

                                <!-- Other Chart Types -->
                                {% if all_charts and all_charts|length > 0 %}
                                    {% for chart in all_charts %}
                                        {% if chart is defined and chart.data is defined %}
                                            <div class="card bg-base-100 shadow-md">
                                                <div class="card-body p-4">
                                                    <h4 class="text-md font-medium text-center">{{ chart.title or chart.type|title + " Chart" }}</h4>
                                                    <div class="chart-container h-80" id="chart_{{ loop.index }}"></div>
                                                </div>
                                            </div>
                                        {% endif %}
                                    {% endfor %}
                                {% endif %}

                                <!-- Show message if no charts are available -->
                                {% if not category_charts and not all_charts %}
                                    <div class="alert alert-warning">
                                        <div class="flex-1">
                                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" class="w-6 h-6 mx-2 stroke-current"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z"></path></svg>
                                            <span>No charts available for this dataset yet.</span>
                                        </div>
                                    </div>
                                {% elif category_charts|length == 0 and all_charts|length == 0 %}
                                    <div class="alert alert-warning">
                                        <div class="flex-1">
                                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" class="w-6 h-6 mx-2 stroke-current"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z"></path></svg>
                                            <span>No charts available for this dataset yet.</span>
                                        </div>
                                    </div>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                </div>

                <!-- EDA Analysis Section -->
                <div id="eda-section" class="analysis-section hidden">
                    <div class="card bg-base-100 shadow-xl">
                        <div class="card-body">
                            <h2 class="card-title">Exploratory Data Analysis</h2>
                            
                            {% if eda_summary %}
                                <!-- Key Indicators -->
                                <div class="card bg-base-200 shadow-sm mb-4">
                                    <div class="card-body p-4">
                                        <h3 class="text-lg font-semibold mb-2">Key Indicators</h3>
                                        {% if eda_summary.key_indicators %}
                                            {% for indicator in eda_summary.key_indicators[:10] %}  <!-- Show top 10 indicators -->
                                                <div class="badge badge-outline badge-success mb-2">
                                                    <strong>{{ indicator.indicator }}</strong> ({{ indicator.indicator_type }}) - 
                                                    Significance: {{ "%.2f"|format(indicator.significance_score) }}
                                                    <div class="tooltip tooltip-bottom" data-tip="{{ indicator.description }}">
                                                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" class="w-4 h-4 ml-1 stroke-current"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                                                    </div>
                                                </div>
                                            {% endfor %}
                                        {% else %}
                                            <p>No key indicators identified.</p>
                                        {% endif %}
                                    </div>
                                </div>

                                <!-- Use Cases -->
                                <div class="card bg-base-200 shadow-sm mb-4">
                                    <div class="card-body p-4">
                                        <h3 class="text-lg font-semibold mb-2">Potential Use Cases</h3>
                                        {% if eda_summary.use_cases %}
                                            {% for use_case in eda_summary.use_cases %}
                                                <div class="alert alert-info mb-3">
                                                    <div class="flex-1">
                                                        <strong>{{ use_case.use_case }}</strong><br>
                                                        <span class="text-sm">{{ use_case.description }}</span><br>
                                                        <span class="text-xs"><strong>Key Inputs:</strong> {{ use_case.key_inputs|join(', ') }}</span><br>
                                                        <span class="text-xs"><strong>Key Indicators:</strong> {{ use_case.key_indicators|join(', ') if use_case.key_indicators else 'N/A' }}</span>
                                                    </div>
                                                </div>
                                            {% endfor %}
                                        {% else %}
                                            <p>No specific use cases detected.</p>
                                        {% endif %}
                                    </div>
                                </div>

                                <!-- Patterns and Relationships -->
                                <div class="card bg-base-200 shadow-sm mb-4">
                                    <div class="card-body p-4">
                                        <h3 class="text-lg font-semibold mb-2">Patterns and Relationships</h3>
                                        {% if eda_summary.patterns_and_relationships %}
                                            {% if eda_summary.patterns_and_relationships.correlations %}
                                                <div class="mb-3">
                                                    <h4 class="font-medium">Correlations</h4>
                                                    <ul class="list-disc pl-5 text-sm">
                                                        {% for corr in eda_summary.patterns_and_relationships.correlations[:5] %}  <!-- Top 5 correlations -->
                                                            <li>
                                                                <strong>{{ corr.variable1 }}</strong> ↔ <strong>{{ corr.variable2 }}</strong>: 
                                                                {{ "%.3f"|format(corr.correlation) }} ({{ corr.strength }} {{ corr.type }} correlation)
                                                            </li>
                                                        {% endfor %}
                                                    </ul>
                                                </div>
                                            {% endif %}
                                            
                                            {% if eda_summary.patterns_and_relationships.trends %}
                                                <div class="mb-3">
                                                    <h4 class="font-medium">Trends</h4>
                                                    <ul class="list-disc pl-5 text-sm">
                                                        {% for trend in eda_summary.patterns_and_relationships.trends[:5] %}  <!-- Top 5 trends -->
                                                            <li>
                                                                <strong>{{ trend.datetime_column }}</strong> → <strong>{{ trend.numeric_column }}</strong>: 
                                                                {{ trend.trend_type }} trend (correlation: {{ "%.3f"|format(trend.trend_correlation) }})
                                                            </li>
                                                        {% endfor %}
                                                    </ul>
                                                </div>
                                            {% endif %}
                                            
                                            {% if eda_summary.patterns_and_relationships.outliers %}
                                                <div class="mb-3">
                                                    <h4 class="font-medium">Outliers</h4>
                                                    <ul class="list-disc pl-5 text-sm">
                                                        {% for outlier in eda_summary.patterns_and_relationships.outliers[:5] %}  <!-- Top 5 outlier columns -->
                                                            <li>
                                                                <strong>{{ outlier.column }}</strong>: {{ outlier.outlier_count }} outliers ({{ "%.2f"|format(outlier.outlier_percentage) }}%)
                                                            </li>
                                                        {% endfor %}
                                                    </ul>
                                                </div>
                                            {% endif %}
                                        {% else %}
                                            <p>No patterns identified.</p>
                                        {% endif %}
                                    </div>
                                </div>

                                <!-- Recommendations -->
                                <div class="card bg-base-200 shadow-sm mb-4">
                                    <div class="card-body p-4">
                                        <h3 class="text-lg font-semibold mb-2">Recommendations</h3>
                                        {% if eda_summary.recommendations %}
                                            {% for rec in eda_summary.recommendations %}
                                                <div class="alert alert-warning mb-3">
                                                    <div class="flex-1">
                                                        <strong>{{ rec.title }}</strong><br>
                                                        <span class="text-sm">{{ rec.description }}</span>
                                                    </div>
                                                </div>
                                            {% endfor %}
                                        {% else %}
                                            <p>No specific recommendations at this time.</p>
                                        {% endif %}
                                    </div>
                                </div>
                            {% else %}
                                <p>EDA analysis not available for this dataset.</p>
                            {% endif %}
                        </div>
                    </div>
                </div>

                <!-- Visualizations Section -->
                <div id="visualizations-section" class="analysis-section hidden">
                    <div class="card bg-base-100 shadow-xl">
                        <div class="card-body">
                            <h2 class="card-title">Advanced Visualizations</h2>
                            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                                {% if eda_summary and eda_summary.patterns_and_relationships %}
                                    {% if eda_summary.patterns_and_relationships.correlations %}
                                        <div class="card bg-base-100 shadow-md">
                                            <div class="card-body p-4">
                                                <h4 class="text-md font-medium text-center">Correlation Heatmap</h4>
                                                <div class="chart-container h-80" id="correlation-heatmap"></div>
                                            </div>
                                        </div>
                                    {% endif %}
                                    
                                    {% if eda_summary.key_indicators %}
                                        <div class="card bg-base-100 shadow-md">
                                            <div class="card-body p-4">
                                                <h4 class="text-md font-medium text-center">Key Indicators</h4>
                                                <div class="chart-container h-80" id="key-indicators-chart"></div>
                                            </div>
                                        </div>
                                    {% endif %}
                                    
                                    {% if eda_summary.patterns_and_relationships.trends %}
                                        <div class="card bg-base-100 shadow-md">
                                            <div class="card-body p-4">
                                                <h4 class="text-md font-medium text-center">Time Series Trends</h4>
                                                <div class="chart-container h-80" id="trends-chart"></div>
                                            </div>
                                        </div>
                                    {% endif %}
                                    
                                    {% if eda_summary.patterns_and_relationships.outliers %}
                                        <div class="card bg-base-100 shadow-md">
                                            <div class="card-body p-4">
                                                <h4 class="text-md font-medium text-center">Outlier Detection</h4>
                                                <div class="chart-container h-80" id="outliers-chart"></div>
                                            </div>
                                        </div>
                                    {% endif %}
                                    
                                    {% if eda_summary.use_cases %}
                                        <div class="card bg-base-100 shadow-md">
                                            <div class="card-body p-4">
                                                <h4 class="text-md font-medium text-center">Use Cases Overview</h4>
                                                <div class="chart-container h-80" id="use-cases-chart"></div>
                                            </div>
                                        </div>
                                    {% endif %}
                                {% endif %}
                                
                                {% if not eda_summary or not eda_summary.patterns_and_relationships %}
                                    <div class="alert alert-warning">
                                        <div class="flex-1">
                                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" class="w-6 h-6 mx-2 stroke-current"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z"></path></svg>
                                            <span>Advanced visualizations not available for this dataset.</span>
                                        </div>
                                    </div>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Column Profiling Section -->
                <div id="column_profiling-section" class="analysis-section hidden">
                    <div class="card bg-base-100 shadow-xl">
                        <div class="card-body">
                            <h2 class="card-title">Column Profiling</h2>
                            <div class="overflow-x-auto">
                                <table class="table table-zebra">
                                    <thead>
                                        <tr>
                                            <th>Column Name</th>
                                            <th>Data Type</th>
                                            <th>Missing Values</th>
                                            <th>Unique Values</th>
                                            <th>Role</th>
                                            <th>Min</th>
                                            <th>Max</th>
                                            <th>Mean</th>
                                            <th>Top Categories</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {% for col in dataset_profile.columns %}
                                        <tr>
                                            <td>{{ col.name }}</td>
                                            <td>{{ col.dtype }}</td>
                                            <td>{{ col.missing_count }}</td>
                                            <td>{{ col.unique_count }}</td>
                                            <td>{{ col.role }}</td>
                                            <td>
                                                {% if col.stats and col.stats.min is not none %}
                                                    {{ col.stats.min }}
                                                {% else %}
                                                    -
                                                {% endif %}
                                            </td>
                                            <td>
                                                {% if col.stats and col.stats.max is not none %}
                                                    {{ col.stats.max }}
                                                {% else %}
                                                    -
                                                {% endif %}
                                            </td>
                                            <td>
                                                {% if col.stats and col.stats.mean is not none %}
                                                    {{ col.stats.mean }}
                                                {% else %}
                                                    -
                                                {% endif %}
                                            </td>
                                            <td>
                                                {% if col.top_categories and col.top_categories|length > 0 %}
                                                    {% for cat in col.top_categories %}
                                                        {{ cat.value }} ({{ cat.count }})
                                                        {% if not loop.last %}, {% endif %}
                                                    {% endfor %}
                                                {% else %}
                                                    -
                                                {% endif %}
                                            </td>
                                        </tr>
                                        {% endfor %}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Precomputed category charts from the backend (data only, no HTML)
        const CATEGORY_CHARTS = {{ category_charts | tojson | safe }};
        const PRIMARY_CHART = {% if primary_chart %}{{ primary_chart | tojson | safe }}{% else %}null{% endif %};
        const ALL_CHARTS = {{ all_charts | tojson | safe }};
        const EDA_SUMMARY = {{ eda_summary | tojson | safe }};

        // Helper function to get the primary chart column name
        const PRIMARY_CHART_COLUMN = PRIMARY_CHART ? PRIMARY_CHART.column : null;

        function showSection(sectionName) {
            // Hide all sections
            document.querySelectorAll('.analysis-section').forEach(section => {
                section.classList.add('hidden');
                section.classList.remove('active');
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('tab-active');
            });
            
            // Show selected section
            const sectionElement = document.getElementById(sectionName + '-section');
            if (sectionElement) {
                sectionElement.classList.remove('hidden');
                sectionElement.classList.add('active');
            }
            
            // Add active class to clicked tab
            event.target.classList.add('tab-active');
            
            // If switching to visualizations section, render the charts
            if (sectionName === 'visualizations' && EDA_SUMMARY) {
                setTimeout(renderAdvancedVisualizations, 100);  // Delay to ensure DOM is ready
            }
        }

        function loadChartForColumn(columnName) {
            const chart = CATEGORY_CHARTS[columnName];
            if (!chart) {
                // No precomputed chart for this column – silently ignore
                return;
            }

            // Find the corresponding chart container and scroll to it
            const containerId = "chart-" + columnName.replace(/[\s.]/g, '_');
            const chartContainer = document.getElementById(containerId);
            if (chartContainer) {
                chartContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                // Add temporary highlight
                chartContainer.style.border = '3px solid #3b82f6'; // Tailwind blue-500
                setTimeout(() => {
                    chartContainer.style.border = '';
                }, 3000);
            }
        }

        // Function to render advanced visualizations based on EDA summary
        function renderAdvancedVisualizations() {
            if (!EDA_SUMMARY) {
                console.warn("No EDA summary available for visualization");
                return;
            }

            // Render correlation heatmap if correlations exist
            if (EDA_SUMMARY.patterns_and_relationships && EDA_SUMMARY.patterns_and_relationships.correlations) {
                renderCorrelationHeatmap();
            }

            // Render key indicators bar chart if key indicators exist
            if (EDA_SUMMARY.key_indicators && EDA_SUMMARY.key_indicators.length > 0) {
                renderKeyIndicatorsChart();
            }

            // Render trends chart if trends exist
            if (EDA_SUMMARY.patterns_and_relationships && EDA_SUMMARY.patterns_and_relationships.trends) {
                renderTrendsChart();
            }

            // Render outliers chart if outliers exist
            if (EDA_SUMMARY.patterns_and_relationships && EDA_SUMMARY.patterns_and_relationships.outliers) {
                renderOutliersChart();
            }

            // Render use cases chart if use cases exist
            if (EDA_SUMMARY.use_cases && EDA_SUMMARY.use_cases.length > 0) {
                renderUseCasesChart();
            }
        }

        // Render correlation heatmap
        function renderCorrelationHeatmap() {
            const correlations = EDA_SUMMARY.patterns_and_relationships.correlations;
            if (!correlations || correlations.length === 0) return;

            // Extract unique variables
            const variables = new Set();
            correlations.forEach(corr => {
                variables.add(corr.variable1);
                variables.add(corr.variable2);
            });
            const varArray = Array.from(variables).slice(0, 10); // Limit to 10 variables for readability

            // Create correlation matrix
            const matrix = Array(varArray.length).fill().map(() => Array(varArray.length).fill(0));
            const varMap = {};
            varArray.forEach((varName, idx) => {
                varMap[varName] = idx;
            });

            correlations.forEach(corr => {
                if (varMap.hasOwnProperty(corr.variable1) && varMap.hasOwnProperty(corr.variable2)) {
                    const i = varMap[corr.variable1];
                    const j = varMap[corr.variable2];
                    matrix[i][j] = corr.correlation;
                    matrix[j][i] = corr.correlation; // Symmetric
                }
            });

            const trace = {
                z: matrix,
                x: varArray,
                y: varArray,
                type: 'heatmap',
                colorscale: 'RdBu',
                zmid: 0,
                text: matrix.map(row => row.map(val => val.toFixed(2))),
                texttemplate: "%{text}",
                textfont: { size: 12 }
            };

            const layout = {
                title: 'Correlation Heatmap (Top Variables)',
                xaxis: { title: 'Variables', automargin: true },
                yaxis: { title: 'Variables', automargin: true },
                margin: { t: 50, l: 100, r: 50, b: 100 }
            };

            Plotly.newPlot('correlation-heatmap', [trace], layout);
        }

        // Render key indicators chart
        function renderKeyIndicatorsChart() {
            const indicators = EDA_SUMMARY.key_indicators.slice(0, 10); // Top 10 indicators
            if (!indicators || indicators.length === 0) return;

            const names = indicators.map(ind => ind.indicator);
            const scores = indicators.map(ind => ind.significance_score);
            const types = indicators.map(ind => ind.indicator_type);

            // Color mapping
            const typeColors = {
                'numeric': '#1f77b4',
                'categorical': '#ff7f0e',
                'datetime': '#2ca02c'
            };
            const colors = types.map(type => typeColors[type] || '#7f7f7f');

            const trace = {
                x: scores,
                y: names,
                type: 'bar',
                orientation: 'h',
                marker: { color: colors }
            };

            const layout = {
                title: 'Top Key Indicators by Significance Score',
                xaxis: { title: 'Significance Score', automargin: true },
                yaxis: { title: 'Indicator', automargin: true },
                margin: { t: 50, l: 150, r: 50, b: 50 }
            };

            Plotly.newPlot('key-indicators-chart', [trace], layout);
        }

        // Render trends chart
        function renderTrendsChart() {
            const trends = EDA_SUMMARY.patterns_and_relationships.trends;
            if (!trends || trends.length === 0) return;

            // Count trend types
            const trendCounts = {
                increasing: 0,
                decreasing: 0,
                stable: 0
            };
            trends.forEach(trend => {
                trendCounts[trend.trend_type]++;
            });

            const trace = {
                labels: Object.keys(trendCounts),
                values: Object.values(trendCounts),
                type: 'pie'
            };

            const layout = {
                title: 'Distribution of Time Series Trends',
                margin: { t: 50, l: 50, r: 50, b: 50 }
            };

            Plotly.newPlot('trends-chart', [trace], layout);
        }

        // Render outliers chart
        function renderOutliersChart() {
            const outliers = EDA_SUMMARY.patterns_and_relationships.outliers.slice(0, 10); // Top 10 outlier columns
            if (!outliers || outliers.length === 0) return;

            const names = outliers.map(out => out.column);
            const outlierCounts = outliers.map(out => out.outlier_count);

            const trace = {
                x: names,
                y: outlierCounts,
                type: 'bar',
                marker: { color: '#d62728' }
            };

            const layout = {
                title: 'Outlier Counts by Column (Top 10)',
                xaxis: { title: 'Column', automargin: true, tickangle: -45 },
                yaxis: { title: 'Outlier Count', automargin: true },
                margin: { t: 50, l: 50, r: 50, b: 100 }
            };

            Plotly.newPlot('outliers-chart', [trace], layout);
        }

        // Render use cases chart
        function renderUseCasesChart() {
            const useCases = EDA_SUMMARY.use_cases;
            if (!useCases || useCases.length === 0) return;

            const names = useCases.map(uc => uc.use_case.substring(0, 20) + (uc.use_case.length > 20 ? '...' : ''));  // Truncate long names
            const keyInputCounts = useCases.map(uc => uc.key_inputs ? uc.key_inputs.length : 0);

            const trace = {
                x: names,
                y: keyInputCounts,
                type: 'bar',
                marker: { color: '#17becf' }
            };

            const layout = {
                title: 'Use Cases and Key Inputs',
                xaxis: { title: 'Use Case', automargin: true, tickangle: -45 },
                yaxis: { title: 'Number of Key Inputs', automargin: true },
                margin: { t: 50, l: 50, r: 50, b: 100 }
            };

            Plotly.newPlot('use-cases-chart', [trace], layout);
        }

        // This function renders category count charts specifically
        function renderCategoryCountChart(spec, containerId) {
            try {
                if (!spec || !spec.data) {
                    console.warn("Invalid category chart data for container:", containerId, spec);
                    renderEmptyChart(containerId, "Invalid chart data");
                    return;
                }

                if (!Array.isArray(spec.data) || spec.data.length === 0) {
                    console.warn("Empty or invalid category chart data for container:", containerId, spec);
                    renderEmptyChart(containerId, "No chart data available");
                    return;
                }

                // Validate that required fields exist in the data
                const validData = spec.data.filter(item =>
                    item.category !== undefined && item.count !== undefined
                );

                if (validData.length === 0) {
                    console.warn("No valid data points for category chart in container:", containerId, spec);
                    renderEmptyChart(containerId, "No valid data points");
                    return;
                }

                const categories = validData.map(row => String(row.category));
                const values = validData.map(row => Number(row.count));

                // Validate that we have numeric values to plot
                const numericValues = values.filter(val => !isNaN(val) && isFinite(val));
                if (numericValues.length === 0) {
                    console.warn("No valid numeric values to plot for category chart in container:", containerId, spec);
                    renderEmptyChart(containerId, "No valid values to plot");
                    return;
                }

                const maxVal = Math.max(...numericValues, 0);
                const avgLabelLen = categories.reduce((sum, c) => sum + c.length, 0) / categories.length;
                const manyCategories = categories.length > 6;
                const longLabels = avgLabelLen > 12;
                const useHorizontal = manyCategories || longLabels;

                const title = spec.title || `Count of ${spec.column || ""}`;

                let trace, layout;

                if (useHorizontal) {
                    trace = {
                        type: "bar",
                        x: numericValues,
                        y: categories,
                        orientation: "h",
                        text: numericValues,
                        textposition: "outside"
                    };

                    layout = {
                        xaxis: {
                            title: "Count",
                            tickformat: ",d",
                            exponentformat: "none",
                            rangemode: "tozero",
                            range: [0, maxVal * 1.15 || 1],
                            automargin: true
                        },
                        yaxis: {
                            title: spec.column || "",
                            categoryorder: "array",
                            categoryarray: categories,
                            automargin: true
                        },
                        margin: {
                            t: 20,  // Very small top margin since we're removing title
                            b: 60,  // Keep bottom margin for labels
                            l: 80,  // Reduced left margin
                            r: 20   // Reduced right margin
                        },
                        height: 320  // Explicit height to match container
                    };
                } else {
                    trace = {
                        type: "bar",
                        x: categories,
                        y: numericValues,
                        text: numericValues,
                        textposition: "outside"
                    };

                    layout = {
                        xaxis: {
                            title: spec.column || "",
                            categoryorder: "array",
                            categoryarray: categories,
                            automargin: true
                        },
                        yaxis: {
                            title: "Count",
                            tickformat: ",d",
                            exponentformat: "none",
                            rangemode: "tozero",
                            range: [0, maxVal * 1.15 || 1],
                            automargin: true
                        },
                        margin: {
                            t: 20,  // Very small top margin since we're removing title
                            b: 80,  // Keep bottom margin for labels
                            l: 60,  // Reduced left margin
                            r: 20   // Reduced right margin
                        },
                        height: 320  // Explicit height to match container
                    };
                }

                // Check if the container element exists before rendering
                const containerElement = document.getElementById(containerId);
                if (!containerElement) {
                    console.warn(`Chart container element with ID '${containerId}' not found.`);
                    return;
                }

                Plotly.react(containerId, [trace], layout);
            } catch (error) {
                console.error(`Error rendering category count chart in container '${containerId}':`, error);
                renderEmptyChart(containerId, "Chart rendering error");
            }
        }

        // Validate chart data and ensure it's in the expected format
        function validateChartData(chartData) {
            if (!chartData) return false;
            if (!chartData.data) return false;

            // Different chart types have different expected data structures
            if (chartData.intent === 'correlation') {
                return chartData.data.categories && Array.isArray(chartData.data.values);
            }

            if (Array.isArray(chartData.data)) {
                if (chartData.data.length === 0) return false;

                // Box plots: array of objects with 'category' and 'values' properties
                if (chartData.intent === 'box_plot' || (chartData.data[0].hasOwnProperty('values'))) {
                    return chartData.data[0].hasOwnProperty('category') && Array.isArray(chartData.data[0].values);
                }

                // Scatter plots: array of objects with 'x' and 'y' properties
                if (chartData.data[0].hasOwnProperty('x') && chartData.data[0].hasOwnProperty('y')) {
                    return true;
                }

                // Time series: array of objects with 'date' and 'value' properties
                if (chartData.data[0].hasOwnProperty('date') && chartData.data[0].hasOwnProperty('value')) {
                    return true;
                }

                // Pie/bar charts: array of objects with 'category' and either 'count' or 'value' properties
                if (chartData.data[0].hasOwnProperty('category') && (chartData.data[0].hasOwnProperty('value') || chartData.data[0].hasOwnProperty('count'))) {
                    return true;
                }
            }

            return false;
        }

        // Validate chart data and ensure it's in the expected format
        function validateChartData(chartData) {
            if (!chartData) return false;
            if (!chartData.data) return false;

            // Handle the new simple renderer format where chartData has a 'type' field
            if (chartData.type) {
                // New simple renderer format: { type: string, data: array, title: string, ... }
                return Array.isArray(chartData.data);
            }

            // Original format handling
            if (chartData.intent === 'correlation') {
                return chartData.data.categories && Array.isArray(chartData.data.values);
            }

            if (Array.isArray(chartData.data)) {
                if (chartData.data.length === 0) return true; // Empty array is valid

                // Box plots: array of objects with 'category' and 'values' properties
                if (chartData.intent === 'box_plot' || (chartData.data[0].hasOwnProperty('values'))) {
                    return chartData.data[0].hasOwnProperty('category') && Array.isArray(chartData.data[0].values);
                }

                // Scatter plots: array of objects with 'x' and 'y' properties
                if (chartData.data[0].hasOwnProperty('x') && chartData.data[0].hasOwnProperty('y')) {
                    return true;
                }

                // Time series: array of objects with 'date' and 'value' properties
                if (chartData.data[0].hasOwnProperty('date') && chartData.data[0].hasOwnProperty('value')) {
                    return true;
                }

                // Pie/bar charts: array of objects with 'category' and either 'count' or 'value' properties
                if (chartData.data[0].hasOwnProperty('category') && (chartData.data[0].hasOwnProperty('value') || chartData.data[0].hasOwnProperty('count'))) {
                    return true;
                }
            }

            return false;
        }

        // This function renders different chart types based on their structure
        function renderChartByType(chartData, containerId) {
            try {
                if (!validateChartData(chartData)) {
                    console.warn("Invalid chart data for container:", containerId, chartData);
                    // Render an empty chart with error message
                    renderEmptyChart(containerId, "Invalid chart data");
                    return;
                }

                // Check if the container element exists before rendering
                const containerElement = document.getElementById(containerId);
                if (!containerElement) {
                    console.warn(`Chart container element with ID '${containerId}' not found.`);
                    return;
                }

                // Handle the new simple renderer format (has 'type' field)
                if (chartData.type) {
                    switch(chartData.type) {
                        case 'bar':
                            _renderSimpleBarChart(chartData, containerId);
                            break;
                        case 'line':
                            _renderSimpleLineChart(chartData, containerId);
                            break;
                        case 'scatter':
                            _renderSimpleScatterChart(chartData, containerId);
                            break;
                        case 'histogram':
                            _renderSimpleHistogramChart(chartData, containerId);
                            break;
                        case 'pie':
                            _renderSimplePieChart(chartData, containerId);
                            break;
                        default:
                            _renderBarChart(chartData, containerId);
                            break;
                    }
                    return;
                }

                // Handle original format
                if (chartData.data.categories && Array.isArray(chartData.data.values)) {
                    // This is a correlation matrix
                    _renderCorrelationChart(chartData, containerId);
                }
                else if (Array.isArray(chartData.data) && chartData.data.length > 0 && chartData.data[0].hasOwnProperty('values')) {
                    // This is a box plot
                    _renderBoxPlot(chartData, containerId);
                }
                else if (Array.isArray(chartData.data) && chartData.data.length > 0 && chartData.data[0].hasOwnProperty('x') && chartData.data[0].hasOwnProperty('y')) {
                    // This is a scatter plot
                    _renderScatterPlot(chartData, containerId);
                }
                else if (Array.isArray(chartData.data) && chartData.data.length > 0 && chartData.data[0].date !== undefined) {
                    // This is a time series chart
                    _renderTimeSeriesChart(chartData, containerId);
                }
                else if (Array.isArray(chartData.data) && chartData.data.length > 0 && chartData.data[0].hasOwnProperty('category') && (chartData.data[0].hasOwnProperty('value') || chartData.data[0].hasOwnProperty('count'))) {
                    // This could be pie or bar chart
                    if (chartData.intent && chartData.intent === 'category_pie') {
                        _renderPieChart(chartData, containerId);
                    } else {
                        _renderBarChart(chartData, containerId);
                    }
                }
                else {
                    // Fallback to bar chart if the data format is unexpected
                    console.warn("Unexpected chart data format for container:", containerId, chartData);
                    _renderBarChart(chartData, containerId);
                }
            } catch (error) {
                console.error(`Error rendering chart by type in container '${containerId}':`, error);
                renderEmptyChart(containerId, "Chart rendering error");
            }
        }

        // Function to render correlation matrix
        function _renderCorrelationChart(chartData, containerId) {
            if (!chartData || !chartData.data || !Array.isArray(chartData.data.values) || !Array.isArray(chartData.data.categories)) {
                console.warn("Invalid correlation matrix data for container:", containerId, chartData);
                return;
            }

            const zData = chartData.data.values;
            const xLabels = chartData.data.categories;
            const yLabels = [...xLabels]; // Same for correlation matrix

            // Validate data dimensions
            if (!Array.isArray(zData) || !Array.isArray(xLabels) || zData.length !== xLabels.length) {
                console.warn("Invalid correlation matrix dimensions for container:", containerId, chartData);
                return;
            }

            const trace = {
                z: zData,
                x: xLabels,
                y: yLabels,
                type: 'heatmap',
                colorscale: 'Viridis',
                text: zData.map(row => row.map(val => {
                    // Ensure valid number for display
                    return (typeof val === 'number' && !isNaN(val) && isFinite(val)) ? val.toFixed(2) : 'N/A';
                })),
                texttemplate: "%{text}",
                textfont: { size: 12 },
                hoverongaps: false,
                hovertemplate:
                    'X: %{x}<br>' +
                    'Y: %{y}<br>' +
                    'Correlation: %{z}<br>' +
                    '<extra></extra>',
            };

            const layout = {
                title: chartData.title || "Correlation Matrix",
                xaxis: { title: "Variables", automargin: true },
                yaxis: { title: "Variables", automargin: true },
                margin: {
                    t: 60,
                    b: 80,
                    l: 80,
                    r: 40
                },
                height: 500
            };

            Plotly.react(containerId, [trace], layout);
        }

        // Function to render box plots
        function _renderBoxPlot(chartData, containerId) {
            if (!chartData.data || !Array.isArray(chartData.data) || chartData.data.length === 0) {
                console.warn("Invalid box plot data for container:", containerId, chartData);
                return;
            }

            // Validate data structure
            const validData = chartData.data.filter(item =>
                item.category !== undefined && Array.isArray(item.values) && item.values.length > 0
            );

            if (validData.length === 0) {
                console.warn("No valid box plot data points for container:", containerId, chartData);
                return;
            }

            const traces = validData.map(item => {
                // Filter out non-numeric values for the box plot
                const numericValues = item.values.filter(val => typeof val === 'number' && !isNaN(val) && isFinite(val));

                return {
                    y: numericValues,
                    type: 'box',
                    name: String(item.category),
                    boxpoints: 'outliers'  // Show outliers
                };
            });

            const layout = {
                title: chartData.title || "Box Plot",
                xaxis: { title: chartData.x_column || "Category", automargin: true },
                yaxis: { title: chartData.y_column || "Value", automargin: true },
                margin: {
                    t: 60,
                    b: 80,
                    l: 60,
                    r: 40
                },
                showlegend: false
            };

            Plotly.react(containerId, traces, layout);
        }

        // Function to render an empty chart with error message
        function renderEmptyChart(containerId, message) {
            const container = document.getElementById(containerId);
            if (container) {
                container.innerHTML = `<div class="flex items-center justify-center h-full text-gray-500 italic">${message}</div>`;
            }
        }

        // Function to render simple bar charts from the new renderer
        function _renderSimpleBarChart(chartData, containerId) {
            if (!chartData.data || chartData.data.length === 0) {
                renderEmptyChart(containerId, "No data available");
                return;
            }

            // Filter valid data points with x and y values
            const validData = chartData.data.filter(d =>
                d.x !== undefined && d.y !== undefined &&
                d.x !== null && d.y !== null
            );

            if (validData.length === 0) {
                renderEmptyChart(containerId, "No valid data points");
                return;
            }

            // Prepare the x and y values
            const xValues = validData.map(d => String(d.x));
            const yValues = validData.map(d => {
                const val = parseFloat(d.y);
                return isNaN(val) ? 0 : val;
            });

            // Determine orientation based on label length and count
            const avgLabelLen = xValues.reduce((sum, label) => sum + label.length, 0) / xValues.length;
            const manyCategories = xValues.length > 6;
            const longLabels = avgLabelLen > 12;
            const useHorizontal = manyCategories || longLabels;

            let trace, layout;

            if (useHorizontal) {
                trace = {
                    x: yValues,
                    y: xValues,
                    type: 'bar',
                    orientation: 'h',
                    text: yValues,
                    textposition: 'auto'
                };

                layout = {
                    title: chartData.title || "Bar Chart",
                    xaxis: { title: "Value", automargin: true },
                    yaxis: { title: chartData.x_col || "Category", automargin: true },
                    margin: {
                        t: 40,
                        b: 60,
                        l: 100,
                        r: 40
                    }
                };
            } else {
                trace = {
                    x: xValues,
                    y: yValues,
                    type: 'bar',
                    text: yValues,
                    textposition: 'auto'
                };

                layout = {
                    title: chartData.title || "Bar Chart",
                    xaxis: { title: chartData.x_col || "Category", automargin: true },
                    yaxis: { title: chartData.y_col || "Value", automargin: true },
                    margin: {
                        t: 40,
                        b: 80,
                        l: 60,
                        r: 40
                    }
                };
            }

            Plotly.react(containerId, [trace], layout);
        }

        // Function to render simple line charts from the new renderer
        function _renderSimpleLineChart(chartData, containerId) {
            if (!chartData.data || chartData.data.length === 0) {
                renderEmptyChart(containerId, "No data available");
                return;
            }

            // Filter valid data points with x and y values
            const validData = chartData.data.filter(d =>
                d.x !== undefined && d.y !== undefined &&
                d.x !== null && d.y !== null
            );

            if (validData.length === 0) {
                renderEmptyChart(containerId, "No valid data points");
                return;
            }

            // Prepare the x and y values
            const xValues = validData.map(d => String(d.x));  // Keep as strings for x-axis
            const yValues = validData.map(d => {
                const val = parseFloat(d.y);
                return isNaN(val) ? 0 : val;
            });

            const trace = {
                x: xValues,
                y: yValues,
                mode: 'lines+markers',
                type: 'scatter',
                line: { shape: 'linear' }
            };

            const layout = {
                title: chartData.title || "Line Chart",
                xaxis: { title: chartData.x_col || "X Values", automargin: true },
                yaxis: { title: chartData.y_col || "Y Values", automargin: true },
                margin: {
                    t: 40,
                    b: 60,
                    l: 60,
                    r: 40
                }
            };

            Plotly.react(containerId, [trace], layout);
        }

        // Function to render simple scatter charts from the new renderer
        function _renderSimpleScatterChart(chartData, containerId) {
            if (!chartData.data || chartData.data.length === 0) {
                renderEmptyChart(containerId, "No data available");
                return;
            }

            // Filter valid data points with x and y values
            const validData = chartData.data.filter(d =>
                d.x !== undefined && d.y !== undefined &&
                d.x !== null && d.y !== null &&
                !isNaN(parseFloat(d.x)) && !isNaN(parseFloat(d.y))
            );

            if (validData.length === 0) {
                renderEmptyChart(containerId, "No valid data points");
                return;
            }

            // Prepare the x and y values
            const xValues = validData.map(d => parseFloat(d.x));
            const yValues = validData.map(d => parseFloat(d.y));

            const trace = {
                x: xValues,
                y: yValues,
                mode: 'markers',
                type: 'scatter',
                marker: {
                    size: 8,
                    opacity: 0.6,
                    color: 'rgba(55, 128, 191, 0.6)'
                }
            };

            const layout = {
                title: chartData.title || "Scatter Chart",
                xaxis: {
                    title: chartData.x_col || "X Values",
                    automargin: true,
                    showgrid: true,
                    gridcolor: 'lightgray'
                },
                yaxis: {
                    title: chartData.y_col || "Y Values",
                    automargin: true,
                    showgrid: true,
                    gridcolor: 'lightgray'
                },
                margin: {
                    t: 40,
                    b: 60,
                    l: 60,
                    r: 40
                }
            };

            Plotly.react(containerId, [trace], layout);
        }

        // Function to render simple histogram charts from the new renderer
        function _renderSimpleHistogramChart(chartData, containerId) {
            if (!chartData.data || chartData.data.length === 0) {
                renderEmptyChart(containerId, "No data available");
                return;
            }

            // Filter valid data points with x and y values
            const validData = chartData.data.filter(d =>
                d.x !== undefined && d.y !== undefined &&
                d.x !== null && d.y !== null
            );

            if (validData.length === 0) {
                renderEmptyChart(containerId, "No valid data points");
                return;
            }

            // Prepare the x and y values
            const xValues = validData.map(d => String(d.x));
            const yValues = validData.map(d => {
                const val = parseFloat(d.y);
                return isNaN(val) ? 0 : val;
            });

            const trace = {
                x: xValues,
                y: yValues,
                type: 'bar'
            };

            const layout = {
                title: chartData.title || "Histogram",
                xaxis: {
                    title: chartData.x_col || "Bins",
                    automargin: true,
                    tickangle: -45
                },
                yaxis: { title: "Frequency", automargin: true },
                margin: {
                    t: 40,
                    b: 80,
                    l: 60,
                    r: 40
                }
            };

            Plotly.react(containerId, [trace], layout);
        }

        // Function to render simple pie charts from the new renderer
        function _renderSimplePieChart(chartData, containerId) {
            if (!chartData.data || chartData.data.length === 0) {
                renderEmptyChart(containerId, "No data available");
                return;
            }

            // Filter valid data points with label and value
            const validData = chartData.data.filter(d =>
                d.label !== undefined && d.value !== undefined &&
                d.label !== null && d.value !== null
            );

            if (validData.length === 0) {
                renderEmptyChart(containerId, "No valid data points");
                return;
            }

            // Prepare the labels and values
            const labels = validData.map(d => String(d.label));
            const values = validData.map(d => {
                const val = parseFloat(d.value);
                return isNaN(val) ? 0 : val;
            });

            const trace = {
                labels: labels,
                values: values,
                type: 'pie',
                textinfo: 'label+percent',
                textposition: 'inside'
            };

            const layout = {
                title: chartData.title || "Pie Chart",
                margin: {
                    t: 40,
                    b: 20,
                    l: 20,
                    r: 20
                }
            };

            Plotly.react(containerId, [trace], layout);
        }

        // Function to render scatter plots
        function _renderScatterPlot(chartData, containerId) {
            const xValues = chartData.data.map(d => d.x);
            const yValues = chartData.data.map(d => d.y);

            const trace = {
                x: xValues,
                y: yValues,
                mode: 'markers',
                type: 'scatter',
                marker: {
                    size: 8,
                    opacity: 0.6,
                    color: 'rgba(55, 128, 191, 0.6)'
                }
            };

            const layout = {
                title: chartData.title || "Scatter Plot",
                xaxis: {
                    title: chartData.x_column || "X Values",
                    automargin: true,
                    showgrid: true,
                    gridcolor: 'lightgray'
                },
                yaxis: {
                    title: chartData.y_column || "Y Values",
                    automargin: true,
                    showgrid: true,
                    gridcolor: 'lightgray'
                },
                margin: {
                    t: 60,
                    b: 80,
                    l: 60,
                    r: 40
                }
            };

            Plotly.react(containerId, [trace], layout);
        }

        // Function to render time series charts
        function _renderTimeSeriesChart(chartData, containerId) {
            const dates = chartData.data.map(row => row.date);
            const values = chartData.data.map(row => row.value);

            const trace = {
                type: "scatter",
                mode: "lines+markers",
                x: dates,
                y: values,
                line: { shape: "linear" }
            };

            const layout = {
                title: chartData.title || "Time Series Chart",
                xaxis: {
                    title: chartData.x_column || "Date",
                    type: "date",
                    tickformat: "%Y-%m-%d",
                    autorange: true,
                    automargin: true
                },
                yaxis: {
                    title: chartData.y_column || "Value",
                    automargin: true
                },
                margin: {
                    t: 40,
                    b: 60,
                    l: 60,
                    r: 40
                }
            };

            Plotly.react(containerId, [trace], layout);
        }

        // Function to render pie charts
        function _renderPieChart(chartData, containerId) {
            const labels = chartData.data.map(row => row.category);
            const values = chartData.data.map(row => row.value);

            const trace = {
                type: 'pie',
                labels: labels,
                values: values,
                textinfo: 'label+percent',
                textposition: 'inside',
                automargin: true
            };

            const layout = {
                title: chartData.title || "Pie Chart",
                margin: {
                    t: 40,
                    b: 20,
                    l: 20,
                    r: 20
                }
            };

            Plotly.react(containerId, [trace], layout);
        }

        // Function to render bar charts (default)
        function _renderBarChart(chartData, containerId) {
            // Handle both formats: with "category" and "count" or "category" and "value"
            let categories;
            let values;

            if (chartData.data[0].hasOwnProperty('count')) {
                categories = chartData.data.map(row => row.category);
                values = chartData.data.map(row => row.count);
            } else if (chartData.data[0].hasOwnProperty('value')) {
                categories = chartData.data.map(row => row.category);
                values = chartData.data.map(row => row.value);
            } else {
                // Unknown data format, skip
                return;
            }

            const maxVal = Math.max(...values, 0);
            const avgLabelLen = categories.reduce((sum, c) => sum + c.length, 0) / categories.length;
            const manyCategories = categories.length > 6;
            const longLabels = avgLabelLen > 12;
            const useHorizontal = manyCategories || longLabels;

            let trace, layout;

            if (useHorizontal) {
                trace = {
                    type: "bar",
                    x: values,
                    y: categories,
                    orientation: "h",
                    text: values,
                    textposition: "outside"
                };

                layout = {
                    xaxis: {
                        title: "Count",
                        tickformat: ",d",
                        exponentformat: "none",
                        rangemode: "tozero",
                        range: [0, maxVal * 1.15 || 1],
                        automargin: true
                    },
                    yaxis: {
                        title: chartData.x_column || "Category",
                        categoryorder: "array",
                        categoryarray: categories,
                        automargin: true
                    },
                    margin: {
                        t: 20,  // Very small top margin since we're removing title
                        b: 60,  // Keep bottom margin for labels
                        l: 80,  // Reduced left margin
                        r: 20   // Reduced right margin
                    },
                    height: 320  // Explicit height to match container
                };
            } else {
                trace = {
                    type: "bar",
                    x: categories,
                    y: values,
                    text: values,
                    textposition: "outside"
                };

                layout = {
                    xaxis: {
                        title: chartData.x_column || "Category",
                        categoryorder: "array",
                        categoryarray: categories,
                        automargin: true
                    },
                    yaxis: {
                        title: "Count",
                        tickformat: ",d",
                        exponentformat: "none",
                        rangemode: "tozero",
                        range: [0, maxVal * 1.15 || 1],
                        automargin: true
                    },
                    margin: {
                        t: 20,  // Very small top margin since we're removing title
                        b: 60,  // Slightly more bottom for labels
                        l: 60,
                        r: 20
                    },
                    height: 320  // Explicit height to match container
                };
            }

            Plotly.react(containerId, [trace], layout);
        }

        document.addEventListener("DOMContentLoaded", function () {
            try {
                // Render all category charts
                if (typeof CATEGORY_CHARTS === 'object' && CATEGORY_CHARTS !== null) {
                    Object.keys(CATEGORY_CHARTS).forEach(function(columnName) {
                        try {
                            const chartData = CATEGORY_CHARTS[columnName];
                            const containerId = "chart-" + columnName.replace(/[\s.]/g, '_');
                            const containerElement = document.getElementById(containerId);

                            if (!containerElement) {
                                console.warn(`Chart container element with ID '${containerId}' not found for category chart.`);
                                return;
                            }

                            renderCategoryCountChart(chartData, containerId);
                        } catch (error) {
                            console.error(`Error rendering category chart for column '${columnName}':`, error);
                        }
                    });
                } else {
                    console.warn("CATEGORY_CHARTS is not available or properly initialized.");
                }

                // Render all other chart types from ALL_CHARTS
                try {
                    if (Array.isArray(ALL_CHARTS)) {
                        ALL_CHARTS.forEach(function(chartData, index) {
                            try {
                                const containerId = "chart_" + (index + 1);
                                const containerElement = document.getElementById(containerId);

                                if (!containerElement) {
                                    console.warn(`Chart container element with ID '${containerId}' not found for chart ${index}.`);
                                    return;
                                }

                                renderChartByType(chartData, containerId);
                            } catch (error) {
                                console.error(`Error rendering chart ${index}:`, error);
                            }
                        });
                    } else {
                        console.warn("ALL_CHARTS is not available or not an array.");
                    }
                } catch (error) {
                    console.error("Error processing ALL_CHARTS:", error);
                }
            } catch (error) {
                console.error("Error in DOMContentLoaded event handler:", error);
            }

            // KPI click wiring
            try {
                const kpiItems = document.querySelectorAll("[data-kpi-column]");
                kpiItems.forEach(item => {
                    item.addEventListener("click", function () {
                        try {
                            const hasChart = this.getAttribute("data-has-chart") === "1";
                            if (!hasChart) {
                                // No chart wired to this KPI; do nothing visibly.
                                // Optional: console.log for debugging
                                // console.log("No chart for KPI:", this.getAttribute("data-kpi-column"));
                                return;
                            }

                            const col = this.getAttribute("data-kpi-column");
                            if (col) {
                                loadChartForColumn(col);
                            }
                        } catch (error) {
                            console.error("Error in KPI click handler:", error);
                        }
                    });
                });
            } catch (error) {
                console.error("Error setting up KPI click handlers:", error);
            }
        });

    </script>
</body>
</html>
```

## src/data/parser.py

```
import os
import pandas as pd
import logging
from typing import Optional, Union, NamedTuple
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Define maximum file size (e.g., 100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024

class LoadResult(NamedTuple):
    """Structured result for data loading operations"""
    success: bool
    df: Optional[pd.DataFrame] = None
    error_code: Optional[str] = None
    detail: Optional[str] = None

def load_csv(file_storage, max_file_size: int = MAX_FILE_SIZE) -> Optional[pd.DataFrame]:
    """
    Load a CSV file from Flask file storage with validation and error handling.

    Args:
        file_storage: the uploaded file object from Flask (request.files["dataset"])
        max_file_size: maximum allowed file size in bytes (not enforced to avoid upload issues)

    Returns: pandas DataFrame or None if something goes wrong
    """
    try:
        # Make sure we're at the start of the file
        if hasattr(file_storage, "seek"):
            file_storage.seek(0)
        elif hasattr(file_storage, "stream") and hasattr(file_storage.stream, "seek"):
            file_storage.stream.seek(0)

        # Try standard UTF-8 read first with additional parameters for robustness
        try:
            # Try with automatic delimiter detection
            df = pd.read_csv(file_storage,
                           encoding='utf-8',
                           engine='python',
                           on_bad_lines='skip')  # Skip problematic lines instead of failing
            logger.info(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            return df
        except UnicodeDecodeError as e:
            logger.warning(f"Unicode error with UTF-8, retrying with latin1: {e}")
            # Retry with a more forgiving encoding
            if hasattr(file_storage, "seek"):
                file_storage.seek(0)
            elif hasattr(file_storage, "stream") and hasattr(file_storage.stream, "seek"):
                file_storage.stream.seek(0)
            df = pd.read_csv(file_storage,
                           encoding="latin1",
                           engine='python',
                           on_bad_lines='skip')
            logger.info(f"Successfully loaded CSV with latin1 encoding")
            return df
        except pd.errors.EmptyDataError:
            logger.error("CSV file is empty")
            return None
        except pd.errors.ParserError as e:
            logger.error(f"Parser error while reading CSV: {e}")
            return None

    except Exception as e:
        logger.exception(f"Unexpected error reading CSV: {e}")
        return None


def load_csv_from_url(url: str, timeout: int = 30) -> Optional[pd.DataFrame]:
    """
    Load a CSV directly from a URL (e.g. GitHub raw link).
    Includes validation, timeout, and error handling.

    Args:
        url: URL to the CSV file
        timeout: Request timeout in seconds

    Returns: pandas DataFrame or None on failure.
    """
    # Validate URL format
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            logger.error(f"Invalid URL format: {url}")
            return None
        if parsed.scheme not in ['http', 'https']:
            logger.error(f"Invalid URL scheme: {parsed.scheme}")
            return None
    except Exception as e:
        logger.error(f"Error parsing URL: {e}")
        return None

    try:
        logger.info(f"Loading CSV from URL: {url}")

        # Configure session with retry strategy
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        response = session.get(url, timeout=timeout)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Read CSV from response content
        import io
        df = pd.read_csv(io.StringIO(response.text))
        logger.info(f"Successfully loaded CSV from URL with {len(df)} rows and {len(df.columns)} columns")
        return df
    except pd.errors.EmptyDataError:
        logger.error(f"CSV from URL is empty: {url}")
        return None
    except pd.errors.ParserError as e:
        logger.error(f"Parser error loading CSV from URL {url}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error loading CSV from URL {url}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Error loading CSV from URL {url}: {e}")
        return None


def load_csv_from_kaggle(slug: str, csv_name: Optional[str] = None, timeout: int = 60) -> Optional[pd.DataFrame]:
    """
    Load a CSV from a Kaggle dataset using kagglehub with validation and error handling.

    Args:
        slug: e.g. "umitka/global-youth-unemployment-dataset"
        csv_name: optional specific CSV filename inside the dataset.
                  If not provided, the first .csv file found will be used.
        timeout: timeout for download operation in seconds

    Requires `pip install kagglehub` and Kaggle credentials configured
    in the environment.
    """
    # Validate slug format (basic validation)
    if not slug or '/' not in slug:
        logger.error(f"Invalid Kaggle dataset slug format: {slug}")
        return None

    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub is not installed. Please 'pip install kagglehub' to use Kaggle sources.")
        return None

    try:
        logger.info(f"Downloading Kaggle dataset: {slug}")
        path = kagglehub.dataset_download(slug, verbose=False)
        logger.info(f"Downloaded Kaggle dataset to: {path}")

        if csv_name:
            target = os.path.join(path, csv_name)
            if not os.path.isfile(target):
                logger.error(f"CSV file '{csv_name}' not found in Kaggle dataset folder: {path}")
                return None
            df = pd.read_csv(target)
            logger.info(f"Successfully loaded specific CSV file from Kaggle dataset: {csv_name}")
            return df

        # Otherwise, pick the first .csv file in the folder
        files = [f for f in os.listdir(path) if f.lower().endswith(".csv")]
        if not files:
            logger.error("No CSV files found in Kaggle dataset folder.")
            return None

        first_csv = os.path.join(path, files[0])
        logger.info(f"Loading first CSV file from Kaggle dataset: {files[0]}")
        df = pd.read_csv(first_csv)
        logger.info(f"Successfully loaded CSV from Kaggle dataset with {len(df)} rows and {len(df.columns)} columns")
        return df

    except Exception as e:
        logger.exception(f"Error loading Kaggle dataset: {e}")
        return None

```

## src/data/analyser.py

```
"""
Advanced data analyzer with robust column classification and semantic inference.
Focuses on identifying identifiers, mixed-unit fields, multi-value text fields,
and improving semantic tagging with confidence scores.
"""

import logging
import pandas as pd
import numpy as np
import re
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_bool_dtype
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import math

logger = logging.getLogger(__name__)

# Configuration parameters with defaults
UNIQUENESS_CUTOFF = 0.5  # Ratio where text vs categorical is determined
AVG_LENGTH_CUTOFF = 30   # Character length where text vs categorical is determined
MIN_DATE = 1900
MAX_DATE = 2100

# Regex patterns for identifying different field types
ID_PATTERNS = [
    r'^[0-9]+$',  # Pure numeric sequences
    r'^[a-zA-Z0-9]{8,}$',  # Long alphanumeric strings (possible UUIDs)
    r'^[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}$',  # Standard UUID format
    r'^[a-zA-Z0-9]{20,}$',  # Very long alphanumeric (possible hash)
]

# Patterns for mixed units (duration, currency, etc.)
MIXED_UNIT_PATTERNS = {
    'duration': [
        r'\d+\s*(min|mins|minute|minutes|h|hour|hours|s|sec|second|seconds|d|day|days)',  # Duration indicators
        r'[0-9:]{4,}'  # HH:MM format
    ],
    'currency': [
        r'\$\s*\d+\.?\d*',  # Dollar amounts
        r'€\s*\d+\.?\d*',  # Euro amounts
        r'£\s*\d+\.?\d*',  # Pound amounts
        r'\d+\.?\d*\s*(USD|EUR|GBP|JPY|CAD|AUD|CHF|CNY|INR)' # Currency codes
    ],
    'percentage': [
        r'\d+\.?\d*\s*%',  # Percentages
        r'\d+\.?\d*/\d+\.?\d*'  # Fractions (e.g., 3/4)
    ]
}

def _calculate_confidence(confidence_factors: Dict[str, float]) -> float:
    """
    Combine different confidence factors into a single score.
    """
    if not confidence_factors:
        return 0.5  # Default medium confidence

    weights = {
        'pattern_match': 0.4,
        'data_consistency': 0.3,
        'cardinality': 0.2,
        'semantic_context': 0.1,
    }

    total_confidence = 0.0
    total_weight = 0.0

    for factor, value in confidence_factors.items():
        weight = weights.get(factor, 0.1)
        total_confidence += value * weight
        total_weight += weight

    return total_confidence / total_weight if total_weight > 0 else 0.5


def _extract_numeric_from_mixed(text_values: pd.Series) -> Tuple[Optional[pd.Series], float]:
    """
    Extract numeric values from mixed-unit text fields (e.g. '90 min', '$123', '45%').

    Returns:
        parsed_series: Series aligned with the sample index, or None if parsing is not meaningful
        confidence: ratio of successfully parsed non-NaN values in the sample
    """
    if text_values.empty:
        return None, 0.0

    # Sample for detection (not full column conversion, just enough to decide)
    sample_size = min(100, len(text_values))
    sample_values = text_values.dropna().head(sample_size)

    if sample_values.empty:
        return None, 0.0

    parsed_values = []
    success_flags = []

    def _extract_first_number(s: str) -> Optional[float]:
        nums = re.findall(r'\d+\.?\d*', s)
        if not nums:
            return None
        try:
            return float(nums[0])
        except Exception:
            return None

    for val in sample_values:
        if pd.isna(val):
            parsed_values.append(np.nan)
            success_flags.append(False)
            continue

        str_val = str(val).strip().lower()
        parsed_num = None

        # Define helper function inside
        def _extract_first_number(s: str) -> Optional[float]:
            nums = re.findall(r'\d+\.?\d*', s)
            if not nums:
                return None
            try:
                return float(nums[0])
            except Exception:
                return None

        # Try duration, currency, percentage patterns in that order
        for unit_type in ["duration", "currency", "percentage"]:
            matched = False
            for pattern in MIXED_UNIT_PATTERNS[unit_type]:
                if re.search(pattern, str_val):
                    num = _extract_first_number(str_val)
                    if num is not None:
                        parsed_num = num
                        matched = True
                        break
            if matched:
                break

        if parsed_num is not None:
            parsed_values.append(parsed_num)
            success_flags.append(True)
        else:
            parsed_values.append(np.nan)
            success_flags.append(False)

    # Compute confidence based on non-NaN parses
    successful_parses = sum(success_flags)
    confidence = successful_parses / len(success_flags) if success_flags else 0.0

    # If too little was successfully parsed, this is not a reliable mixed-numeric field
    if successful_parses == 0 or confidence < 0.3:
        return None, 0.0

    parsed_series = pd.Series(parsed_values, index=sample_values.index)
    return parsed_series, confidence


def _is_multi_value_field(series: pd.Series, delimiter_chars: List[str] = [',', ';', '|', '/']) -> Tuple[bool, str, float]:
    """
    Detect if a field contains multiple values separated by delimiters.

    Args:
        series: The pandas Series to analyze
        delimiter_chars: List of potential delimiter characters

    Returns:
        Tuple of (is_multi_value, primary_delimiter, confidence_score)
    """
    if series.dtype != 'object' and not str(series.dtype).startswith('string'):
        return False, "", 0.0

    sample_size = min(100, len(series))
    sample_values = series.dropna().head(sample_size)

    if len(sample_values) == 0:
        return False, "", 0.0

    delimiter_scores = {}

    for delim in delimiter_chars:
        splits = sample_values.astype(str).str.contains(delim, na=False)
        if splits.any():
            split_ratio = splits.sum() / len(sample_values)
            # Consider it multi-value if at least 20% of values contain the delimiter
            if split_ratio >= 0.2:
                delimiter_scores[delim] = split_ratio

    if delimiter_scores:
        best_delimiter = max(delimiter_scores, key=delimiter_scores.get)
        confidence = delimiter_scores[best_delimiter]

        # Additional validation: check if the splits seem meaningful
        sample_with_delim = sample_values[sample_values.astype(str).str.contains(best_delimiter, na=False)]
        if len(sample_with_delim) > 0:
            avg_splits = sample_with_delim.astype(str).str.count(best_delimiter).mean() + 1
            # If on average there are more than 2 values per field, it's likely multi-value
            if avg_splits >= 2:
                return True, best_delimiter, confidence

    return False, "", 0.0


def _is_likely_identifier(
    series: pd.Series,
    uniqueness_threshold: float = 0.95
) -> Tuple[bool, float]:
    """
    Detect if a series is likely an identifier based on various heuristics.

    We now combine:
      - high uniqueness
      - ID-like patterns
      - column name hints

    This is to avoid misclassifying genuine numeric metrics as identifiers.
    """
    n_total = len(series)
    if n_total == 0:
        return False, 0.0

    n_unique = series.nunique(dropna=True)
    unique_ratio = n_unique / n_total if n_total > 0 else 0.0

    # Check if column name suggests it's an ID
    name_lower = (series.name or "").lower()
    id_name_tokens = [
        "id", "identifier", "uuid", "guid", "key", "account", "user", "customer",
        "client", "booking", "transaction", "order", "invoice", "code", "number"
    ]
    looks_like_id_name = any(token in name_lower for token in id_name_tokens)

    # Check for ID-like patterns in the values
    sample_values = series.dropna().head(min(100, len(series))).astype(str)
    id_pattern_matches = 0

    for val in sample_values:
        for pattern in ID_PATTERNS:
            if re.fullmatch(pattern, val.strip(), re.IGNORECASE):
                id_pattern_matches += 1
                break

    pattern_confidence = id_pattern_matches / len(sample_values) if len(sample_values) > 0 else 0.0

    # High uniqueness is necessary for ID classification, but not sufficient on its own
    # Only classify as ID if either name looks like ID or strong pattern matching occurs
    if unique_ratio > uniqueness_threshold:
        if looks_like_id_name:
            # Strong confidence if both high uniqueness and name suggests ID
            confidence = min(1.0, unique_ratio * 1.2)  # Boost for name match
            return True, confidence
        elif pattern_confidence > 0.5:
            # Moderate confidence if high uniqueness and strong pattern match
            confidence = min(0.9, unique_ratio * pattern_confidence * 1.5)
            return True, confidence
        elif unique_ratio > 0.99:  # Very high uniqueness might indicate ID anyway
            # Lower confidence if just high uniqueness but no other indicators
            confidence = min(0.7, unique_ratio)
            return True, confidence

    return False, 0.0


def _infer_role_advanced(
    series: pd.Series,
    uniqueness_cutoff: float = 0.5,
    avg_length_cutoff: int = 30
) -> Tuple[str, float, str, Dict[str, float], List[str]]:
    """
    Advanced role inference with confidence scoring and semantic tags.

    Returns:
        role: core role ("numeric", "categorical", "datetime", "text", "boolean", "identifier", "ordinal")
        confidence: 0–1
        provenance: short string indicating the main decision path
        confidence_factors: breakdown of the confidence calculation
        semantic_tags: list of semantic hints like ["geographic"], ["duration"], ["monetary"], ["percentage"], ["multi_value"]
    """
    semantic_tags = []

    if series.empty:
        return "text", 0.5, "empty_series", {"data_consistency": 0.0}, semantic_tags

    # 0) Identifier detection (uses name + uniqueness + patterns)
    is_id, id_confidence = _is_likely_identifier(series)
    if is_id and id_confidence > 0.7:
        return "identifier", id_confidence, f"identifier_detection_{id_confidence:.2f}", {
            "pattern_match": id_confidence,
            "data_consistency": 1.0,
            "cardinality": 1.0
        }, semantic_tags

    # 1) Multi-value text field detection
    is_multi, delimiter, multi_confidence = _is_multi_value_field(series)
    if is_multi and multi_confidence > 0.5:
        semantic_tags.append("multi_value")
        return "text", multi_confidence, f"multivalue_delim_{delimiter}_conf_{multi_confidence:.2f}", {
            "pattern_match": multi_confidence,
            "data_consistency": 0.8,
            "cardinality": 0.3
        }, semantic_tags

    # 2) Boolean detection
    unique_vals = series.dropna().unique()
    if len(unique_vals) <= 2:
        unique_str = [str(val).lower() for val in unique_vals if pd.notna(val)]
        boolean_indicators = {'true', 'false', 'yes', 'no', '1', '0', 't', 'f', 'y', 'n', '1.0', '0.0', 'on', 'off', 'true.', 'false.'}
        if set(unique_str).issubset(boolean_indicators) or pd.api.types.is_bool_dtype(series):
            return "boolean", 0.9, f"boolean_values_{len(unique_vals)}", {
                "pattern_match": 0.9,
                "data_consistency": 1.0,
                "cardinality": 0.1
            }, semantic_tags

    name_lower = (series.name or "").lower()

    # 3) Native numeric dtype (includes potential geo/monetary/unit values which we tag separately)
    if pd.api.types.is_numeric_dtype(series):
        # Check for potential datetime in numeric format (e.g. years)
        s_nonnull = series.dropna()
        if not s_nonnull.empty:
            col_min = float(s_nonnull.min())
            col_max = float(s_nonnull.max())

            # Year-like numeric datetime
            if (
                MIN_DATE <= col_min <= MAX_DATE and
                MIN_DATE <= col_max <= MAX_DATE and
                any(keyword in name_lower for keyword in ["year", "yr"])
            ):
                return "datetime", 0.8, "numeric_year_keyword", {
                    "pattern_match": 0.7,
                    "data_consistency": 0.9,
                    "semantic_context": 0.8
                }, semantic_tags

        # For numeric data, check for semantic tags based on name
        # Geographic indicators
        geo_indicators = ["lat", "lon", "long", "latitude", "longitude", "coord", "x_", "y_", "x", "y"]
        if any(indicator in name_lower for indicator in geo_indicators):
            semantic_tags.append("geographic")

        # Currency indicators
        money_tokens = ["price", "cost", "revenue", "sales", "amount", "income", "expense", "salary", "wage", "fee",
                        "charge", "payment", "profit", "budget", "fund", "investment", "capital", "value", "total"]
        if any(token in name_lower for token in money_tokens):
            semantic_tags.append("monetary")

        # Percentage indicators
        perc_tokens = ["percent", "percentage", "pct", "ratio", "rate", "_pct", "prop", "proportion"]
        if any(token in name_lower for token in perc_tokens):
            semantic_tags.append("percentage")

        # Duration/timing indicators
        dur_tokens = ["duration", "length", "time", "period", "span", "interval", "delay", "lag", "gap"]
        if any(token in name_lower for token in dur_tokens):
            semantic_tags.append("duration")

        return "numeric", 0.8, "numeric_dtype", {
            "pattern_match": 0.8,
            "data_consistency": 0.9,
            "cardinality": 0.5
        }, semantic_tags

    # 4) Native datetime dtype
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime", 0.9, "datetime_dtype", {
            "pattern_match": 0.9,
            "data_consistency": 1.0,
            "cardinality": 0.5
        }, semantic_tags

    # 5) Attempt to parse datetime from object/string
    if series.dtype == "object":
        sample = series.dropna().astype(str).head(50)  # Sample for performance
        if not sample.empty:
            try:
                parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
                valid_dates = parsed.notna().mean()
                if valid_dates > 0.7:  # Majority are parseable as dates
                    return "datetime", valid_dates, f"datetime_parsed_{valid_dates:.2f}", {
                        "pattern_match": valid_dates,
                        "data_consistency": 0.8,
                        "cardinality": 0.5
                    }, semantic_tags
            except Exception as e:
                logger.debug(f"Date parsing failed for column {series.name}: {e}")

    # 6) Mixed-unit numeric embedded in text (duration, currency, percentage)
    if series.dtype == "object":
        parsed_nums, mixed_confidence = _extract_numeric_from_mixed(series)
        if parsed_nums is not None and mixed_confidence > 0.6:
            # Determine the semantic tag based on patterns in the original text
            sample_values = series.dropna().head(min(50, len(series))).astype(str)
            unit_votes = Counter()

            for val in sample_values:
                val_lower = val.lower()
                # Check for duration patterns
                for pattern in MIXED_UNIT_PATTERNS['duration']:
                    if re.search(pattern, val_lower):
                        unit_votes['duration'] += 1
                        break
                # Check for currency patterns
                for pattern in MIXED_UNIT_PATTERNS['currency']:
                    if re.search(pattern, val_lower):
                        unit_votes['currency'] += 1
                        break
                # Check for percentage patterns
                for pattern in MIXED_UNIT_PATTERNS['percentage']:
                    if re.search(pattern, val_lower):
                        unit_votes['percentage'] += 1
                        break

            # Assign semantic tag based on most common pattern
            if unit_votes:
                dominant_unit, _ = unit_votes.most_common(1)[0]
                if dominant_unit == 'duration':
                    semantic_tags.append("duration")
                elif dominant_unit == 'currency':
                    semantic_tags.append("monetary")
                elif dominant_unit == 'percentage':
                    semantic_tags.append("percentage")

            # Still return numeric role but with semantic tags
            return "numeric", mixed_confidence, "mixed_unit_numeric", {
                "pattern_match": mixed_confidence,
                "data_consistency": 0.7,
                "cardinality": 0.4
            }, semantic_tags

    # 7) Check for ordinal categorical patterns (e.g. small sets of ordered categories)
    if series.dtype == "object":
        str_values = series.dropna().astype(str).str.lower().unique()
        ordinal_patterns = [
            {'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth'},
            {'low', 'medium', 'high'},
            {'small', 'medium', 'large'},
            {'beginner', 'intermediate', 'advanced'},
            {'junior', 'senior', 'expert'},
            {'one', 'two', 'three', 'four', 'five'},
            {'none', 'low', 'medium', 'high', 'maximum'},
            {'min', 'mid', 'max'},
            {'start', 'middle', 'end'},
            {'level 1', 'level 2', 'level 3', 'level 4', 'level 5'},
            {'grade 1', 'grade 2', 'grade 3', 'grade 4', 'grade 5'},
            {'tier 1', 'tier 2', 'tier 3'},
            {'class 1', 'class 2', 'class 3'}
        ]

        for pattern_set in ordinal_patterns:
            if set(str_values).issubset(pattern_set):
                return "ordinal", 0.85, f"ordinal_pattern_{len(set(str_values))}", {
                    "pattern_match": 0.85,
                    "data_consistency": 0.8,
                    "semantic_context": 0.6
                }, semantic_tags

    # 8) For text/object types, check semantic hints in name and decide role based on characteristics
    # Geographic semantic tag based on column name (but keep text role if not obviously categorical)
    geo_name_tokens = ["country", "city", "state", "province", "address", "location", "region", "zone", "lat", "lon", "long", "coord"]
    if any(indicator in name_lower for indicator in geo_name_tokens):
        semantic_tags.append("geographic")

    text_indicators = ["desc", "comment", "note", "text", "message", "review", "comments", "notes", "info", "details"]
    if any(indicator in name_lower for indicator in text_indicators):
        semantic_tags.append("textual")

    # 9) Use cardinality and length heuristics to decide between text and categorical
    n_rows = len(series)
    n_unique = series.nunique(dropna=True)
    unique_ratio = n_unique / n_rows if n_rows > 0 else 0.0

    if series.notna().any():
        try:
            avg_len = series.dropna().astype(str).str.len().mean()
        except:
            logger.warning(f"Could not compute average length for series {series.name}")
            avg_len = 0
    else:
        avg_len = 0

    # Very high cardinality with long values = text
    if unique_ratio > 0.9 and n_unique > 100 and avg_len > 20:
        return "text", 0.9, f"high_cardinality_long_text_unique_ratio_{unique_ratio:.2f}_avg_len_{avg_len:.1f}", {
            "pattern_match": 0.9,
            "data_consistency": 0.9,
            "cardinality": 1.0
        }, semantic_tags
    # High cardinality or long values = text
    elif unique_ratio > uniqueness_cutoff or avg_len > avg_length_cutoff:
        return "text", max(0.6, min(0.9, unique_ratio)), f"high_cardinality_or_long_values_unique_ratio_{unique_ratio:.2f}_avg_len_{avg_len:.1f}", {
            "pattern_match": min(0.9, unique_ratio),
            "data_consistency": 0.7,
            "cardinality": unique_ratio
        }, semantic_tags
    # Low cardinality = categorical
    elif unique_ratio < 0.05 and n_unique <= 10:
        return "categorical", max(0.7, 1.0 - unique_ratio), f"low_cardinality_n_{n_unique}_unique_ratio_{unique_ratio:.2f}", {
            "pattern_match": 0.8,
            "data_consistency": 0.8,
            "cardinality": 1.0 - unique_ratio
        }, semantic_tags
    else:
        # Medium cardinality - categorical with lower confidence
        return "categorical", 0.6, f"medium_cardinality_n_{n_unique}_unique_ratio_{unique_ratio:.2f}", {
            "pattern_match": 0.6,
            "data_consistency": 0.7,
            "cardinality": 0.6
        }, semantic_tags


def build_dataset_profile(df: pd.DataFrame, max_cols: int = 50) -> Dict[str, Any]:
    """
    Build a rich dataset profile with confidence scores and semantic tags.
    Returns a dict with structured profile information.
    """
    if df.empty:
        logger.warning("DataFrame is empty, returning empty profile")
        return {
            "n_rows": 0,
            "n_cols": 0,
            "role_counts": {"numeric": 0, "datetime": 0, "categorical": 0, "text": 0, "identifier": 0, "boolean": 0, "ordinal": 0, "other": 0},
            "columns": []
        }

    n_rows = int(len(df))
    n_cols = int(df.shape[1])

    # Set limits
    max_cols = min(max_cols, n_cols)

    columns = []
    role_counts = {
        "numeric": 0,
        "datetime": 0,
        "categorical": 0,
        "text": 0,
        "identifier": 0,
        "boolean": 0,
        "ordinal": 0,
        "other": 0,
    }

    for i, col in enumerate(df.columns):
        if i >= max_cols:
            break

        s = df[col]

        # Handle edge case where column has all NaN values
        if s.isna().all():
            role = "text"
            confidence = 0.3
            provenance = "all_nan"
            confidence_factors = {"data_consistency": 0.0}
            semantic_tags = []
        else:
            role, confidence, provenance, confidence_factors, semantic_tags = _infer_role_advanced(
                s, uniqueness_cutoff=UNIQUENESS_CUTOFF, avg_length_cutoff=AVG_LENGTH_CUTOFF
            )

        # Track role counts
        if role in role_counts:
            role_counts[role] += 1
        else:
            role_counts["other"] += 1

        # Default: no stats and no top categories
        stats = None
        top_categories = []

        # Compute column statistics based on role
        if role in ("numeric", "numeric_duration", "numeric_currency", "numeric_percentage", "numeric_mixed"):
            # Process as numeric
            s_clean = pd.to_numeric(s, errors='coerce')
            s_clean = s_clean.dropna()

            if len(s_clean) > 0:
                # Compute all statistics in one pass for efficiency
                stats = {
                    "min": float(s_clean.min()) if len(s_clean) > 0 else None,
                    "max": float(s_clean.max()) if len(s_clean) > 0 else None,
                    "mean": float(s_clean.mean()) if len(s_clean) > 0 else None,
                    "std": float(s_clean.std()) if len(s_clean) > 1 else 0.0,
                    "median": float(s_clean.median()) if len(s_clean) > 0 else None,
                    "q25": float(s_clean.quantile(0.25)) if len(s_clean) > 0 else None,
                    "q75": float(s_clean.quantile(0.75)) if len(s_clean) > 0 else None,
                    "sum": float(s_clean.sum()) if len(s_clean) > 0 else None,
                    "variance": float(s_clean.var()) if len(s_clean) > 1 else 0.0,
                    "skewness": float(s_clean.skew()) if len(s_clean) > 2 else 0.0,
                    "kurtosis": float(s_clean.kurtosis()) if len(s_clean) > 3 else 0.0,
                    "count": int(len(s_clean))
                }

                # Add specific semantic-based stats
                if "monetary" in semantic_tags:
                    stats["currency_units"] = float(s_clean.abs().sum())  # Total monetary value
                elif "percentage" in semantic_tags:
                    stats["average_percentage"] = float(s_clean.mean())
                elif "duration" in semantic_tags:
                    stats["total_duration"] = float(s_clean.sum())

        elif role == "datetime":
            try:
                s_dt = pd.to_datetime(s, errors="coerce")
                s_dt_clean = s_dt.dropna()
                if len(s_dt_clean) > 0:
                    stats = {
                        "min": s_dt_clean.min().isoformat() if not s_dt_clean.empty else None,
                        "max": s_dt_clean.max().isoformat() if not s_dt_clean.empty else None,
                        "range_days": (s_dt_clean.max() - s_dt_clean.min()).days if not s_dt_clean.empty else 0,
                        "count": int(len(s_dt_clean))
                    }
            except Exception as e:
                logger.warning(f"Error computing datetime stats for {col}: {e}")
                stats = None

        elif role in ("categorical", "ordinal", "identifier", "boolean", "text"):
            try:
                value_counts = s.value_counts(dropna=True)
                top_categories = [
                    {"value": str(idx), "count": int(cnt), "percentage": f"{(cnt/len(s))*100:.2f}%"}
                    for idx, cnt in value_counts.head(10).items()  # Top 10 categories
                ]

                # For categorical and ordinal roles, also compute additional stats
                if role in ("categorical", "ordinal"):
                    stats = {
                        "unique_count": int(s.nunique()),
                        "unique_ratio": float(s.nunique() / len(s)) if len(s) > 0 else 0.0,
                        "top_category": str(value_counts.index[0]) if not value_counts.empty else None,
                        "top_category_count": int(value_counts.iloc[0]) if not value_counts.empty else 0,
                        "top_category_percentage": float((value_counts.iloc[0] / len(s)) * 100) if len(s) > 0 and not value_counts.empty else 0.0,
                        "count": int(len(s))
                    }
                elif role == "identifier":
                    stats = {
                        "unique_count": int(s.nunique()),
                        "unique_ratio": float(s.nunique() / len(s)) if len(s) > 0 else 0.0,
                        "is_unique": bool(s.nunique() == len(s)),
                        "count": int(len(s))
                    }
                elif role == "boolean":
                    stats = {
                        "true_count": int((s == True).sum()),
                        "false_count": int((s == False).sum()),
                        "true_ratio": float((s == True).sum() / len(s)) if len(s) > 0 else 0.0,
                        "count": int(len(s))
                    }
            except Exception as e:
                logger.warning(f"Error computing categorical stats for {col}: {e}")
                top_categories = []
                stats = None

        # Build the column profile
        column_profile = {
            "name": col,
            "dtype": str(s.dtype),
            "role": role,
            "confidence": float(confidence),
            "confidence_factors": confidence_factors,
            "semantic_tags": semantic_tags,
            "missing_count": int(s.isna().sum()),
            "unique_count": int(s.nunique()),
            "stats": stats,
            "top_categories": top_categories,
            "provenance": provenance
        }

        columns.append(column_profile)

    logger.info(f"Dataset profile built for {len(columns)} out of {n_cols} total columns")
    logger.info(f"Role counts: {role_counts}")

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "role_counts": role_counts,
        "columns": columns,
    }


def basic_profile(df: pd.DataFrame, max_cols: int = 10) -> List[Dict[str, Any]]:
    """
    Basic profile per column (maintained for API compatibility).
    """
    if df.empty:
        logger.warning("DataFrame is empty, returning empty profile")
        return []

    profile = []
    max_cols = min(max_cols, len(df.columns))

    for i, col in enumerate(df.columns):
        if i >= max_cols:
            break

        series = df[col]

        # Determine role using our advanced function
        role, confidence, provenance, confidence_factors, semantic_tags = _infer_role_advanced(
            series, uniqueness_cutoff=UNIQUENESS_CUTOFF, avg_length_cutoff=AVG_LENGTH_CUTOFF
        )

        # Compute basic stats based on the detected role
        basic_stats = {}

        if role in ("numeric", "numeric_duration", "numeric_currency", "numeric_percentage", "numeric_mixed"):
            s_clean = pd.to_numeric(series, errors='coerce')
            s_clean = s_clean.dropna()
            if len(s_clean) > 0:
                basic_stats = {
                    "min": float(s_clean.min()),
                    "max": float(s_clean.max()),
                    "mean": float(s_clean.mean()),
                    "std": float(s_clean.std()) if len(s_clean) > 1 else 0.0,
                    "count": int(len(s_clean))
                }

        profile.append({
            "column": col,
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique()),
            "role": role,
            "confidence": float(confidence),
            "semantic_tags": semantic_tags,
            "stats": basic_stats,
            "provenance": provenance
        })

    logger.info(f"Basic profile generated for {len(profile)} columns")
    return profile
```

## src/ml/kpi_generator.py

```
"""
Advanced KPI generator with robust column analysis and semantic understanding.
Addresses core issues with misidentification of IDs, inappropriate correlations,
and meaningless KPI scoring that plagued the original implementation.
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats

logger = logging.getLogger(__name__)

def _semantic_column_analysis(df: pd.DataFrame, column_name: str) -> List[str]:
    """
    Analyze column name semantically to identify potential meanings
    with improved accuracy and confidence scoring.
    """
    col_lower = column_name.lower()

    # Define semantic categories and their patterns with confidence scores
    semantic_patterns = {
        'monetary': {
            'patterns': [
                r'price', r'cost', r'revenue', r'sales', r'amount', r'value',
                r'income', r'expense', r'salary', r'wage', r'fee', r'charge',
                r'payment', r'profit', r'budget', r'funding', r'investment',
                r'capital', r'dividend', r'tip', r'rate', r'premium', r'total'
            ],
            'confidence_boost': 2.0
        },
        'time': {
            'patterns': [
                r'time', r'date', r'hour', r'minute', r'second', r'day',
                r'week', r'month', r'year', r'season', r'period', r'interval',
                r'morning', r'evening', r'night', r'afternoon', r'duration'
            ],
            'confidence_boost': 1.5
        },
        'identifier': {
            'patterns': [
                r'id', r'identifier', r'code', r'number', r'num', r'index',
                r'key', r'uid', r'uuid', r'pk', r'ssn', r'pin', r'isbn',
                r'product', r'item', r'account', r'customer', r'user', r'client',
                r'order', r'transaction', r'invoice'
            ],
            'confidence_boost': -1.0  # Negative because identifiers should not be KPIs
        },
        'geographic': {
            'patterns': [
                r'country', r'city', r'state', r'province', r'county', r'address',
                r'location', r'coord', r'longitude', r'latitude', r'lat', r'lon',
                r'zip', r'postal', r'area', r'region', r'zone', r'neighborhood',
                r'continent', r'address'
            ],
            'confidence_boost': 1.2
        },
        'demographic': {
            'patterns': [
                r'age', r'gender', r'sex', r'race', r'ethnicity', r'nationality',
                r'education', r'occupation', r'marital', r'family',
                r'children', r'birth', r'death', r'life'
            ],
            'confidence_boost': 1.5
        },
        'rating': {
            'patterns': [
                r'rating', r'score', r'grade', r'rank', r'level', r'point',
                r'quality', r'satisfaction', r'review', r'feedback'
            ],
            'confidence_boost': 1.8
        },
        'quantity': {
            'patterns': [
                r'count', r'quantity', r'qty', r'volume', r'weight', r'height',
                r'width', r'length', r'depth', r'size', r'area', r'population',
                r'frequency', r'number', r'amount'
            ],
            'confidence_boost': 1.5
        },
        'percentage': {
            'patterns': [
                r'percent', r'percentage', r'pct', r'ratio', r'proportion',
                r'rate', r'fraction', r'part', r'discount', r'tax', r'interest',
                r'change'
            ],
            'confidence_boost': 1.5
        }
    }

    # Identify semantic categories with confidence scores
    identified_categories = []
    total_patterns_found = 0
    
    for category, data in semantic_patterns.items():
        category_matches = 0
        for pattern in data['patterns']:
            if re.search(pattern, col_lower):
                category_matches += 1
                total_patterns_found += 1
        if category_matches > 0:
            identified_categories.append(category)

    return identified_categories


def _is_likely_identifier(series: pd.Series, uniqueness_threshold: float = 0.95) -> bool:
    """
    Robustly detect if a series is likely an identifier based on multiple heuristics.
    """
    n_total = len(series)
    if n_total == 0:
        return False

    n_unique = series.nunique(dropna=True)
    unique_ratio = n_unique / n_total if n_total > 0 else 0.0

    # Check if column name suggests it's an ID
    name_lower = (series.name or "").lower()
    id_name_tokens = [
        "id", "identifier", "uuid", "guid", "key", "account", "user", "customer",
        "client", "booking", "transaction", "order", "invoice", "code", "number"
    ]
    looks_like_id_name = any(token in name_lower for token in id_name_tokens)

    # Check for sequential numeric patterns (common in internal IDs)
    if pd.api.types.is_numeric_dtype(series):
        numeric_values = pd.to_numeric(series, errors='coerce').dropna()
        if len(numeric_values) > 2:
            sorted_values = numeric_values.sort_values()
            diffs = sorted_values.diff().dropna()
            # If mostly step of 1, this is likely an internal sequential ID
            sequential_ratio = (diffs == 1).mean() if len(diffs) > 0 else 0.0
            if sequential_ratio > 0.8:
                return True

    # Check for UUID patterns in string values
    if series.dtype == 'object':
        sample_values = series.dropna().head(min(50, len(series))).astype(str)
        uuid_matches = 0
        for val in sample_values:
            if re.match(r'^[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}$', val, re.IGNORECASE):
                uuid_matches += 1
        uuid_confidence = uuid_matches / len(sample_values) if len(sample_values) > 0 else 0
        
        if uuid_confidence > 0.5:
            return True

    # Use combination of uniqueness and name/context clues
    if unique_ratio > uniqueness_threshold:
        if looks_like_id_name:
            return True
        elif unique_ratio > 0.99:  # Very high uniqueness might indicate ID anyway
            return True

    return False


def _calculate_outliers(series: pd.Series, method: str = 'iqr') -> int:
    """
    Calculate outliers in a numeric series using IQR method (or other methods).
    """
    series = pd.to_numeric(series, errors='coerce').dropna()
    if len(series) < 4:  # Need at least 4 points for meaningful outlier detection
        return 0

    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR if pd.notna(IQR) and IQR != 0 else Q1 - 1.5
        upper_bound = Q3 + 1.5 * IQR if pd.notna(IQR) and IQR != 0 else Q3 + 1.5
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return len(outliers)

    return 0


def _calculate_distribution_metrics(series: pd.Series) -> Tuple[float, float]:
    """
    Calculate distribution metrics: skewness and kurtosis using pandas.
    Returns (skewness, kurtosis) or (0, 0) if not enough data.
    """
    series = pd.to_numeric(series, errors='coerce').dropna()
    if len(series) < 4:  # Need at least 4 points for meaningful distribution metrics
        return 0.0, 0.0

    try:
        skewness = series.skew()
        kurtosis = series.kurtosis()
        # Ensure valid values
        skewness = float(skewness) if pd.notna(skewness) and np.isfinite(skewness) else 0.0
        kurtosis = float(kurtosis) if pd.notna(kurtosis) and np.isfinite(kurtosis) else 0.0
        return skewness, kurtosis
    except Exception as e:
        logger.warning(f"Error calculating distribution metrics: {e}")
        return 0.0, 0.0


def _calculate_correlations(df: pd.DataFrame) -> List[Tuple[float, float, str, str]]:
    """
    Calculate correlation matrix for meaningful numeric columns with filtering to avoid spurious correlations.
    Returns list of correlations as (absolute_corr, corr_value, col1, col2).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Filter to only meaningful numeric columns (not IDs/codes)
    meaningful_numeric_cols = []
    for col in numeric_cols:
        series = df[col]
        
        # Skip if it's likely an identifier
        if _is_likely_identifier(series):
            continue
            
        # Skip if it has very low variance (essentially constant)
        std_val = pd.to_numeric(series, errors='coerce').std()
        if pd.isna(std_val) or std_val < 0.001:  # Essentially constant
            continue
            
        # Skip if it has too many unique values that look like codes/IDs
        unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
        if unique_ratio > 0.95 and len(series) > 10:
            # Check if these high-cardinality values are mostly integer-like (indicating IDs)
            numeric_values = pd.to_numeric(series, errors='coerce')
            integer_ratio = (numeric_values == numeric_values.round()).sum() / len(numeric_values) if len(numeric_values) > 0 else 0
            if integer_ratio > 0.9:
                continue  # Likely ID field, skip
                
        meaningful_numeric_cols.append(col)

    if len(meaningful_numeric_cols) < 2:
        logger.info(f"Not enough meaningful numeric columns to calculate correlations (found {len(meaningful_numeric_cols)})")
        return []

    corr_matrix = df[meaningful_numeric_cols].corr()

    # Get pairs with highest absolute correlation (excluding self-correlations)
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            
            # Only include if correlation is valid and meaningful (not too close to 0)
            if pd.notna(corr_value) and abs(corr_value) > 0.1:  # Only correlations above threshold
                correlations.append((abs(corr_value), corr_value, col1, col2))

    # Sort by absolute correlation value (highest first)
    correlations.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"Found {len(correlations)} meaningful correlation pairs")
    return correlations


def _calculate_significance_score(series: pd.Series, semantic_categories: List[str]) -> float:
    """
    Calculate a significance score for a column based on statistical properties and semantic meaning.
    Higher scores indicate more important KPIs.
    """
    if series.empty or series.isna().all():
        return 0.0

    # Check if this is likely an identifier (should have low significance)
    if _is_likely_identifier(series):
        return 0.05  # Very low score for identifiers

    score = 0.0

    # Statistical significance factors
    n_valid = len(series.dropna())
    if n_valid == 0:
        return 0.0

    # Variability score: meaningful metrics tend to have variability
    if pd.api.types.is_numeric_dtype(series):
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        if len(numeric_series) > 1:
            std_dev = numeric_series.std()
            mean_val = numeric_series.mean()
            
            if pd.notna(std_dev) and pd.notna(mean_val) and mean_val != 0:
                cv = abs(std_dev / mean_val)  # Coefficient of variation
                score += min(0.5, cv)  # Cap at 0.5 to prevent extreme scores
            elif pd.notna(std_dev):
                score += min(0.5, std_dev / 10)  # If mean is 0, use raw std deviation capped

    # Uniqueness score: avoid both too unique (IDs) and too uniform (constants)
    n_unique = series.nunique(dropna=True)
    unique_ratio = n_unique / n_valid if n_valid > 0 else 0.0

    # Score peaks at medium uniqueness, penalizes extreme uniqueness (like IDs) or low uniqueness (like constants)
    if 0.05 <= unique_ratio <= 0.90:  # Good range for meaningful categorical variables
        uniqueness_bonus = 0.2
        # Further boost if it's in the sweet spot (not too many, not too few categories)
        if 2 <= n_unique <= 20:  # Ideal range for categorical KPIs
            uniqueness_bonus += 0.1
        score += uniqueness_bonus
    elif unique_ratio > 0.95:  # Probably an ID, reduce score
        score -= 0.3

    # Semantic significance boosts
    for semantic_cat in semantic_categories:
        if semantic_cat in ['monetary', 'rating', 'quantity']:
            score += 0.3  # High importance semantic categories
        elif semantic_cat in ['demographic', 'percentage']:
            score += 0.2  # Medium importance
        elif semantic_cat in ['time']:
            score += 0.15  # Time fields are often important

    # Distribution shape significance (for numeric fields)
    if pd.api.types.is_numeric_dtype(series):
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        if len(numeric_series) >= 4:  # Need enough points for distribution metrics
            skewness, kurtosis = _calculate_distribution_metrics(numeric_series)
            if abs(skewness) > 1.0:  # Highly skewed - might be important for analysis
                score += 0.1
            if abs(kurtosis) > 1.0:  # Heavy or light-tailed - might be important
                score += 0.1

    # Outlier significance
    if pd.api.types.is_numeric_dtype(series):
        outlier_count = _calculate_outliers(series)
        if outlier_count > 0:
            outlier_ratio = outlier_count / n_valid if n_valid > 0 else 0
            # More outliers might indicate interesting phenomena
            score += min(0.1, outlier_ratio * 0.2)  # Cap the outlier boost

    # Ensure score is between 0 and 1
    score = max(0.0, min(1.0, score))
    return score


def generate_kpis(df: pd.DataFrame, dataset_profile: Dict[str, Any],
                 min_variability_threshold: float = 0.01,
                 min_unique_ratio: float = 0.01,
                 max_unique_ratio: float = 0.9,
                 top_k: int = 10,
                 eda_summary: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Enhanced KPIs with robust statistical metrics and semantic understanding.
    Addresses issues with misidentified IDs, inappropriate correlations, and 
    meaningless KPI scoring.

    Args:
        df: Input DataFrame
        dataset_profile: Dataset profile from analyser
        min_variability_threshold: Minimum standard deviation to consider a column variable
        min_unique_ratio: Minimum unique ratio for categorical columns
        max_unique_ratio: Maximum unique ratio (to avoid IDs)
        top_k: Number of top KPIs to return

    Returns:
        List of KPI dictionaries with significance scores and semantic meaning
    """
    if df.empty:
        logger.warning("Empty dataframe provided to KPI generator")
        return []

    logger.info(f"Generating enhanced KPIs with parameters: min_variability_threshold={min_variability_threshold}, "
                f"min_unique_ratio={min_unique_ratio}, max_unique_ratio={max_unique_ratio}, top_k={top_k}")

    kpis = []
    columns = dataset_profile["columns"]
    n_rows = dataset_profile.get("n_rows", len(df))

    if n_rows == 0:
        logger.warning("No data rows provided to KPI generator")
        return []

    # Prepare a comprehensive analysis of each column
    column_analyses = []
    
    for col in columns:
        col_name = col["name"]
        series = df[col_name]
        
        # Skip if series is all NaN
        if series.isna().all():
            continue
            
        # Determine if this is likely an identifier
        is_identifier = _is_likely_identifier(series)
        
        # Perform semantic analysis
        semantic_categories = _semantic_column_analysis(df, col_name)
        
        # Calculate significance score
        significance_score = _calculate_significance_score(series, semantic_categories)
        
        # Additional metrics for KPI characterization
        n_unique = series.nunique(dropna=True)
        n_valid = len(series.dropna())
        unique_ratio = n_unique / n_valid if n_valid > 0 else 0.0
        
        # Calculate basic statistics for numeric fields
        numeric_stats = {}
        if pd.api.types.is_numeric_dtype(series):
            numeric_series = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_series) > 0:
                numeric_stats = {
                    'mean': float(numeric_series.mean()) if len(numeric_series) > 0 else None,
                    'std': float(numeric_series.std()) if len(numeric_series) > 1 else 0.0,
                    'min': float(numeric_series.min()) if len(numeric_series) > 0 else None,
                    'max': float(numeric_series.max()) if len(numeric_series) > 0 else None,
                    'median': float(numeric_series.median()) if len(numeric_series) > 0 else None,
                }
        
        # Count outliers if numeric
        outlier_count = _calculate_outliers(series) if pd.api.types.is_numeric_dtype(series) else 0
        
        column_analyses.append({
            'name': col_name,
            'role': col.get('role', 'unknown'),
            'semantic_categories': semantic_categories,
            'is_identifier': is_identifier,
            'significance_score': significance_score,
            'n_unique': n_unique,
            'n_valid': n_valid,
            'unique_ratio': unique_ratio,
            'numeric_stats': numeric_stats,
            'outlier_count': outlier_count
        })

    # Sort by significance score to identify the most important columns
    column_analyses.sort(key=lambda x: x['significance_score'], reverse=True)
    
    # Generate KPIs based on the analyses
    for analysis in column_analyses[:top_k]:  # Take top K by significance
        col_name = analysis['name']
        col_role = analysis['role']
        significance_score = analysis['significance_score']
        
        # Skip identifiers as they are not meaningful KPIs
        if analysis['is_identifier']:
            continue
            
        # Prepare KPI information
        kpi_info = {
            "label": col_name,
            "type": col_role,
            "significance_score": significance_score,
            "semantic_categories": analysis['semantic_categories'],
            "provenance": "enhanced_analysis"
        }
        
        # Add specific metrics based on the column's role and semantic categories
        series = df[col_name]
        
        if col_role == 'numeric':
            numeric_stats = analysis['numeric_stats']
            if numeric_stats and numeric_stats.get('mean') is not None:
                avg_val = numeric_stats['mean']
                kpi_info["value"] = f"{avg_val:.2f} (±{numeric_stats['std']:.2f})"
            else:
                # Compute from series directly if stats weren't cached
                numeric_series = pd.to_numeric(series, errors='coerce').dropna()
                if len(numeric_series) > 0:
                    mean_val = float(numeric_series.mean())
                    std_val = float(numeric_series.std()) if len(numeric_series) > 1 else 0.0
                    kpi_info["value"] = f"{mean_val:.2f} (±{std_val:.2f})"
                else:
                    kpi_info["value"] = "No valid numeric values"

            # Add outlier information
            if analysis['outlier_count'] > 0:
                kpi_info["value"] += f" [{analysis['outlier_count']} outliers]"

        elif col_role in ['categorical', 'text']:
            # For categorical/text, show top value and distribution
            top_value_counts = series.value_counts(dropna=True).head(3)
            if len(top_value_counts) > 0:
                top_val = top_value_counts.index[0]
                top_count = top_value_counts.iloc[0]
                total_valid = analysis['n_valid']
                pct_top = (top_count / total_valid) * 100 if total_valid > 0 else 0
                kpi_info["value"] = f"Top: '{top_val}' ({top_count}, {pct_top:.1f}%)"
            else:
                kpi_info["value"] = "No valid values"
                
        elif col_role == 'datetime':
            # For datetime, show time range
            try:
                datetime_series = pd.to_datetime(series, errors='coerce').dropna()
                if len(datetime_series) > 0:
                    min_date = datetime_series.min().strftime('%Y-%m-%d')
                    max_date = datetime_series.max().strftime('%Y-%m-%d')
                    kpi_info["value"] = f"Range: {min_date} to {max_date}"
                else:
                    kpi_info["value"] = "No valid dates"
            except Exception as e:
                logger.warning(f"Error processing datetime column {col_name}: {e}")
                kpi_info["value"] = "Invalid datetime format"
                
        else:
            # For other roles, show basic statistics
            kpi_info["value"] = f"Unique: {analysis['n_unique']}, Missing: {series.isna().sum()}"
        
        # Add semantic context to the value description if relevant
        if analysis['semantic_categories']:
            semantic_info = ", ".join(analysis['semantic_categories'])
            kpi_info["value"] += f" [{semantic_info}]"
            
        kpis.append(kpi_info)

    # Additionally, add some correlation-based KPIs
    try:
        correlations = _calculate_correlations(df)
        for abs_corr, corr_val, col1, col2 in correlations[:3]:  # Top 3 correlations
            kpis.append({
                "label": f"Correlation: {col1} ↔ {col2}",
                "value": f"{corr_val:.3f}",
                "type": "correlation",
                "correlation_value": corr_val,
                "columns": [col1, col2],
                "significance_score": abs(corr_val),  # Correlation strength as significance
                "provenance": "correlation_insight"
            })
    except Exception as e:
        logger.warning(f"Error calculating correlations: {e}")

    logger.info(f"Generated {len(kpis)} enhanced KPIs")
    return kpis


def generate_basic_kpis(df: pd.DataFrame, dataset_profile: Dict[str, Any],
                       min_variability_threshold: float = 0.01,
                       min_unique_ratio: float = 0.01,
                       max_unique_ratio: float = 0.9,
                       top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Wrapper function to maintain backward compatibility while using the enhanced generator.
    """
    return generate_kpis(
        df=df,
        dataset_profile=dataset_profile,
        min_variability_threshold=min_variability_threshold,
        min_unique_ratio=min_unique_ratio,
        max_unique_ratio=max_unique_ratio,
        top_k=top_k
    )
```

## src/ml/chart_selector.py

```
"""
Advanced chart selection engine with semantic awareness and meaningful visualization recommendations.
Replaces naive heuristics with context-aware chart suggestions based on column roles, 
semantic meanings, and statistical properties.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from scipy.stats import chi2_contingency
import math
import re

logger = logging.getLogger(__name__)

def _analyze_column_for_viz(series: pd.Series, role: str, semantic_tags: List[str] = []) -> Dict[str, Any]:
    """
    Analyze a column's characteristics for appropriate visualization selection.
    
    Args:
        series: The pandas Series to analyze
        role: The semantic role of the column ('numeric', 'categorical', 'datetime', etc.)
        semantic_tags: List of semantic tags associated with the column
        
    Returns:
        Dictionary with analysis results including suitability for different chart types
    """
    n_total = len(series)
    if n_total == 0:
        return {"suitable_charts": [], "stats": {}, "confidence": 0.0}
    
    n_unique = series.nunique(dropna=True)
    unique_ratio = n_unique / n_total if n_total > 0 else 0.0
    missing_ratio = series.isna().sum() / n_total if n_total > 0 else 0.0
    
    analysis = {
        "role": role,
        "semantic_tags": semantic_tags,
        "n_total": n_total,
        "n_unique": n_unique,
        "unique_ratio": unique_ratio,
        "missing_ratio": missing_ratio,
        "suitable_charts": [],
        "stats": {},
        "confidence": 0.7  # Default medium confidence
    }
    
    # Add role-specific statistics
    if role == "numeric":
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        if len(numeric_series) > 0:
            analysis["stats"] = {
                "min": float(numeric_series.min()) if pd.notna(numeric_series.min()) else None,
                "max": float(numeric_series.max()) if pd.notna(numeric_series.max()) else None,
                "mean": float(numeric_series.mean()) if pd.notna(numeric_series.mean()) else None,
                "std": float(numeric_series.std()) if len(numeric_series) > 1 and pd.notna(numeric_series.std()) else 0.0,
                "median": float(numeric_series.median()) if pd.notna(numeric_series.median()) else None,
                "q25": float(numeric_series.quantile(0.25)) if len(numeric_series) > 0 and pd.notna(numeric_series.quantile(0.25)) else None,
                "q75": float(numeric_series.quantile(0.75)) if len(numeric_series) > 0 and pd.notna(numeric_series.quantile(0.75)) else None,
                "skewness": float(numeric_series.skew()) if len(numeric_series) > 2 and pd.notna(numeric_series.skew()) else 0.0,
                "kurtosis": float(numeric_series.kurtosis()) if len(numeric_series) > 3 and pd.notna(numeric_series.kurtosis()) else 0.0
            }
            # Determine suitable charts for numeric data based on characteristics
            std_val = analysis["stats"]["std"]
            if std_val > 0.001:  # Has meaningful variance
                analysis["suitable_charts"].extend(["histogram", "box_plot", "scatter", "line"])
                analysis["confidence"] = 0.9  # High confidence for well-behaved numeric data
            else:
                analysis["suitable_charts"].append("summary_stat")  # No meaningful variance
                analysis["confidence"] = 0.3  # Low confidence for constant data
    elif role in ["categorical", "text"]:
        # Determine suitable charts for categorical data based on cardinality
        if n_unique <= 2:
            # Binary categorical - pie, bar charts are suitable
            analysis["suitable_charts"].extend(["bar", "pie", "donut"])
            analysis["confidence"] = 0.8  # High confidence for binary categorical
        elif n_unique <= 10:
            # Low cardinality categorical - bar is excellent, pie is okay
            analysis["suitable_charts"].extend(["bar", "pie", "stacked_bar"])
            analysis["confidence"] = 0.85  # High confidence for low-cardinality categorical
        elif n_unique <= 50:
            # Medium cardinality - bar charts are still appropriate
            analysis["suitable_charts"].extend(["bar", "horizontal_bar", "top_categories"])
            analysis["confidence"] = 0.7  # Medium-high confidence
        else:
            # High cardinality - only show top categories or summary, avoid pie charts
            analysis["suitable_charts"].extend(["top_categories", "summary_stat"])
            analysis["confidence"] = 0.4  # Lower confidence due to high cardinality
    elif role == "datetime":
        analysis["suitable_charts"].extend(["line", "time_series", "calendar_heatmap", "bar"])
        analysis["confidence"] = 0.9  # High confidence for datetime (good for time series)
    
    # Apply confidence adjustments based on data quality
    if missing_ratio > 0.5:
        analysis["confidence"] *= 0.6  # Reduce confidence for high missingness
    elif missing_ratio > 0.2:
        analysis["confidence"] *= 0.8  # Moderate reduction for moderate missingness

    if unique_ratio > 0.98:  # Potential identifier
        analysis["confidence"] *= 0.4  # Significant reduction if likely an identifier
        
    analysis["confidence"] = max(0.1, min(1.0, analysis["confidence"]))  # Clamp to [0.1, 1.0]
    
    return analysis


def _is_likely_identifier_with_confidence(s: pd.Series, name: str) -> Tuple[bool, str, float]:
    """
    Check if a series is likely an identifier with confidence scoring.
    
    Args:
        s: The pandas Series to analyze
        name: The column name
        
    Returns:
        Tuple of (is_identifier, detection_method, confidence_score)
    """
    n_total = len(s)
    if n_total == 0:
        return False, "empty", 0.0

    n_unique = s.nunique()
    unique_ratio = n_unique / n_total if n_total > 0 else 0.0

    detection_signals = {}

    # Signal 1: High cardinality (potential ID)
    if unique_ratio > 0.98:
        detection_signals["very_high_cardinality"] = min(0.95, unique_ratio)
    elif unique_ratio > 0.95:
        detection_signals["high_cardinality"] = min(0.85, unique_ratio * 0.9)
    elif unique_ratio > 0.90:
        detection_signals["moderate_cardinality"] = unique_ratio * 0.6

    # Signal 2: Sequential numeric pattern (common in internal IDs)
    if pd.api.types.is_numeric_dtype(s):
        numeric_vals = pd.to_numeric(s, errors='coerce').dropna()
        if len(numeric_vals) > 5:  # Need at least 5 values to check sequence
            sorted_vals = numeric_vals.sort_values()
            diffs = sorted_vals.diff().dropna()
            if len(diffs) > 0:
                # Check for mostly constant differences (sequential IDs)
                unique_diffs = diffs.unique()
                if len(unique_diffs) == 1 and abs(unique_diffs[0] - 1) < 0.01:  # Step of 1
                    detection_signals["sequential_step1"] = min(0.95, len(numeric_vals) / max(len(numeric_vals), 10))
                elif len(unique_diffs) <= 3 and diffs.std() < diffs.mean() * 0.1:  # Low variance in steps
                    detection_signals["sequential_low_variance"] = min(0.85, diffs.mean() * 0.7)

    # Signal 3: UUID pattern
    if s.dtype == 'object':
        sample = s.dropna().head(20).astype(str)
        uuid_matches = 0
        for val in sample:
            # Check for UUID v4 pattern (with case insensitivity)
            if re.match(r'^[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}$', val, re.IGNORECASE):
                uuid_matches += 1
        if len(sample) > 0:
            uuid_ratio = uuid_matches / len(sample)
            if uuid_ratio > 0.5:  # More than 50% are UUIDs
                detection_signals["uuid_pattern"] = uuid_ratio

    # Signal 4: Name-based detection (semantic heuristics)
    name_lower = name.lower()
    id_keywords = [
        "id", "uuid", "guid", "key", "code", "no", "number", "index",
        "account", "user", "customer", "product", "item", "order",
        "transaction", "invoice", "booking", "session", "token", "hash"
    ]

    matching_keywords = [kw for kw in id_keywords if kw in name_lower]
    if matching_keywords:
        # Calculate confidence based on how many keywords match and their position in name
        keyword_confidence = min(0.8, len(matching_keywords) * 0.3)
        # Boost confidence if important keywords are found
        important_keywords = ["id", "uuid", "key", "code", "account", "user", "customer"]
        important_matches = sum(1 for kw in matching_keywords if kw in important_keywords)
        keyword_confidence += important_matches * 0.15
        detection_signals["name_pattern"] = min(1.0, keyword_confidence)

    # Calculate overall confidence based on signal strengths and weights
    if detection_signals:
        # Weight different signals appropriately
        weights = {
            "uuid_pattern": 1.0,              # Highest confidence for UUIDs
            "sequential_step1": 0.95,         # High confidence for clear sequential patterns
            "very_high_cardinality": 0.9,     # High confidence for extremely high uniqueness
            "sequential_low_variance": 0.85,  # High confidence for sequential patterns
            "high_cardinality": 0.8,          # Good confidence for high uniqueness
            "name_pattern": 0.75,             # Good confidence for name patterns
            "moderate_cardinality": 0.4       # Lower confidence for moderate uniqueness
        }

        max_confidence = 0
        best_signal = ""

        for signal, score in detection_signals.items():
            weight = weights.get(signal, 0.6)  # Default weight of 0.6
            weighted_score = score * weight
            if weighted_score > max_confidence:
                max_confidence = min(1.0, weighted_score)
                best_signal = signal

        # Consider it an identifier if confidence exceeds threshold
        is_identifier = max_confidence > 0.6

        return is_identifier, best_signal, max_confidence

    return False, "no_signals", 0.0


def _is_likely_identifier_with_confidence(s: pd.Series, name: str) -> Tuple[bool, str, float]:
    """
    Check if a series is likely an identifier with confidence scoring.

    Args:
        s: The pandas Series to analyze
        name: The column name

    Returns:
        Tuple of (is_identifier, detection_method, confidence_score)
    """
    n_total = len(s)
    if n_total == 0:
        return False, "empty", 0.0

    n_unique = s.nunique()
    unique_ratio = n_unique / n_total if n_total > 0 else 0.0

    detection_signals = {}

    # Signal 1: High cardinality (potential ID)
    if unique_ratio > 0.98:
        detection_signals["very_high_cardinality"] = min(0.95, unique_ratio)
    elif unique_ratio > 0.95:
        detection_signals["high_cardinality"] = min(0.85, unique_ratio * 0.9)
    elif unique_ratio > 0.90:
        detection_signals["moderate_cardinality"] = unique_ratio * 0.6

    # Signal 2: Sequential numeric pattern (common in internal IDs)
    if pd.api.types.is_numeric_dtype(s):
        numeric_vals = pd.to_numeric(s, errors='coerce').dropna()
        if len(numeric_vals) > 5:  # Need at least 5 values to check sequence
            sorted_vals = numeric_vals.sort_values()
            diffs = sorted_vals.diff().dropna()
            if len(diffs) > 0:
                # Check for mostly constant differences (sequential IDs)
                unique_diffs = diffs.unique()
                if len(unique_diffs) == 1 and abs(unique_diffs[0] - 1) < 0.01:  # Step of 1
                    detection_signals["sequential_step1"] = min(0.95, len(numeric_vals) / max(len(numeric_vals), 10))
                elif len(unique_diffs) <= 3 and diffs.std() < diffs.mean() * 0.1:  # Low variance in steps
                    detection_signals["sequential_low_variance"] = min(0.85, diffs.mean() * 0.7)

    # Signal 3: UUID pattern
    if s.dtype == 'object':
        sample = s.dropna().head(20).astype(str)
        uuid_matches = 0
        for val in sample:
            # Check for UUID v4 pattern (with case insensitivity)
            if re.match(r'^[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}$', val, re.IGNORECASE):
                uuid_matches += 1
        if len(sample) > 0:
            uuid_ratio = uuid_matches / len(sample)
            if uuid_ratio > 0.5:  # More than 50% are UUIDs
                detection_signals["uuid_pattern"] = uuid_ratio

    # Signal 4: Name-based detection (semantic heuristics)
    name_lower = name.lower()
    id_keywords = [
        "id", "uuid", "guid", "key", "code", "no", "number", "index",
        "account", "user", "customer", "product", "item", "order",
        "transaction", "invoice", "booking", "session", "token", "hash"
    ]

    matching_keywords = [kw for kw in id_keywords if kw in name_lower]
    if matching_keywords:
        # Calculate confidence based on how many keywords match and their position in name
        keyword_confidence = min(0.8, len(matching_keywords) * 0.3)
        # Boost confidence if important keywords are found
        important_keywords = ["id", "uuid", "key", "code", "account", "user", "customer"]
        important_matches = sum(1 for kw in matching_keywords if kw in important_keywords)
        keyword_confidence += important_matches * 0.15
        detection_signals["name_pattern"] = min(1.0, keyword_confidence)

    # Calculate overall confidence based on signal strengths and weights
    if detection_signals:
        # Weight different signals appropriately
        weights = {
            "uuid_pattern": 1.0,              # Highest confidence for UUIDs
            "sequential_step1": 0.95,         # High confidence for clear sequential patterns
            "very_high_cardinality": 0.9,     # High confidence for extremely high uniqueness
            "sequential_low_variance": 0.85,  # High confidence for sequential patterns
            "high_cardinality": 0.8,          # Good confidence for high uniqueness
            "name_pattern": 0.75,             # Good confidence for name patterns
            "moderate_cardinality": 0.4       # Lower confidence for moderate uniqueness
        }

        max_confidence = 0
        best_signal = ""

        for signal, score in detection_signals.items():
            weight = weights.get(signal, 0.6)  # Default weight of 0.6
            weighted_score = score * weight
            if weighted_score > max_confidence:
                max_confidence = min(1.0, weighted_score)
                best_signal = signal

        # Consider it an identifier if confidence exceeds threshold
        is_identifier = max_confidence > 0.6

        return is_identifier, best_signal, max_confidence

    return False, "no_signals", 0.0


def _is_likely_identifier(s: pd.Series, name: str) -> bool:
    """
    Simplified function to check if a series is likely an identifier.
    Uses the confidence-based function internally but returns only a boolean.
    """
    is_id, _, confidence = _is_likely_identifier_with_confidence(s, name)
    return is_id


def _is_multi_value_field(series: pd.Series, delimiter_chars: List[str] = [',', ';', '|', '/', ' | ']) -> Tuple[bool, str, float]:
    """
    Detect if a field contains multiple values separated by delimiters.
    
    Args:
        series: The pandas Series to analyze
        delimiter_chars: List of potential delimiter characters
        
    Returns:
        Tuple of (is_multi_value, primary_delimiter, confidence_score)
    """
    if series.dtype != 'object' and not str(series.dtype).startswith('string'):
        return False, "", 0.0

    n_total = len(series)
    if n_total == 0:
        return False, "", 0.0

    sample_size = min(100, n_total)
    sample_values = series.dropna().head(sample_size).astype(str)

    if len(sample_values) == 0:
        return False, "", 0.0

    delimiter_scores = {}

    for delim in delimiter_chars:
        # Count how many values contain the delimiter
        contains_delim = sample_values.str.contains(delim, na=False, regex=False)
        delimiter_scores[delim] = contains_delim.sum() / len(sample_values)

    if delimiter_scores:
        # Get the delimiter with highest ratio
        best_delimiter = max(delimiter_scores, key=delimiter_scores.get)
        best_ratio = delimiter_scores[best_delimiter]
        
        # Additional validation: see if splitting creates multiple meaningful parts
        if best_ratio > 0.1:  # At least 10% of values contain the delimiter
            sample_with_delim = sample_values[sample_values.str.contains(best_delimiter, na=False, regex=False)]
            if len(sample_with_delim) > 0:
                avg_parts = sample_with_delim.str.split(best_delimiter).apply(len).mean()
                if avg_parts > 1.5:  # On average more than 1 part after splitting
                    # Confidence is based on both ratio and average number of parts
                    confidence = min(1.0, best_ratio * avg_parts * 0.7)
                    return True, best_delimiter, confidence

    return False, "", 0.0


def _is_meaningful_for_correlation(series1: pd.Series, series2: pd.Series,
                                 df: pd.DataFrame, col1: str, col2: str) -> bool:
    """
    Determine if two columns are meaningful to correlate together.

    Args:
        series1, series2: The two series to evaluate
        df: The dataframe they come from
        col1, col2: Their respective column names

    Returns:
        True if correlation between these columns would be meaningful
    """
    # If either series is likely an identifier, correlation is not meaningful
    if _is_likely_identifier(series1, col1) or _is_likely_identifier(series2, col2):
        return False

    # Check if both are numeric (requirement for standard correlation)
    if not (pd.api.types.is_numeric_dtype(series1) and pd.api.types.is_numeric_dtype(series2)):
        return False

    # Check if both have sufficient variance to be meaningful
    clean_s1 = pd.to_numeric(series1, errors='coerce').dropna()
    clean_s2 = pd.to_numeric(series2, errors='coerce').dropna()

    if len(clean_s1) < 3 or len(clean_s2) < 3:
        return False  # Need at least 3 points for meaningful correlation

    std1 = clean_s1.std()
    std2 = clean_s2.std()

    if pd.isna(std1) or std1 < 0.001 or pd.isna(std2) or std2 < 0.001:
        return False  # One or both have very little variance

    # Align indices to ensure same data points
    aligned_df = pd.concat([clean_s1, clean_s2], axis=1).dropna()

    if len(aligned_df) < 3:  # Need at least 3 aligned points for meaningful correlation
        return False

    s1_aligned = aligned_df.iloc[:, 0]
    s2_aligned = aligned_df.iloc[:, 1]

    if len(s1_aligned) == 0 or len(s2_aligned) == 0 or len(s1_aligned) != len(s2_aligned):
        return False

    try:
        correlation = s1_aligned.corr(s2_aligned)
        # Only consider correlations meaningful if abs(correlation) > 0.1
        return abs(correlation) > 0.1 if not pd.isna(correlation) else False
    except Exception:
        return False


def _suggest_appropriate_charts_for_columns(df: pd.DataFrame, dataset_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Suggest charts based on column analysis and semantic understanding with identifier filtering.
    
    Args:
        df: The input DataFrame
        dataset_profile: Profile containing column information including roles and semantic tags
        
    Returns:
        List of chart specifications with appropriate chart types and fields
    """
    charts = []
    columns = dataset_profile["columns"]
    
    # Group columns by their roles for appropriate chart suggestions
    numeric_cols = []
    categorical_cols = []
    datetime_cols = []
    identifier_cols = []
    text_cols = []

    for col in columns:
        role = col["role"]
        unique_count = col.get("unique_count", 0)
        
        # First check if it looks like an identifier regardless of role
        series_sample = df[col["name"]].head(100)  # Sample for checking
        is_id, _, id_conf = _is_likely_identifier_with_confidence(series_sample, col["name"])

        if is_id and id_conf > 0.6:  # Higher threshold to be more conservative
            identifier_cols.append(col)
        elif role == "numeric":
            # Verify that it's truly meaningful numeric data (not an ID disguised as a number)
            series = df[col["name"]]
            unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
            if unique_ratio < 0.95:  # Exclude columns that are almost all unique (likely IDs)
                numeric_cols.append(col)
        elif role in ["categorical", "text"]:
            # Check if it's a multi-value text field
            series = df[col["name"]]
            is_multi, delimiter, multi_conf = _is_multi_value_field(series)
            
            if is_multi and multi_conf > 0.3:
                # Multi-value fields should be treated specially if not too many unique combinations
                if unique_count <= 50:
                    categorical_cols.append(col)
            elif unique_count <= 50:  # Low cardinality categorical/text
                categorical_cols.append(col)
            else:
                # High cardinality - treat as text
                text_cols.append(col)
        elif role == "datetime":
            datetime_cols.append(col)
        else:
            # Default to treating as text if unknown
            text_cols.append(col)
    
    logger.info(f"Column classification: {len(numeric_cols)} numeric, "
                f"{len(categorical_cols)} categorical, "
                f"{len(datetime_cols)} datetime, "
                f"{len(identifier_cols)} identifiers, "
                f"{len(text_cols)} text")
    
    # 1. Distribution charts for meaningful numeric variables (excluding identifiers)
    for col in numeric_cols:
        col_name = col["name"]
        series = df[col_name]
        
        # Skip if this looks like an identifier (double-check)
        if any(id_col["name"] == col_name for id_col in identifier_cols):
            continue
            
        # Skip if constant or nearly constant
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        if len(numeric_series) > 1:
            std_val = numeric_series.std()
            if pd.notna(std_val) and std_val < 0.001:  # Nearly constant
                continue
        
        # Suggest distribution chart (histogram) if meaningful
        if len(numeric_series) > 5 and numeric_series.std() > 0.001:
            charts.append({
                "id": f"dist_{col_name}",
                "title": f"Distribution of {col_name.replace('_', ' ').title()}",
                "chart_type": "histogram",
                "intent": "distribution",
                "x_field": col_name,
                "y_field": None,
                "agg_func": "count",
                "priority": 1
            })
        
        # Suggest box plot if there are enough data points and meaningful variance
        if len(numeric_series) > 10 and numeric_series.std() > 0.001:
            charts.append({
                "id": f"box_{col_name}",
                "title": f"Box Plot of {col_name.replace('_', ' ').title()}",
                "chart_type": "box",
                "intent": "box_plot",
                "x_field": None,
                "y_field": col_name,
                "agg_func": None,
                "priority": 2
            })
    
    # 2. Time series charts for datetime + numeric combinations (excluding identifiers)
    for dt_col in datetime_cols:
        for num_col in numeric_cols:  # Only numeric non-identifier columns
            dt_name = dt_col["name"]
            num_name = num_col["name"]
            
            # Skip if either is an identifier
            if any(id_col["name"] == dt_name for id_col in identifier_cols) or \
               any(id_col["name"] == num_name for id_col in identifier_cols):
                continue
                
            datetime_series = pd.to_datetime(df[dt_name], errors='coerce').dropna()
            numeric_series = pd.to_numeric(df[num_name], errors='coerce')
            
            # Align the series
            aligned = pd.concat([datetime_series, numeric_series], axis=1).dropna()
            
            if len(aligned) > 2:  # Need at least 3 points for meaningful time series
                charts.append({
                    "id": f"ts_{dt_name}_{num_name}",
                    "title": f"Trend of {num_name.replace('_', ' ').title()} Over {dt_name.replace('_', ' ').title()}",
                    "chart_type": "line",
                    "intent": "time_series",
                    "x_field": dt_name,
                    "y_field": num_name,
                    "agg_func": "mean",  # Use mean aggregation for potential duplicate dates
                    "priority": 1
                })
    
    # 3. Categorical charts for low-cardinality categorical variables (excluding identifiers)
    for col in categorical_cols:
        col_name = col["name"]
        series = df[col_name]
        
        # Skip if this looks like an identifier
        if any(id_col["name"] == col_name for id_col in identifier_cols):
            continue
            
        # Get value counts for the series
        value_counts = series.value_counts(dropna=True)
        
        # Suggest bar chart if not too many categories
        if len(value_counts) > 1 and len(value_counts) <= 20:  # Not too many categories for a readable bar chart
            charts.append({
                "id": f"cat_{col_name}",
                "title": f"Count of {col_name.replace('_', ' ').title()}",
                "chart_type": "bar",
                "intent": "category_count",
                "x_field": col_name,
                "y_field": None,
                "agg_func": "count",
                "priority": 1
            })
        
        # Suggest pie chart if not too many categories (max 10 for readability)
        if len(value_counts) > 1 and len(value_counts) <= 10:
            charts.append({
                "id": f"pie_{col_name}",
                "title": f"Distribution of {col_name.replace('_', ' ').title()}",
                "chart_type": "pie",
                "intent": "category_distribution",
                "x_field": col_name,
                "y_field": None,
                "agg_func": "count",
                "priority": 2
            })
    
    # 4. Scatter plots for meaningful numeric-numeric relationships (excluding identifiers)
    numeric_non_id = [col for col in numeric_cols if not any(id_col["name"] == col["name"] for id_col in identifier_cols)]

    for i, col1 in enumerate(numeric_non_id):
        for j, col2 in enumerate(numeric_non_id[i+1:], i+1):  # Avoid duplicate pairs
            series1 = df[col1["name"]]
            series2 = df[col2["name"]]

            # Check if this correlation would be meaningful before suggesting scatter
            if _is_meaningful_for_correlation(series1, series2, df, col1["name"], col2["name"]):
                charts.append({
                    "id": f"scatter_{col1['name']}_{col2['name']}",
                    "title": f"{col1['name'].replace('_', ' ').title()} vs {col2['name'].replace('_', ' ').title()}",
                    "chart_type": "scatter",
                    "intent": "correlation",
                    "x_field": col2["name"],
                    "y_field": col1["name"],
                    "agg_func": None,
                    "priority": 3
                })

    # 5. Correlation heatmap only for meaningful numeric columns (excluding identifiers)
    meaningful_numeric_cols = [col for col in numeric_cols
                              if not any(id_col["name"] == col["name"] for id_col in identifier_cols)]
    
    if len(meaningful_numeric_cols) >= 2:
        # Extract data for only meaningful numeric columns
        column_names = [col["name"] for col in meaningful_numeric_cols]
        numeric_data = df[column_names]
        
        # Only include truly numeric columns (filter out any remaining non-numeric data)
        numeric_data = numeric_data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) >= 2:
            # Compute correlation matrix
            corr_matrix = numeric_data.corr()
            
            # Only add heatmap if we have meaningful correlations (>0.1 absolute value)
            has_meaningful_corrs = False
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if pd.notna(corr_val) and abs(corr_val) > 0.1:
                        has_meaningful_corrs = True
                        break
                if has_meaningful_corrs:
                    break
            
            if has_meaningful_corrs:
                charts.append({
                    "id": "correlation_heatmap",
                    "title": "Correlation Heatmap (Meaningful Numeric Variables)",
                    "chart_type": "heatmap",
                    "intent": "correlation_matrix",
                    "x_field": "variables",
                    "y_field": "variables",
                    "agg_func": "correlation",
                    "priority": 2
                })

    # 6. Group by charts: meaningful categorical vs numeric relationships (excluding identifiers)
    numeric_non_id = [col for col in numeric_cols if not any(id_col["name"] == col["name"] for id_col in identifier_cols)]

    for cat_col in categorical_cols:
        for num_col in numeric_non_id:  # Only numeric non-identifier columns
            cat_name = cat_col["name"]
            num_name = num_col["name"]

            # Skip if categorical column has too many unique values (would create unreadable chart)
            if cat_col["unique_count"] > 20:
                continue

            series_cat = df[cat_name]
            series_num = pd.to_numeric(df[num_name], errors='coerce')

            # Create grouped statistics
            grouped = pd.concat([series_cat, series_num], axis=1).dropna()

            if len(grouped) > 5 and grouped[cat_name].nunique() >= 2:  # Enough data and categories
                charts.append({
                    "id": f"group_bar_{cat_name}_{num_name}",
                    "title": f"Avg {num_name.replace('_', ' ').title()} by {cat_name.replace('_', ' ').title()}",
                    "chart_type": "bar",  # Bar chart showing average by category
                    "intent": "group_comparison",
                    "x_field": cat_name,
                    "y_field": num_name,
                    "agg_func": "mean",
                    "priority": 2
                })

                # Also suggest a box plot for distribution comparison if not too many categories
                if grouped[cat_name].nunique() <= 10:
                    charts.append({
                        "id": f"group_box_{cat_name}_{num_name}",
                        "title": f"Distribution of {num_name.replace('_', ' ').title()} by {cat_name.replace('_', ' ').title()}",
                        "chart_type": "box",
                        "intent": "distribution_by_category",
                        "x_field": cat_name,
                        "y_field": num_name,
                        "agg_func": None,
                        "priority": 3
                    })
    
    # Sort charts by priority (lower number means higher priority)
    charts.sort(key=lambda x: x.get("priority", 999))
    
    # Limit number of charts to prevent overwhelming the user (max 20 charts)
    max_charts = min(20, len(charts))
    return charts[:max_charts]


def suggest_charts(df: pd.DataFrame, dataset_profile: Dict[str, Any], kpis: List[Dict[str, Any]] = []) -> List[Dict[str, Any]]:
    """
    Intelligent chart suggestion system that considers column roles, semantic tags,
    and meaningful relationships instead of naive heuristics.
    
    Args:
        df: Input DataFrame
        dataset_profile: Dataset profile with column roles and semantic information
        kpis: List of KPIs to consider for chart suggestions (optional)
        
    Returns:
        List of chart specifications tailored to the dataset's meaningful characteristics
    """
    if df.empty:
        logger.warning("Empty dataframe provided to chart selector")
        return []

    n_rows, n_cols = df.shape
    if n_rows == 0 or n_cols == 0:
        logger.warning(f"Invalid dataframe shape: {n_rows}x{n_cols}")
        return []

    logger.info(f"Suggesting charts for dataset with {n_rows} rows and {n_cols} columns")

    try:
        suggested_charts = _suggest_appropriate_charts_for_columns(df, dataset_profile)

        logger.info(f"Suggested {len(suggested_charts)} meaningful charts")

        return suggested_charts

    except Exception as e:
        logger.error(f"Error in chart suggestion: {e}")
        import traceback
        traceback.print_exc()

        # Fallback: return minimal charts based on simple heuristics
        fallback_charts = []

        # At minimum, suggest one histogram for a meaningful numeric column
        for col in dataset_profile.get("columns", []):
            if col.get("role") == "numeric":
                col_name = col["name"]
                series = df[col_name]
                
                # Only suggest if it's not an identifier
                if not _is_likely_identifier(series, col_name):
                    numeric_data = pd.to_numeric(series, errors='coerce').dropna()
                    if len(numeric_data) > 5 and numeric_data.std() > 0.001:
                        fallback_charts.append({
                            "id": f"hist_{col_name}",
                            "title": f"Distribution of {col_name.replace('_', ' ').title()}",
                            "chart_type": "histogram",
                            "intent": "distribution",
                            "x_field": col_name,
                            "y_field": None,
                            "agg_func": "count",
                            "priority": 1
                        })
                        break  # Only add one fallback histogram

        logger.info(f"Fallback: generated {len(fallback_charts)} charts")
        return fallback_charts
```

## src/ml/correlation_engine.py

```
"""
Advanced correlation analysis engine with meaningful relationship detection.
Implements proper filtering to avoid spurious correlations between identifiers
and meaningful correlation analysis based on column roles and semantics.
"""

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, List, Any, Tuple, Optional
from scipy.stats import pearsonr, spearmanr
import math

logger = logging.getLogger(__name__)


def _is_likely_identifier(series: pd.Series, name: str = "") -> bool:
    """
    Determine if a series is likely an identifier based on multiple heuristics.
    """
    n_total = len(series)
    if n_total == 0:
        return False

    n_unique = series.nunique()
    unique_ratio = n_unique / n_total if n_total > 0 else 0.0

    detection_signals = {}

    # Signal 1: High cardinality (potential ID)
    if unique_ratio > 0.98:
        detection_signals["very_high_cardinality"] = min(0.95, unique_ratio)
    elif unique_ratio > 0.95:
        detection_signals["high_cardinality"] = min(0.85, unique_ratio * 0.9)
    elif unique_ratio > 0.90:
        detection_signals["moderate_cardinality"] = unique_ratio * 0.6

    # Signal 2: Sequential numeric pattern (common in internal IDs)
    if pd.api.types.is_numeric_dtype(series):
        numeric_vals = pd.to_numeric(series, errors='coerce').dropna()
        if len(numeric_vals) > 5:  # Need at least 5 values to check sequence
            sorted_vals = numeric_vals.sort_values()
            diffs = sorted_vals.diff().dropna()
            if len(diffs) > 0:
                # Check for mostly constant differences (sequential IDs)
                unique_diffs = diffs.unique()
                if len(unique_diffs) == 1 and abs(unique_diffs[0] - 1) < 0.01:  # Step of 1
                    detection_signals["sequential_step1"] = min(0.95, len(numeric_vals) / max(len(numeric_vals), 10))
                elif len(unique_diffs) <= 3 and diffs.std() < diffs.mean() * 0.1:  # Low variance in steps
                    detection_signals["sequential_low_variance"] = min(0.85, diffs.mean() * 0.7)

    # Signal 3: UUID pattern
    if series.dtype == 'object':
        sample = series.dropna().head(20).astype(str)
        uuid_matches = 0
        for val in sample:
            # Check for UUID v4 pattern (with case insensitivity)
            if re.match(r'^[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}$', val, re.IGNORECASE):
                uuid_matches += 1
        if len(sample) > 0:
            uuid_ratio = uuid_matches / len(sample)
            if uuid_ratio > 0.5:  # More than 50% are UUIDs
                detection_signals["uuid_pattern"] = uuid_ratio

    # Signal 4: Name-based detection (semantic heuristics)
    name_lower = name.lower()
    id_keywords = [
        "id", "uuid", "guid", "key", "code", "no", "number", "index",
        "account", "user", "customer", "product", "item", "order",
        "transaction", "invoice", "booking", "session", "token", "hash"
    ]

    matching_keywords = [kw for kw in id_keywords if kw in name_lower]
    if matching_keywords:
        # Calculate confidence based on how many keywords match
        keyword_confidence = min(0.8, len(matching_keywords) * 0.3)
        # Boost confidence if important keywords are found
        important_keywords = ["id", "uuid", "key", "code", "account", "user", "customer"]
        important_matches = sum(1 for kw in matching_keywords if kw in important_keywords)
        keyword_confidence += important_matches * 0.15
        detection_signals["name_pattern"] = min(1.0, keyword_confidence)

    # Calculate overall confidence based on signal strengths and weights
    if detection_signals:
        # Weight different signals appropriately
        weights = {
            "uuid_pattern": 1.0,              # Highest confidence for UUIDs
            "sequential_step1": 0.95,         # High confidence for clear sequential patterns
            "very_high_cardinality": 0.9,     # High confidence for extremely high uniqueness
            "sequential_low_variance": 0.85,  # High confidence for sequential patterns
            "high_cardinality": 0.8,          # Good confidence for high uniqueness
            "name_pattern": 0.75,             # Good confidence for name patterns
            "moderate_cardinality": 0.4       # Lower confidence for moderate uniqueness
        }

        max_confidence = 0
        best_signal = ""

        for signal, score in detection_signals.items():
            weight = weights.get(signal, 0.6)  # Default weight of 0.6
            weighted_score = score * weight
            if weighted_score > max_confidence:
                max_confidence = min(1.0, weighted_score)
                best_signal = signal

        # Consider it an identifier if confidence exceeds threshold
        is_identifier = max_confidence > 0.6

        return is_identifier

    return False


def _has_meaningful_variance(series: pd.Series, threshold: float = 0.001) -> bool:
    """
    Check if a numeric series has meaningful variance for correlation analysis.

    Args:
        series: The numeric series to evaluate
        threshold: Minimum standard deviation to be considered meaningful

    Returns:
        True if the series has meaningful variance
    """
    clean_series = pd.to_numeric(series, errors='coerce').dropna()
    if len(clean_series) < 3:  # Need at least 3 points for meaningful variance
        return False

    std_val = clean_series.std()
    mean_val = clean_series.mean()

    if pd.isna(std_val) or std_val < threshold:
        return False  # Very low variance

    # Additional check: if std is tiny compared to mean, it might be nearly constant
    if pd.notna(mean_val) and abs(mean_val) > threshold and std_val/abs(mean_val) < 0.01:
        return False  # Very low coefficient of variation

    return True


def _is_stable_correlation(series1: pd.Series, series2: pd.Series, stability_threshold: float = 0.1) -> bool:
    """
    Check if the correlation between two series is stable and meaningful.
    
    Args:
        series1, series2: The two series to evaluate
        stability_threshold: Minimum correlation value to be considered stable
        
    Returns:
        True if correlation is stable and above threshold
    """
    clean_s1 = pd.to_numeric(series1, errors='coerce').dropna()
    clean_s2 = pd.to_numeric(series2, errors='coerce').dropna()
    
    # Align series to have the same indices
    aligned_df = pd.concat([clean_s1, clean_s2], axis=1).dropna()
    
    if len(aligned_df) < 10:  # Need at least 10 aligned points for stable correlation
        return False
    
    s1_aligned = aligned_df.iloc[:, 0]
    s2_aligned = aligned_df.iloc[:, 1]
    
    try:
        # Calculate Pearson correlation
        pearson_corr, p_value = pearsonr(s1_aligned, s2_aligned) if len(s1_aligned) > 3 else (0.0, 1.0)
        
        # If correlation is weak, it might be spurious
        if pd.isna(pearson_corr) or abs(pearson_corr) < stability_threshold:
            return False
            
        # Only return True if both have meaningful variance
        return _has_meaningful_variance(s1_aligned) and _has_meaningful_variance(s2_aligned)
    except Exception:
        return False


def _identify_meaningful_correlations(df: pd.DataFrame, columns: List[Dict[str, Any]], 
                                    min_correlation: float = 0.1, min_variance: float = 0.001) -> List[Dict[str, Any]]:
    """
    Identify meaningful correlations between numeric columns, excluding identifiers.
    
    Args:
        df: Input DataFrame
        columns: List of column profiles including roles and names
        min_correlation: Minimum absolute correlation value to be considered meaningful
        min_variance: Minimum variance for columns to be considered meaningful
        
    Returns:
        List of meaningful correlations as dictionaries
    """
    # Filter to only numeric columns that are not likely identifiers
    numeric_cols = []
    for col_info in columns:
        if col_info.get("role") == "numeric":
            col_name = col_info["name"]
            series = df[col_name]
            
            # Check if this is likely an identifier
            if not _is_likely_identifier(series, col_name):
                # Additional validation: check for sufficient variance
                if _has_meaningful_variance(series, min_variance):
                    numeric_cols.append(col_name)
    
    if len(numeric_cols) < 2:
        logger.info(f"Not enough meaningful numeric columns for correlation analysis (found {len(numeric_cols)})")
        return []
    
    logger.info(f"Analyzing correlations for {len(numeric_cols)} meaningful numeric columns")
    
    correlations = []
    
    # Compute pair-wise correlations for meaningful numeric columns only
    for i, col1_name in enumerate(numeric_cols):
        for j, col2_name in enumerate(numeric_cols[i+1:], i+1):  # Avoid duplicate pairs
            series1 = df[col1_name]
            series2 = df[col2_name]
            
            # Clean the data before correlation
            clean_s1 = pd.to_numeric(series1, errors='coerce').dropna()
            clean_s2 = pd.to_numeric(series2, errors='coerce').dropna()
            
            # Align series to have the same indices
            aligned_df = pd.concat([clean_s1, clean_s2], axis=1).dropna()
            
            if len(aligned_df) < 3:  # Need at least 3 aligned points
                continue
                
            s1_aligned = aligned_df.iloc[:, 0]
            s2_aligned = aligned_df.iloc[:, 1]
            
            if len(s1_aligned) == 0 or len(s2_aligned) == 0 or len(s1_aligned) != len(s2_aligned):
                continue
            
            try:
                # Calculate Pearson correlation
                correlation, p_value = pearsonr(s1_aligned, s2_aligned)
                
                if pd.isna(correlation):
                    continue
                    
                abs_corr = abs(correlation)
                
                # Only include if correlation is meaningful
                if abs_corr >= min_correlation:
                    # Determine correlation strength
                    strength = "weak"
                    if abs_corr >= 0.7:
                        strength = "strong"
                    elif abs_corr >= 0.5:
                        strength = "moderate"
                    elif abs_corr >= 0.3:
                        strength = "moderate_weak"

                    # Determine correlation type
                    correlation_type = "positive" if correlation > 0 else "negative"

                    # Get basic statistics for the relationship
                    slope, intercept = np.polyfit(s1_aligned, s2_aligned, 1) if len(s1_aligned) > 1 else (0, 0)

                    correlations.append({
                        "variable1": col1_name,
                        "variable2": col2_name,
                        "correlation": float(correlation),
                        "abs_correlation": float(abs_corr),
                        "strength": strength,
                        "type": correlation_type,
                        "p_value": float(p_value),
                        "sample_size": int(len(aligned_df)),
                        "slope": float(slope),
                        "intercept": float(intercept)
                    })
                    
            except Exception as e:
                logger.warning(f"Error calculating correlation between {col1_name} and {col2_name}: {e}")
                continue
    
    logger.info(f"Identified {len(correlations)} meaningful correlations")
    return correlations


def _identify_cross_type_relationships(df: pd.DataFrame, columns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Identify meaningful relationships between different column types (numerical vs categorical).
    
    Args:
        df: Input DataFrame
        columns: List of column profiles including roles and names
        
    Returns:
        List of meaningful cross-type relationships as dictionaries
    """
    numeric_cols = []
    categorical_cols = []
    
    # Separate numeric and categorical columns, filtering out identifiers
    for col_info in columns:
        col_name = col_info["name"]
        series = df[col_name]
        
        # Skip if likely an identifier
        if _is_likely_identifier(series, col_name):
            continue
            
        if col_info.get("role") == "numeric":
            if _has_meaningful_variance(series):
                numeric_cols.append(col_name)
        elif col_info.get("role") in ["categorical", "text"]:
            if col_info.get("unique_count", 0) <= 50:  # Limit to low-cardinality categorical
                categorical_cols.append(col_name)
    
    relationships = []
    
    # Analyze numeric vs categorical relationships using ANOVA-like approach
    for num_col in numeric_cols:
        for cat_col in categorical_cols:
            series_num = pd.to_numeric(df[num_col], errors='coerce').dropna()
            series_cat = df[cat_col].dropna()
            
            # Align series
            aligned_df = pd.concat([series_num, series_cat], axis=1).dropna()
            
            if len(aligned_df) < 10:  # Need sufficient data
                continue
                
            aligned_num = aligned_df.iloc[:, 0]
            aligned_cat = aligned_df.iloc[:, 1]
            
            # Check if categorical has sufficient different values to be meaningful
            n_unique_cats = aligned_cat.nunique()
            if n_unique_cats < 2 or n_unique_cats > 20:  # Not meaningful if too few or too many categories
                continue
                
            try:
                # Group by category and calculate statistics
                grouped = aligned_num.groupby(aligned_cat)
                group_means = grouped.mean()
                group_sizes = grouped.count()
                
                # Calculate overall statistics
                overall_mean = aligned_num.mean()
                
                # Calculate between-group and within-group variance (ANOVA-like)
                between_sum_sq = sum(group_sizes * ((group_means - overall_mean) ** 2))
                within_sum_sq = sum([
                    ((aligned_num[aligned_cat == cat] - group_means[cat]) ** 2).sum() 
                    for cat in group_means.index
                ])
                
                # Calculate effect size (eta-squared - variance explained by group membership)
                total_sum_sq = between_sum_sq + within_sum_sq
                eta_squared = between_sum_sq / total_sum_sq if total_sum_sq > 0 else 0
                
                # Calculate significance using basic approximation (for larger samples)
                f_stat = (between_sum_sq / (n_unique_cats - 1)) / (within_sum_sq / (len(aligned_num) - n_unique_cats)) if within_sum_sq > 0 and (len(aligned_num) - n_unique_cats) > 0 else 0
                # This is a simplified approximation; for exact p-values, we'd need scipy.stats.f
                
                # Only include if there's meaningful variance explained
                if eta_squared >= 0.02:  # At least 2% variance explained
                    relationships.append({
                        "numeric_variable": num_col,
                        "categorical_variable": cat_col,
                        "effect_size": float(eta_squared),
                        "group_means": {str(cat): float(mean_val) for cat, mean_val in group_means.items()},
                        "group_sizes": {str(cat): int(size) for cat, size in group_sizes.items()},
                        "f_statistic": float(f_stat) if not pd.isna(f_stat) else 0.0,
                        "sample_size": int(len(aligned_num)),
                        "n_groups": int(n_unique_cats)
                    })
            except Exception as e:
                logger.warning(f"Error analyzing relationship between {num_col} and {cat_col}: {e}")
                continue
    
    logger.info(f"Identified {len(relationships)} meaningful cross-type relationships")
    return relationships


def _detect_spurious_correlations(df: pd.DataFrame, columns: List[Dict[str, Any]], 
                                correlations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detect spurious correlations in the already computed correlations.
    
    Args:
        df: Input DataFrame
        columns: List of column profiles
        correlations: List of computed correlation dictionaries
        
    Returns:
        List of potentially spurious correlations
    """
    spurious_correlations = []
    
    # Check for correlations involving likely identifiers (shouldn't happen if filtered properly, but double-check)
    for corr in correlations:
        var1 = corr["variable1"]
        var2 = corr["variable2"]
        
        series1 = df[var1]
        series2 = df[var2]
        
        is_id1 = _is_likely_identifier(series1, var1)
        is_id2 = _is_likely_identifier(series2, var2)
        
        if is_id1 or is_id2:
            spurious_correlations.append({
                "variables": [var1, var2],
                "correlation": corr["correlation"],
                "reason": "at_least_one_is_identifier",
                "confidence": 0.9
            })
    
    # Check for correlations between nearly constant variables
    for corr in correlations:
        var1 = corr["variable1"]
        var2 = corr["variable2"]
        
        series1 = df[var1]
        series2 = df[var2]
        
        # Check if either variable has very low variance
        has_low_var1 = not _has_meaningful_variance(series1)
        has_low_var2 = not _has_meaningful_variance(series2)
        
        if has_low_var1 or has_low_var2:
            reason = f"{'first' if has_low_var1 else 'second'} variable has low variance"
            spurious_correlations.append({
                "variables": [var1, var2],
                "correlation": corr["correlation"],
                "reason": reason,
                "confidence": 0.8
            })
    
    # Check for correlations with very small sample sizes
    for corr in correlations:
        if corr.get("sample_size", 0) < 10:
            spurious_correlations.append({
                "variables": [corr["variable1"], corr["variable2"]],
                "correlation": corr["correlation"],
                "reason": "insufficient sample size",
                "confidence": 0.7
            })
    
    logger.info(f"Detected {len(spurious_correlations)} potentially spurious correlations")
    return spurious_correlations


def analyze_correlations(df: pd.DataFrame, dataset_profile: Dict[str, Any], 
                       min_correlation: float = 0.1, 
                       min_variance: float = 0.001) -> Dict[str, Any]:
    """
    Comprehensive correlation analysis with proper filtering to avoid spurious correlations.
    
    Args:
        df: Input DataFrame
        dataset_profile: Dataset profile containing column information
        min_correlation: Minimum absolute correlation value to be considered meaningful
        min_variance: Minimum variance for columns to be considered meaningful
        
    Returns:
        Dictionary containing correlation analysis results including:
        - meaningful_correlations: List of meaningful correlations
        - cross_type_relationships: Relationships between different column types
        - spurious_correlations: Identified spurious correlations
        - summary_stats: Summary statistics about correlation analysis
    """
    if df.empty:
        logger.warning("Empty dataframe provided to correlation analysis")
        return {
            "meaningful_correlations": [],
            "cross_type_relationships": [],
            "spurious_correlations": [],
            "summary_stats": {"total_analyzed_pairs": 0, "meaningful_pairs": 0, "spurious_pairs": 0}
        }
    
    columns = dataset_profile.get("columns", [])
    if not columns:
        logger.warning("No columns found in dataset profile for correlation analysis")
        return {
            "meaningful_correlations": [],
            "cross_type_relationships": [],
            "spurious_correlations": [],
            "summary_stats": {"total_analyzed_pairs": 0, "meaningful_pairs": 0, "spurious_pairs": 0}
        }
    
    logger.info(f"Starting correlation analysis for {len(columns)} columns")
    
    # Perform meaningful correlation analysis
    meaningful_correlations = _identify_meaningful_correlations(
        df, columns, min_correlation, min_variance
    )
    
    # Identify cross-type relationships
    cross_type_relationships = _identify_cross_type_relationships(df, columns)
    
    # Detect spurious correlations
    spurious_correlations = _detect_spurious_correlations(df, columns, meaningful_correlations)
    
    # Create summary statistics
    n_numeric_cols = len([col for col in columns if col.get("role") == "numeric"])
    n_meaningful_numeric = len([col for col in columns 
                               if col.get("role") == "numeric" and 
                               not _is_likely_identifier(df[col["name"]], col["name"]) and
                               _has_meaningful_variance(df[col["name"]])])
    
    # Calculate total possible pairs for numeric columns (n*(n-1)/2)
    total_possible_pairs = (n_meaningful_numeric * (n_meaningful_numeric - 1)) // 2
    
    summary_stats = {
        "total_possible_pairs": total_possible_pairs,
        "total_analyzed_pairs": len(meaningful_correlations),
        "meaningful_pairs": len(meaningful_correlations),
        "spurious_pairs_detected": len(spurious_correlations),
        "analyzed_numeric_columns": n_meaningful_numeric,
        "original_numeric_columns": n_numeric_cols,
        "min_correlation_threshold": min_correlation,
        "min_variance_threshold": min_variance
    }
    
    logger.info(f"Correlation analysis completed: {len(meaningful_correlations)} meaningful relationships found out of {total_possible_pairs} possible pairs")
    
    return {
        "meaningful_correlations": meaningful_correlations,
        "cross_type_relationships": cross_type_relationships,
        "spurious_correlations": spurious_correlations,
        "summary_stats": summary_stats
    }


def generate_correlation_insights(correlation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate insights from the correlation analysis results.
    
    Args:
        correlation_results: Results from the analyze_correlations function
        
    Returns:
        List of insight dictionaries
    """
    insights = []
    
    # Insight from meaningful correlations
    meaningful_correlations = correlation_results.get("meaningful_correlations", [])
    if meaningful_correlations:
        # Find strongest correlations
        sorted_corrs = sorted(meaningful_correlations, key=lambda x: abs(x["correlation"]), reverse=True)
        strong_corrs = [c for c in sorted_corrs if abs(c["correlation"]) >= 0.7]
        
        if strong_corrs:
            insights.append({
                "type": "strong_correlation",
                "title": f"Strong Correlations Detected ({len(strong_corrs)} found)",
                "description": f"Identified {len(strong_corrs)} strongly correlated variable pairs (>0.7 correlation)",
                "details": [{"variables": [c["variable1"], c["variable2"]], "correlation": c["correlation"]} for c in strong_corrs[:5]],  # Limit to top 5
                "confidence": 0.9
            })
        
        # Find moderate correlations
        moderate_corrs = [c for c in meaningful_correlations if 0.5 <= abs(c["correlation"]) < 0.7]
        if moderate_corrs:
            insights.append({
                "type": "moderate_correlation",
                "title": f"Moderate Correlations Detected ({len(moderate_corrs)} found)",
                "description": f"Identified {len(moderate_corrs)} moderately correlated variable pairs (0.5-0.7 correlation)",
                "details": [{"variables": [c["variable1"], c["variable2"]], "correlation": c["correlation"]} for c in moderate_corrs[:5]],
                "confidence": 0.8
            })
    
    # Insight from cross-type relationships
    cross_relationships = correlation_results.get("cross_type_relationships", [])
    if cross_relationships:
        high_impact_relations = [r for r in cross_relationships if r["effect_size"] >= 0.15]  # High effect size
        
        if high_impact_relations:
            insights.append({
                "type": "cross_type_relationship",
                "title": f"Strong Cross-Type Relationships ({len(high_impact_relations)} found)",
                "description": f"Identified {len(high_impact_relations)} categorical variables that strongly influence numeric variables",
                "details": [{"numeric": r["numeric_variable"], "categorical": r["categorical_variable"], "effect_size": r["effect_size"]} for r in high_impact_relations[:5]],
                "confidence": 0.85
            })
    
    # Insight from spurious correlations detection
    spurious_correlations = correlation_results.get("spurious_correlations", [])
    if spurious_correlations:
        insights.append({
            "type": "data_quality_warning",
            "title": f"Spurious Correlations Flagged ({len(spurious_correlations)} found)",
            "description": f"Flagged {len(spurious_correlations)} potentially spurious correlations that may not be meaningful",
            "details": [{"variables": c["variables"], "correlation": c["correlation"], "reason": c["reason"]} for c in spurious_correlations[:5]],
            "confidence": 0.7
        })
    
    # Summary insight
    summary_stats = correlation_results.get("summary_stats", {})
    meaningful_pairs = summary_stats.get("meaningful_pairs", 0)
    analyzed_cols = summary_stats.get("analyzed_numeric_columns", 0)
    
    if analyzed_cols > 1:
        insights.append({
            "type": "correlation_summary",
            "title": f"Correlation Analysis Summary",
            "description": f"Analyzed {analyzed_cols} meaningful numeric columns and found {meaningful_pairs} significant relationships",
            "details": summary_stats,
            "confidence": 0.8
        })
    
    logger.info(f"Generated {len(insights)} correlation insights")
    return insights
```

## src/eda/insights_generator.py

```
"""
Advanced EDA and Insights Generator for the ML Dashboard
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from scipy.stats import pearsonr
from collections import Counter
import re
from src.ml.correlation_engine import analyze_correlations, generate_correlation_insights

logger = logging.getLogger(__name__)


def _is_likely_identifier(series: pd.Series, name: str = "") -> bool:
    """
    Determine if a series is likely an identifier based on multiple heuristics.
    This is a simplified version of the identifier detection logic from chart_selector.py.

    Args:
        series: The pandas Series to analyze
        name: The column name

    Returns:
        True if the series is likely an identifier
    """
    n_total = len(series)
    if n_total == 0:
        return False

    n_unique = series.nunique()
    unique_ratio = n_unique / n_total if n_total > 0 else 0.0

    # Check for high cardinality (potential ID)
    if unique_ratio > 0.98:
        # Check if it's numeric (potential sequential ID)
        if pd.api.types.is_numeric_dtype(series):
            numeric_vals = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_vals) > 5:  # Need at least 5 values to check sequence
                sorted_vals = numeric_vals.sort_values()
                diffs = sorted_vals.diff().dropna()
                if len(diffs) > 0:
                    # Check for mostly constant differences (sequential IDs)
                    unique_diffs = diffs.unique()
                    if len(unique_diffs) == 1 and abs(unique_diffs[0] - 1) < 0.01:  # Step of 1
                        return True
        # Check for UUID patterns in string values
        if series.dtype == 'object':
            sample = series.dropna().head(20).astype(str)
            uuid_matches = 0
            for val in sample:
                # Check for UUID v4 pattern (with case insensitivity)
                if re.match(r'^[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}$', val, re.IGNORECASE):
                    uuid_matches += 1
            if uuid_matches / len(sample) > 0.5:  # More than 50% are UUIDs
                return True

    # Check for ID-like names
    name_lower = name.lower()
    id_keywords = [
        "id", "uuid", "guid", "key", "code", "no", "number", "index",
        "account", "user", "customer", "product", "item", "order",
        "transaction", "invoice", "booking", "session", "token", "hash"
    ]

    if any(keyword in name_lower for keyword in id_keywords):
        return True

    return False

def detect_pattern_relationships(df: pd.DataFrame, dataset_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the dataset to detect patterns, relationships, and correlations
    """
    n_rows = dataset_profile.get("n_rows", len(df))
    if n_rows == 0:
        logger.warning("Empty dataframe provided to pattern detection")
        return {
            "correlations": [],
            "trends": [],
            "patterns": [],
            "outliers": [],
            "anomalies": []
        }

    columns = dataset_profile.get("columns", []) if dataset_profile else []
    numeric_cols = [col.get("name") for col in columns if col.get("role") == "numeric" and col.get("name") in df.columns]
    if not numeric_cols:
        # Fall back to dtype-based detection if profile is missing or empty
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    results = {
        "correlations": [],
        "trends": [],
        "patterns": [],
        "outliers": [],
        "anomalies": [],
        "distribution_insights": []
    }

    # 1. Find meaningful correlations using the advanced correlation engine
    correlation_analysis = analyze_correlations(df, dataset_profile)

    # Update results with meaningful correlations from the new engine
    results["correlations"] = correlation_analysis.get("meaningful_correlations", [])
    results["cross_type_relationships"] = correlation_analysis.get("cross_type_relationships", [])
    results["spurious_correlations_detected"] = correlation_analysis.get("spurious_correlations", [])
    results["correlation_summary"] = correlation_analysis.get("summary_stats", {})

    # If the correlation engine returns empty correlations, fall back to basic analysis
    if not results["correlations"] and len(numeric_cols) >= 2:
        # Basic correlation analysis as fallback
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]

                # Check if either column is likely an identifier
                series1 = df[col1]
                series2 = df[col2]

                # Skip if either column is likely an identifier
                if _is_likely_identifier(series1, col1) or _is_likely_identifier(series2, col2):
                    continue

                # Convert to numeric and drop NaN
                s1_numeric = pd.to_numeric(series1, errors='coerce')
                s2_numeric = pd.to_numeric(series2, errors='coerce')

                # Remove NaN values
                mask = ~(s1_numeric.isna() | s2_numeric.isna())
                s1_clean = s1_numeric[mask]
                s2_clean = s2_numeric[mask]

                if len(s1_clean) > 2:  # Need at least 3 points for correlation
                    try:
                        corr, p_value = pearsonr(s1_clean, s2_clean)

                        if not np.isnan(corr):
                            # Only include if correlation is meaningful (>0.1) and both have variance
                            std1 = s1_clean.std()
                            std2 = s2_clean.std()

                            if abs(corr) > 0.1 and std1 > 0.001 and std2 > 0.001:
                                results["correlations"].append({
                                    "variable1": col1,
                                    "variable2": col2,
                                    "correlation": corr,
                                    "p_value": p_value,
                                    "strength": "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak",
                                    "type": "positive" if corr > 0 else "negative",
                                    "sample_size": len(s1_clean)
                                })
                    except Exception as e:
                        logger.warning(f"Error calculating correlation between {col1} and {col2}: {e}")
    
    # 2. Detect potential trends in time-based data
    datetime_cols = [col['name'] for col in dataset_profile['columns'] if col['role'] == 'datetime']
    
    for dt_col in datetime_cols:
        for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            try:
                dt_series = pd.to_datetime(df[dt_col], errors='coerce')
                num_series = pd.to_numeric(df[num_col], errors='coerce')
                
                # Create a valid data frame
                temp_df = pd.DataFrame({dt_col: dt_series, num_col: num_series}).dropna()
                
                if len(temp_df) > 2:
                    # Use the index as a proxy for time and calculate correlation
                    temp_df = temp_df.sort_values(dt_col)
                    temp_df['time_index'] = range(len(temp_df))
                    
                    trend_corr, p_value = pearsonr(temp_df['time_index'], temp_df[num_col])
                    
                    if not np.isnan(trend_corr):
                        trend_type = "increasing" if trend_corr > 0.1 else "decreasing" if trend_corr < -0.1 else "stable"
                        results["trends"].append({
                            "datetime_column": dt_col,
                            "numeric_column": num_col,
                            "trend_correlation": trend_corr,
                            "trend_type": trend_type,
                            "p_value": p_value
                        })
            except Exception as e:
                logger.warning(f"Error detecting trend for {dt_col} and {num_col}: {e}")
    
    # 3. Identify patterns in categorical data
    categorical_cols = [col['name'] for col in dataset_profile['columns'] if col['role'] in ['categorical', 'boolean']]
    
    for col in categorical_cols:
        try:
            # Most common categories
            value_counts = df[col].value_counts()
            total_count = len(df[col])
            
            # Identify dominant categories (>20% of the data)
            dominant_categories = []
            for cat, count in value_counts.items():
                ratio = count / total_count
                if ratio > 0.2:  # More than 20% of the data
                    dominant_categories.append({
                        "category": cat,
                        "count": count,
                        "percentage": ratio * 100
                    })
            
            if dominant_categories:
                results["patterns"].append({
                    "column": col,
                    "pattern_type": "dominant_categories",
                    "categories": dominant_categories
                })
                
            # Identify low-variety categories (high concentration)
            unique_count = len(value_counts)
            if unique_count > 1 and total_count > 0:
                entropy = -sum((count/total_count) * np.log2(count/total_count) for count in value_counts if count > 0)
                max_entropy = np.log2(unique_count)
                
                if max_entropy > 0:
                    normalized_entropy = entropy / max_entropy
                    if normalized_entropy < 0.5:  # Low entropy = high concentration
                        results["patterns"].append({
                            "column": col,
                            "pattern_type": "low_entropy",
                            "entropy": normalized_entropy,
                            "unique_values": unique_count
                        })
        except Exception as e:
            logger.warning(f"Error detecting patterns for column {col}: {e}")
    
    # 4. Detect outliers in numeric data
    for col in numeric_cols:
        try:
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            
            if len(series) > 4:  # Need at least 5 values to detect outliers meaningfully
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                
                if len(outliers) > 0:
                    results["outliers"].append({
                        "column": col,
                        "outlier_count": len(outliers),
                        "outlier_percentage": (len(outliers) / len(series)) * 100,
                        "outlier_values": outliers.head(10).tolist()  # Limit to first 10 outliers
                    })
        except Exception as e:
            logger.warning(f"Error detecting outliers for column {col}: {e}")
    
    # 5. Detect anomalies based on data distribution
    for col in dataset_profile['columns']:
        try:
            if col['role'] in ['categorical', 'text'] and col['unique_count'] > 1:
                # Detect potential data quality issues
                value_counts = df[col['name']].value_counts()
                
                # Check for very low frequency values that might be typos
                rare_values = value_counts[value_counts < 3]  # Values that appear less than 3 times
                if len(rare_values) > 0:
                    results["anomalies"].append({
                        "column": col['name'],
                        "anomaly_type": "rare_values",
                        "rare_values_count": len(rare_values),
                        "example_values": rare_values.head(5).index.tolist()
                    })
                    
                # Check for highly imbalanced distributions
                if len(value_counts) > 2:
                    total = sum(value_counts)
                    largest_category = value_counts.iloc[0]
                    if largest_category / total > 0.95:  # 95% of values are the same
                        results["anomalies"].append({
                            "column": col['name'],
                            "anomaly_type": "highly_imbalanced",
                            "largest_category_ratio": largest_category / total,
                            "largest_category": value_counts.index[0]
                        })
        except Exception as e:
            logger.warning(f"Error detecting anomalies for column {col['name']}: {e}")
    
    # 6. Generate distribution insights
    for col in numeric_cols:
        try:
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            
            if len(series) > 2:
                # Calculate skewness and kurtosis
                skewness = series.skew()
                kurtosis = series.kurtosis()
                
                results["distribution_insights"].append({
                    "column": col,
                    "skewness": skewness,
                    "kurtosis": kurtosis,
                    "distribution_type": "right_skewed" if skewness > 1 else "left_skewed" if skewness < -1 else "symmetric",
                    "tail_type": "heavy_tailed" if kurtosis > 0 else "light_tailed"
                })
        except Exception as e:
            logger.warning(f"Error calculating distribution insights for column {col}: {e}")
    
    return results


def extract_use_cases(df: pd.DataFrame, dataset_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract potential use cases from the dataset based on content analysis
    """
    use_cases = []

    n_rows = dataset_profile.get("n_rows", len(df))
    if n_rows == 0:
        return []

    # Extract semantic clues from column names and types
    column_names = [col['name'] for col in dataset_profile['columns']]

    # Identify potential use cases based on column patterns
    semantic_categories = {
        'sales': {
            'keywords': ['price', 'cost', 'revenue', 'sales', 'amount', 'profit', 'fee', 'charge', 'payment', 'discount', 'tax', 'margin'],
            'key_inputs': ['product_id', 'customer_id', 'order_date', 'quantity'],
            'indicators': ['total_revenue', 'profit_margin', 'sales_volume']
        },
        'demographics': {
            'keywords': ['age', 'gender', 'race', 'ethnicity', 'income', 'education', 'occupation', 'family', 'children', 'marital', 'birth'],
            'key_inputs': ['id', 'date_of_birth', 'location', 'survey_date'],
            'indicators': ['avg_income', 'education_level_distribution', 'age_median']
        },
        'location': {
            'keywords': ['city', 'state', 'country', 'address', 'location', 'region', 'zip', 'postal', 'latitude', 'longitude', 'area'],
            'key_inputs': ['coordinates', 'administrative_divisions', 'time_zone'],
            'indicators': ['population_density', 'geographic_distribution', 'distance_metrics']
        },
        'time': {
            'keywords': ['date', 'time', 'day', 'week', 'month', 'year', 'season', 'period', 'duration', 'timestamp', 'hour', 'minute'],
            'key_inputs': ['id', 'event_type', 'start_time', 'end_time'],
            'indicators': ['trend_over_time', 'seasonal_patterns', 'frequency']
        },
        'rating': {
            'keywords': ['rating', 'score', 'grade', 'review', 'feedback', 'satisfaction', 'rating_count', 'stars', 'vote'],
            'key_inputs': ['reviewer_id', 'item_id', 'review_text', 'review_date'],
            'indicators': ['avg_rating', 'rating_distribution', 'review_sentiment']
        },
        'quantity': {
            'keywords': ['count', 'quantity', 'number', 'volume', 'size', 'frequency', 'frequency', 'instances', 'cases', 'instances'],
            'key_inputs': ['item_id', 'category', 'measurement_unit'],
            'indicators': ['total_count', 'avg_quantity', 'distribution']
        },
        'health': {
            'keywords': ['patient', 'diagnosis', 'treatment', 'symptom', 'medication', 'disease', 'condition', 'blood_pressure', 'pulse', 'temperature'],
            'key_inputs': ['patient_id', 'doctor_id', 'diagnosis_date', 'symptom onset'],
            'indicators': ['recovery_rate', 'diagnosis_distribution', 'treatment_success']
        },
        'education': {
            'keywords': ['student', 'grade', 'score', 'subject', 'school', 'enrollment', 'test', 'exam', 'course', 'gpa', 'attendance'],
            'key_inputs': ['student_id', 'course_id', 'exam_date', 'instructor'],
            'indicators': ['avg_score', 'pass_rate', 'attendance_rate']
        },
        'finance': {
            'keywords': ['account', 'balance', 'transaction', 'credit', 'loan', 'interest', 'investment', 'portfolio', 'return', 'equity'],
            'key_inputs': ['account_id', 'transaction_date', 'counterparty', 'reference'],
            'indicators': ['cash_flow', 'return_on_investment', 'risk_metrics']
        },
        'technology': {
            'keywords': ['device', 'os', 'platform', 'software', 'version', 'model', 'type', 'cpu', 'memory', 'resolution', 'bandwidth'],
            'key_inputs': ['device_id', 'manufacturer', 'release_date', 'specifications'],
            'indicators': ['usage_patterns', 'performance_metrics', 'adoption_rate']
        },
        'transportation': {
            'keywords': ['vehicle', 'model', 'year', 'mileage', 'route', 'trip', 'distance', 'speed', 'fuel', 'departure', 'arrival'],
            'key_inputs': ['vehicle_id', 'driver_id', 'route_id', 'timestamp'],
            'indicators': ['avg_speed', 'fuel_efficiency', 'on_time_rate']
        },
        'marketing': {
            'keywords': ['campaign', 'click', 'impression', 'conversion', 'revenue', 'cost', 'cpc', 'cpa', 'roi', 'engagement'],
            'key_inputs': ['campaign_id', 'ad_group', 'keyword', 'audience'],
            'indicators': ['conversion_rate', 'roi', 'cost_per_conversion']
        },
        'retail': {
            'keywords': ['product', 'inventory', 'stock', 'sku', 'brand', 'category', 'supplier', 'shelf', 'vendor', 'order'],
            'key_inputs': ['product_id', 'store_id', 'supplier_id', 'reorder_date'],
            'indicators': ['inventory_turnover', 'stockout_rate', 'profit_margin']
        },
        'social_media': {
            'keywords': ['user', 'post', 'like', 'comment', 'share', 'follower', 'engagement', 'reach', 'impression', 'hashtag'],
            'key_inputs': ['user_id', 'post_id', 'timestamp', 'content_type'],
            'indicators': ['engagement_rate', 'follower_growth', 'content_popularity']
        }
    }

    # Find which semantic categories are present in the dataset
    present_categories = []
    for cat, info in semantic_categories.items():
        category_match = False

        # Check keywords in column names
        for col_name in column_names:
            if any(keyword in col_name.lower() for keyword in info['keywords']):
                category_match = True
                break

        # Check if this category is present
        if category_match:
            present_categories.append((cat, info))

    # Generate use case suggestions based on detected semantic categories
    for cat, info in present_categories:
        # Identify key columns relevant to this category
        key_columns = [name for name in column_names if any(kw in name.lower() for kw in info['keywords'])]

        # Add key inputs if they exist in the dataset
        for input_col in info['key_inputs']:
            if input_col in column_names and input_col not in key_columns:
                key_columns.append(input_col)

        # Use case specific to this category
        if cat == 'sales':
            use_cases.append({
                "use_case": "Sales Performance Analysis",
                "description": "Analyze product performance, revenue trends, and sales metrics",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Revenue by time", "Top selling products", "Sales by category", "Profit margins"]
            })
        elif cat == 'demographics':
            use_cases.append({
                "use_case": "Demographic Analysis",
                "description": "Understand customer or subject demographics and patterns",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Age distribution", "Gender breakdown", "Income vs other factors", "Education level"]
            })
        elif cat == 'location':
            use_cases.append({
                "use_case": "Geographic Analysis",
                "description": "Analyze geographic patterns and location-based trends",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Sales by region", "Geographic distribution", "Location vs other metrics", "Heatmaps"]
            })
        elif cat == 'time':
            use_cases.append({
                "use_case": "Time Series Analysis",
                "description": "Analyze trends, seasonality, and time-based patterns",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Trends over time", "Seasonal patterns", "Period comparisons", "Moving averages"]
            })
        elif cat == 'rating':
            use_cases.append({
                "use_case": "Performance Rating Analysis",
                "description": "Analyze ratings, scores, and performance metrics",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Rating distributions", "Average scores by category", "Rating trends", "Review sentiment"]
            })
        elif cat == 'health':
            use_cases.append({
                "use_case": "Healthcare Analysis",
                "description": "Analyze patient data, treatments, and health outcomes",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Condition distribution", "Treatment effectiveness", "Patient demographics", "Health metrics over time"]
            })
        elif cat == 'education':
            use_cases.append({
                "use_case": "Educational Performance Analysis",
                "description": "Analyze student performance, course outcomes, and educational metrics",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Average scores by subject", "Pass rates", "Attendance patterns", "Performance trends"]
            })
        elif cat == 'finance':
            use_cases.append({
                "use_case": "Financial Analysis",
                "description": "Analyze financial performance, risk, and investment outcomes",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Revenue trends", "Risk metrics", "Investment returns", "Cash flow analysis"]
            })
        elif cat == 'technology':
            use_cases.append({
                "use_case": "Technology Usage Analysis",
                "description": "Analyze device usage, software adoption, and technology performance",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Device usage patterns", "Software adoption", "Performance metrics", "Technology trends"]
            })
        elif cat == 'transportation':
            use_cases.append({
                "use_case": "Transportation Analysis",
                "description": "Analyze route efficiency, vehicle performance, and transportation metrics",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Average speed patterns", "Fuel efficiency", "On-time performance", "Route optimization"]
            })
        elif cat == 'marketing':
            use_cases.append({
                "use_case": "Marketing Campaign Analysis",
                "description": "Analyze campaign performance, conversion rates, and marketing ROI",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Conversion rates", "ROI by channel", "Cost per acquisition", "Engagement metrics"]
            })
        elif cat == 'retail':
            use_cases.append({
                "use_case": "Retail Operations Analysis",
                "description": "Analyze inventory, sales, and retail operational metrics",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Inventory turnover", "Product performance", "Stock levels", "Sales patterns"]
            })
        elif cat == 'social_media':
            use_cases.append({
                "use_case": "Social Media Engagement Analysis",
                "description": "Analyze user engagement, content performance, and social metrics",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Engagement rates", "Follower growth", "Content popularity", "User activity patterns"]
            })

    # Generate a general use case if no specific ones were detected
    if not use_cases:
        use_cases.append({
            "use_case": "General Data Exploration",
            "description": "Explore and understand the structure and content of the dataset",
            "key_inputs": column_names[:5],  # First 5 columns
            "key_indicators": ["data_completeness", "uniqueness", "data_types"],
            "suggested_visualizations": ["Column distributions", "Missing value patterns", "Data types overview"]
        })

    # Add cross-domain use cases if multiple categories are detected
    if len(present_categories) > 1:
        # Identify common columns that might connect different categories
        common_columns = []
        for col in dataset_profile['columns']:
            col_name = col['name']
            # Look for ID columns that might connect different domains
            if 'id' in col_name.lower() or 'key' in col_name.lower():
                common_columns.append(col_name)

        if common_columns:
            use_cases.append({
                "use_case": "Cross-Domain Analysis",
                "description": "Analyze relationships between different data domains using common identifiers",
                "key_inputs": common_columns,
                "key_indicators": ["connection_strength", "data_integration_points"],
                "suggested_visualizations": ["Domain correlations", "Common identifier distributions", "Cross-domain patterns"]
            })

    return use_cases


def identify_key_indicators(df: pd.DataFrame, dataset_profile: Dict[str, Any], correlations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Identify key indicators based on data patterns, correlations, and statistical significance
    """
    key_indicators = []

    n_rows = dataset_profile.get("n_rows", len(df))
    if n_rows == 0:
        return []

    # 1. High-impact numeric columns (high variance or frequently correlated)
    numeric_cols = [col['name'] for col in dataset_profile['columns'] if col['role'] == 'numeric']

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors='coerce').dropna()

        if len(series) > 0:
            # Calculate the coefficient of variation (std/mean) - high values indicate high variability
            mean_val = series.mean()
            std_val = series.std()

            if mean_val != 0:  # Avoid division by zero
                cv = abs(std_val / mean_val) if std_val is not None else 0
            else:
                cv = std_val if std_val is not None else 0

            # Count how many times this column appears in strong correlations
            correlation_count = sum(1 for corr in correlations
                                  if corr['strength'] == 'strong' and col in [corr['variable1'], corr['variable2']])

            # Calculate outlier impact
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = len(series[(series < lower_bound) | (series > upper_bound)])
            outlier_ratio = outliers / len(series) if len(series) > 0 else 0

            # Calculate skewness
            skewness = series.skew()

            key_indicators.append({
                "indicator": col,
                "indicator_type": "numeric",
                "significance_score": cv + correlation_count * 0.5 + outlier_ratio * 2,  # Combine variability, correlation, and outlier impact
                "metric_type": "continuous",
                "description": f"Numeric column with {correlation_count} strong correlation(s), CV of {cv:.2f}, and {outlier_ratio:.2%} outliers",
                "statistical_properties": {
                    "mean": float(mean_val),
                    "std": float(std_val) if std_val is not None else 0,
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "coefficient_of_variation": cv,
                    "skewness": skewness,
                    "outlier_ratio": outlier_ratio
                }
            })

    # 2. High-impact categorical columns (high cardinality or low entropy)
    categorical_cols = [col['name'] for col in dataset_profile['columns'] if col['role'] in ['categorical', 'boolean']]

    for col in categorical_cols:
        series = df[col].dropna()
        unique_count = series.nunique()
        total_count = len(series)

        if total_count > 0:
            # Calculate entropy to understand distribution diversity
            value_counts = series.value_counts()
            probs = value_counts / total_count
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            max_entropy = np.log2(unique_count) if unique_count > 0 else 0

            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

            # Calculate imbalance ratio (how much the most common category dominates)
            most_common_ratio = value_counts.iloc[0] / total_count if len(value_counts) > 0 else 0

            key_indicators.append({
                "indicator": col,
                "indicator_type": "categorical",
                "significance_score": (1 - normalized_entropy) * unique_count + (most_common_ratio * 10),  # Higher score for low entropy + high unique count + high imbalance
                "metric_type": "categorical",
                "description": f"Categorical column with {unique_count} unique values, normalized entropy of {normalized_entropy:.2f}, and max category ratio of {most_common_ratio:.2f}",
                "statistical_properties": {
                    "unique_count": unique_count,
                    "total_count": total_count,
                    "normalized_entropy": normalized_entropy,
                    "most_common_value": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "most_common_ratio": most_common_ratio,
                    "top_categories": [
                        {"value": str(idx), "count": int(cnt), "percentage": f"{(cnt/total_count)*100:.2f}%"}
                        for idx, cnt in value_counts.head(5).items()
                    ]
                }
            })

    # 3. DateTime columns (time-based indicators)
    datetime_cols = [col['name'] for col in dataset_profile['columns'] if col['role'] == 'datetime']

    for col in datetime_cols:
        try:
            dt_series = pd.to_datetime(df[col], errors='coerce').dropna()

            if len(dt_series) > 0:
                time_span = dt_series.max() - dt_series.min()
                time_span_days = time_span.days if hasattr(time_span, 'days') else time_span.total_seconds() / 86400

                key_indicators.append({
                    "indicator": col,
                    "indicator_type": "datetime",
                    "significance_score": time_span_days,  # Longer time spans may be more significant
                    "metric_type": "datetime",
                    "description": f"Datetime column spanning {time_span_days:.2f} days",
                    "statistical_properties": {
                        "min_date": dt_series.min().isoformat() if not pd.isna(dt_series.min()) else None,
                        "max_date": dt_series.max().isoformat() if not pd.isna(dt_series.max()) else None,
                        "time_span_days": time_span_days,
                        "total_observations": len(dt_series)
                    }
                })
        except Exception as e:
            logger.warning(f"Error processing datetime column {col}: {e}")

    # Sort indicators by significance score in descending order
    key_indicators.sort(key=lambda x: x['significance_score'], reverse=True)

    return key_indicators


def generate_eda_summary(df: pd.DataFrame, dataset_profile: Dict[str, Any], correlation_insights: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive EDA summary for the dataset

    Args:
        df: Input DataFrame
        dataset_profile: Dataset profile with column information
        correlation_insights: Optional list of correlation insights to incorporate
    """
    logger.info("Starting EDA analysis for dataset")

    # Perform pattern and relationship analysis
    patterns = detect_pattern_relationships(df, dataset_profile)

    # If correlation insights are provided, use them instead of the basic correlations
    if correlation_insights is not None:
        patterns['correlation_insights'] = correlation_insights
    else:
        # Generate correlation insights using the new correlation engine
        try:
            correlation_analysis = analyze_correlations(df, dataset_profile)
            correlation_insights = generate_correlation_insights(correlation_analysis)
            patterns['correlation_insights'] = correlation_insights
        except Exception as e:
            logger.warning(f"Error generating correlation insights: {e}")
            patterns['correlation_insights'] = []
            correlation_insights = []

    # Extract potential use cases
    use_cases = extract_use_cases(df, dataset_profile)

    # Identify key indicators using the correlation insights
    meaningful_correlations = patterns.get('meaningful_correlations', [])
    key_indicators = identify_key_indicators(df, dataset_profile, meaningful_correlations)

    # Create the EDA summary object
    eda_summary = {
        "summary_statistics": {
            "total_rows": dataset_profile.get("n_rows", len(df)),
            "total_columns": dataset_profile.get("n_cols", len(df.columns)),
            "numeric_columns": len([c for c in dataset_profile['columns'] if c['role'] == 'numeric']),
            "categorical_columns": len([c for c in dataset_profile['columns'] if c['role'] in ['categorical', 'boolean']]),
            "datetime_columns": len([c for c in dataset_profile['columns'] if c['role'] == 'datetime']),
            "text_columns": len([c for c in dataset_profile['columns'] if c['role'] == 'text'])
        },
        "patterns_and_relationships": patterns,
        "use_cases": use_cases,
        "key_indicators": key_indicators,
        "correlation_insights": correlation_insights,
        "recommendations": generate_recommendations(df, dataset_profile, patterns, use_cases, key_indicators)
    }

    logger.info("EDA analysis completed")
    return eda_summary


def generate_recommendations(df: pd.DataFrame, dataset_profile: Dict[str, Any], 
                           patterns: Dict[str, Any], use_cases: List[Dict[str, Any]], 
                           key_indicators: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate actionable recommendations based on EDA analysis
    """
    recommendations = []
    
    # Recommendation based on correlations
    strong_correlations = [corr for corr in patterns['correlations'] if corr['strength'] == 'strong']
    if strong_correlations:
        recommendations.append({
            "type": "correlation_insight",
            "title": "Strong correlations detected",
            "description": f"Detected {len(strong_correlations)} strong correlations. These variables may have causal relationships or shared underlying factors.",
            "details": strong_correlations[:3]  # Limit to top 3 for brevity
        })
    
    # Recommendation based on outliers
    outlier_columns = [out for out in patterns['outliers'] if out['outlier_percentage'] > 5]
    if outlier_columns:
        recommendations.append({
            "type": "data_quality",
            "title": "Potential data quality issues",
            "description": f"Detected outliers in {len(outlier_columns)} columns (>5% of values). These may be data entry errors or genuine extreme values.",
            "details": outlier_columns[:3]  # Limit to top 3 for brevity
        })
    
    # Recommendation based on anomalies
    if patterns['anomalies']:
        recommendations.append({
            "type": "data_insight",
            "title": "Anomalies detected",
            "description": f"Found {len(patterns['anomalies'])} anomalies in the dataset that may require further investigation.",
            "details": patterns['anomalies'][:3]  # Limit to top 3 for brevity
        })
    
    # Recommendation based on trends
    if patterns['trends']:
        trend_directions = [trend['trend_type'] for trend in patterns['trends']]
        increasing_trends = trend_directions.count('increasing')
        decreasing_trends = trend_directions.count('decreasing')
        
        recommendations.append({
            "type": "trend_analysis",
            "title": "Time-based trends identified",
            "description": f"Found {len(patterns['trends'])} time-based trends ({increasing_trends} increasing, {decreasing_trends} decreasing).",
            "details": patterns['trends'][:3]  # Limit to top 3 for brevity
        })
    
    # Recommendation based on use cases
    if use_cases:
        recommendations.append({
            "type": "use_case",
            "title": "Suggested use cases for this dataset",
            "description": "Based on the content of your dataset, you could focus on these analytical approaches:",
            "details": use_cases[:2]  # Limit to top 2 for brevity
        })
    
    # Recommendation based on key indicators
    if key_indicators:
        top_indicators = key_indicators[:3]  # Top 3 indicators
        recommendations.append({
            "type": "key_indicator",
            "title": "Key indicators to focus on",
            "description": "These variables have high significance scores and should be prioritized in your analysis:",
            "details": top_indicators
        })
    
    return recommendations
```

## src/core/pipeline.py

```
# src/core/pipeline.py
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import pandas as pd
from datetime import datetime
import time
from src.data.parser import load_csv
from src.data.analyser import basic_profile, build_dataset_profile
from src.ml.kpi_generator import generate_kpis
from src.ml.chart_selector import suggest_charts
from src.viz.plotly_renderer import build_category_count_charts, build_charts_from_specs
from src.viz.simple_renderer import generate_all_chart_data
from src.eda.insights_generator import generate_eda_summary
from src.ml.correlation_engine import analyze_correlations, generate_correlation_insights

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Structured result with warnings and errors"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    errors: List[str] = None
    warnings: List[str] = None
    timing: Dict[str, float] = None

@dataclass
class DashboardState:
    """Structured return type for dashboard state"""
    df: pd.DataFrame
    dataset_profile: Dict[str, Any]
    profile: List[Dict[str, Any]]
    kpis: List[Dict[str, Any]]
    charts: List[Dict[str, Any]]
    primary_chart: Optional[Dict[str, Any]]
    category_charts: Dict[str, Any]
    all_charts: List[Dict[str, Any]]
    eda_summary: Optional[Dict[str, Any]] = None
    correlation_analysis: Optional[Dict[str, Any]] = None
    original_filename: Optional[str] = None

def build_dashboard_from_df(df: pd.DataFrame, max_cols: Optional[int] = None,
                           max_categories: int = 10, max_charts: int = 20,
                           kpi_thresholds: Optional[Dict[str, float]] = None) -> Optional[DashboardState]:
    """
    Core dashboard builder that works from an-in-memory DataFrame.
    All data sources (upload, URL, Kaggle, etc.) should end up here.
    """
    if df is None:
        logger.error("Input DataFrame is None")
        return None

    # Cap rows and columns to prevent expensive processing
    MAX_ROWS = 100000
    if len(df) > MAX_ROWS:
        logger.warning(f"DataFrame has {len(df)} rows, sampling to {MAX_ROWS} for performance")
        df = df.sample(n=min(MAX_ROWS, len(df)), random_state=42)

    start_time = time.time()
    timing = {}

    # 1) Determine max columns
    if max_cols is None:
        MAX_COLS = 50
        max_cols = min(df.shape[1], MAX_COLS)

    logger.info(f"Building dashboard for DataFrame with {df.shape[0]} rows and {df.shape[1]} columns (using up to {max_cols})")

    # 2) Build dataset profile
    profile_start = time.time()
    try:
        dataset_profile = build_dataset_profile(df, max_cols=max_cols)
        if dataset_profile is None:
            logger.error("Dataset profile generation failed")
            return None
        logger.info(f"Dataset profile built with {dataset_profile['n_cols']} columns")
    except Exception as e:
        logger.exception("Error building dataset profile")
        return None
    timing['profile'] = time.time() - profile_start

    # 3) Legacy/simple profile (optional)
    profile_start = time.time()
    try:
        profile = basic_profile(df)
        logger.info(f"Basic profile built for {len(profile)} columns")
    except Exception as e:
        logger.exception("Error building basic profile")
        profile = []
    timing['basic_profile'] = time.time() - profile_start

    # 4) Generate correlation insights using the new engine
    correlation_start = time.time()
    try:
        correlation_analysis = analyze_correlations(df, dataset_profile)
        # Get correlation insights from the analysis
        correlation_insights = generate_correlation_insights(correlation_analysis)
        logger.info(f"Correlation analysis completed with {len(correlation_insights)} insights generated")
    except Exception as e:
        logger.exception("Error generating correlation insights")
        correlation_analysis = None
        correlation_insights = []
    timing['correlation_analysis'] = time.time() - correlation_start

    # 5) Generate EDA summary (now incorporating correlation insights)
    eda_start = time.time()
    try:
        eda_summary = generate_eda_summary(df, dataset_profile, correlation_insights=correlation_insights)
        logger.info(f"EDA summary generated with {len(eda_summary.get('use_cases', []))} use cases and {len(eda_summary.get('key_indicators', []))} key indicators")
    except Exception as e:
        logger.exception("Error generating EDA summary")
        eda_summary = {}
    timing['eda_summary'] = time.time() - eda_start

    # 6) KPIs (now using EDA insights for better identification)
    kpi_start = time.time()
    try:
        kpis = generate_kpis(df, dataset_profile, eda_summary=eda_summary)
        logger.info(f"Generated {len(kpis)} KPIs")
    except Exception as e:
        logger.exception("Error generating KPIs")
        kpis = []
    timing['kpis'] = time.time() - kpi_start

    # 7) Chart suggestions (generic ChartSpec-like dicts)
    chart_start = time.time()
    try:
        charts = suggest_charts(df, dataset_profile, kpis)
        logger.info(f"Suggested {len(charts)} charts")
    except Exception as e:
        logger.exception("Error suggesting charts")
        charts = []
    timing['charts'] = time.time() - chart_start

    # 6) Build multiple category_count charts with semantic awareness and pick a primary one
    category_start = time.time()
    try:
        category_charts = build_category_count_charts(
            df,
            charts,  # chart_specs are the suggestions from suggest_charts
            dataset_profile=dataset_profile,
            max_categories=max_categories,
            max_charts=max_charts
        )
        logger.info(f"Built {len(category_charts)} category count charts")
        primary_chart = next(iter(category_charts.values()), None)
    except Exception as e:
        logger.exception("Error building category charts")
        category_charts = {}
        primary_chart = None
    timing['category_charts'] = time.time() - category_start

    # 7) Build all charts using the new intelligent renderer with semantic awareness
    all_charts_start = time.time()
    try:
        all_charts = build_charts_from_specs(
            df,
            charts,
            dataset_profile=dataset_profile,
            eda_summary=eda_summary,
            max_categories=max_categories,
            max_charts=max_charts
        )
        logger.info(f"Generated {len(all_charts)} all charts")
    except Exception as e:
        logger.exception("Error generating all charts")
        all_charts = []
    timing['all_charts'] = time.time() - all_charts_start

    total_time = time.time() - start_time
    timing['total'] = total_time
    logger.info(f"Dashboard build completed in {total_time:.2f}s")
    logger.info(f"Timing breakdown: {timing}")

    return DashboardState(
        df=df,
        dataset_profile=dataset_profile,
        profile=profile,
        kpis=kpis,
        charts=charts,
        primary_chart=primary_chart,
        category_charts=category_charts,
        all_charts=all_charts,
        eda_summary=eda_summary,
        correlation_analysis=correlation_analysis,
        original_filename=None
    )


def build_dashboard_from_file(file_storage, max_cols: Optional[int] = None,
                             max_categories: int = 10, max_charts: int = 20,
                             kpi_thresholds: Optional[Dict[str, float]] = None,
                             original_filename: Optional[str] = None) -> Optional[DashboardState]:
    """
    Orchestrates the full dashboard build from an uploaded file.
    Keeps the old interface for the upload flow.
    """
    start_time = time.time()
    try:
        df = load_csv(file_storage)
        if df is None:
            logger.error("Failed to load CSV from file storage")
            return None

        state = build_dashboard_from_df(df, max_cols=max_cols,
                                      max_categories=max_categories,
                                      max_charts=max_charts,
                                      kpi_thresholds=kpi_thresholds)

        if state:
            state.original_filename = original_filename

        total_time = time.time() - start_time
        logger.info(f"Dashboard built from file in {total_time:.2f}s")
        return state

    except Exception as e:
        logger.exception("Error in build_dashboard_from_file")
        return None

```

## src/viz/plotly_renderer.py

```
"""
Advanced visualization renderer with semantic awareness and intelligent chart selection.
Works with the new correlation engine and EDA insights to generate meaningful visualizations.
"""

import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from plotly.subplots import make_subplots
import re
import math

logger = logging.getLogger(__name__)

@dataclass
class ChartPayload:
    """Typed chart payload object with validation"""
    title: str
    column: str
    data: List[Dict[str, Any]]
    type: Optional[str] = None
    schema_version: str = "1.0"


def _is_likely_identifier(series: pd.Series, name: str = "") -> bool:
    """
    Robust identifier detection that matches the new correlation engine logic.
    """
    n_total = len(series)
    if n_total == 0:
        return False

    n_unique = series.nunique()
    unique_ratio = n_unique / n_total if n_total > 0 else 0.0

    # High cardinality check
    if unique_ratio > 0.98:
        # Numeric sequential pattern check
        if pd.api.types.is_numeric_dtype(series):
            numeric_vals = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_vals) > 5:
                sorted_vals = numeric_vals.sort_values()
                diffs = sorted_vals.diff().dropna()
                if len(diffs) > 0:
                    # If diffs are mostly 1, likely sequential ID
                    sequential_ratio = (diffs == 1).mean()
                    if sequential_ratio > 0.8:
                        return True
        # UUID pattern check
        if series.dtype == 'object':
            sample = series.dropna().head(20).astype(str)
            uuid_matches = 0
            for val in sample:
                if re.match(r'^[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}$', val, re.IGNORECASE):
                    uuid_matches += 1
            if uuid_matches / len(sample) > 0.5:  # More than 50% are UUIDs
                return True

    # Name-based check
    name_lower = name.lower()
    id_keywords = [
        "id", "uuid", "guid", "key", "code", "no", "number", "index",
        "account", "user", "customer", "product", "item", "order",
        "transaction", "invoice", "booking", "session", "token", "hash"
    ]

    if any(keyword in name_lower for keyword in id_keywords):
        return True

    return False


def _build_category_count_data(
    df: pd.DataFrame,
    column: str,
    max_categories: int = 10,
    dataset_profile: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Builds category count data with intelligent truncation and ID exclusion.
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in dataframe")
        return None

    # Check if this is likely an identifier column
    series = df[column]
    if dataset_profile:
        # Check if column is marked as identifier in dataset profile
        col_profile = next((col for col in dataset_profile.get('columns', []) if col['name'] == column), None)
        if col_profile and col_profile.get('role') == 'identifier':
            logger.info(f"Skipping identifier column '{column}' from category count chart")
            return None
    
    # Also check using our identifier detection function
    if _is_likely_identifier(series, column):
        logger.info(f"Skipping likely identifier column '{column}' from category count chart")
        return None

    # Get value counts
    counts = df[column].value_counts(dropna=True)

    # Handle high cardinality: truncate and add "Others" category
    if len(counts) > max_categories:
        # Keep top N categories and group the rest as "Others"
        top_counts = counts.head(max_categories - 1)  # Leave room for "Others"
        others_count = counts.iloc[max_categories - 1:].sum()

        if others_count > 0:
            top_counts.loc["Others"] = others_count

        logger.info(f"Truncated categories for '{column}' from {len(counts)} to {len(top_counts)} with 'Others' bucket")
        counts = top_counts

    categories = [str(idx) for idx in counts.index if idx is not None]
    values = [int(v) for v in counts.values if not pd.isna(v)]

    if not categories:
        logger.warning(f"No valid categories found for column '{column}' after filtering")
        return None

    table_data = [
        {"category": cat, "count": val}
        for cat, val in zip(categories, values)
    ]

    # Validate and normalize the chart payload
    chart_payload = ChartPayload(
        title=f"Count of {column.replace('_', ' ').title()}",
        column=column,
        data=table_data,
        type="category_count"
    )

    # Convert to dictionary format expected by frontend
    return {
        "title": chart_payload.title,
        "column": chart_payload.column,
        "data": chart_payload.data,
        "type": chart_payload.type
    }


def _build_histogram_data(
    df: pd.DataFrame,
    column: str,
    bins: int = 20,
    dataset_profile: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Builds histogram data with adaptive binning and identifier exclusion.
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in dataframe")
        return None

    # Check if this is likely an identifier column
    series = df[column]
    if dataset_profile:
        # Check if column is marked as identifier in dataset profile
        col_profile = next((col for col in dataset_profile.get('columns', []) if col['name'] == column), None)
        if col_profile and col_profile.get('role') == 'identifier':
            logger.info(f"Skipping identifier column '{column}' from histogram")
            return None
    
    # Also check using our identifier detection function
    if _is_likely_identifier(series, column):
        logger.info(f"Skipping likely identifier column '{column}' from histogram")
        return None

    # Ensure the column is numeric
    series = pd.to_numeric(df[column], errors='coerce')
    # Drop NaN values
    series = series.dropna()
    if series.empty:
        logger.warning(f"Column '{column}' has no valid numeric values after cleaning")
        return None

    # Adaptive binning strategy based on data size and skew
    n_samples = len(series)
    if n_samples < 10:
        # Very small dataset - use fewer bins
        bins = 3
    elif n_samples < 50:
        bins = 5
    elif n_samples > 10000:
        # Very large dataset - limit to reasonable number of bins
        bins = min(bins, 100)
    else:
        bins = min(bins, n_samples // 10)  # At most 10 samples per bin

    # Check skewness and adjust if needed
    if n_samples > 2 and len(series) > 2:
        try:
            # Calculate skewness
            skewness = series.skew()
            if abs(skewness) > 1:  # Highly skewed
                # For highly skewed data, use quantile-based bins instead of equal-width bins
                try:
                    # Divide the data into quantile-based bins for skewed distributions
                    quantiles = pd.qcut(series, bins, duplicates='drop', precision=1)
                    value_counts = quantiles.value_counts(sort=False)
                    
                    categories = [f"{interval.left:.2f} - {interval.right:.2f}" for interval in value_counts.index]
                    values = [int(count) for count in value_counts.values]
                except Exception:
                    # If quantile binning fails, fall back to regular binning
                    hist, bin_edges = pd.cut(series, bins=bins, retbins=True)
                    value_counts = hist.value_counts(sort=False)  # Sort by bin order, not by count
                    valid_indices = [i for i, interval in enumerate(value_counts.index) if pd.notna(interval.left)]
                    categories = [f"{value_counts.index[i].left:.2f} - {value_counts.index[i].right:.2f}" for i in valid_indices]
                    values = [int(value_counts.iloc[i]) for i in valid_indices]
            else:
                # Normal distribution - use equal-width bins
                hist, bin_edges = pd.cut(series, bins=bins, retbins=True)
                value_counts = hist.value_counts(sort=False)  # Sort by bin order, not by count
                valid_indices = [i for i, interval in enumerate(value_counts.index) if pd.notna(interval.left)]
                categories = [f"{value_counts.index[i].left:.2f} - {value_counts.index[i].right:.2f}" for i in valid_indices]
                values = [int(value_counts.iloc[i]) for i in valid_indices]
        except Exception as e:
            logger.warning(f"Error in adaptive binning for {column}: {e}")
            # Fall back to regular binning
            hist, bin_edges = pd.cut(series, bins=bins, retbins=True)
            value_counts = hist.value_counts(sort=False)  # Sort by bin order, not by count
            valid_indices = [i for i, interval in enumerate(value_counts.index) if pd.notna(interval.left)]
            categories = [f"{value_counts.index[i].left:.2f} - {value_counts.index[i].right:.2f}" for i in valid_indices]
            values = [int(value_counts.iloc[i]) for i in valid_indices]
    else:
        # Default case for small datasets
        hist, bin_edges = pd.cut(series, bins=bins, retbins=True)
        value_counts = hist.value_counts(sort=False)  # Sort by bin order, not by count
        valid_indices = [i for i, interval in enumerate(value_counts.index) if pd.notna(interval.left)]
        categories = [f"{value_counts.index[i].left:.2f} - {value_counts.index[i].right:.2f}" for i in valid_indices]
        values = [int(value_counts.iloc[i]) for i in valid_indices]

    # If there are too many bins with low counts, consider it sparse and reduce bins
    if len(categories) > 10:
        low_count_bins = sum(1 for count in values if count < 2)
        if low_count_bins > len(values) * 0.6:  # 60% of bins have low counts
            # Reduce number of bins
            new_bins = max(5, len(categories) // 2)
            try:
                hist, bin_edges = pd.cut(series, bins=new_bins, retbins=True)
                value_counts = hist.value_counts(sort=False)  # Sort by bin order, not by count
                valid_indices = [i for i, interval in enumerate(value_counts.index) if pd.notna(interval.left)]
                categories = [f"{value_counts.index[i].left:.2f} - {value_counts.index[i].right:.2f}" for i in valid_indices]
                values = [int(value_counts.iloc[i]) for i in valid_indices]
            except Exception as e:
                logger.warning(f"Error adjusting bins for {column}: {e}")

    if not categories:
        if not series.empty:
            # If we have data but couldn't create bins, just use the single value
            single_value = series.iloc[0]
            categories = [str(single_value)]
            values = [len(series)]
        else:
            logger.warning(f"No valid categories found for histogram of column '{column}'")
            return None

    table_data = [
        {"bin_range": cat, "count": val}
        for cat, val in zip(categories, values)
    ]

    # Validate and normalize the chart payload
    chart_payload = ChartPayload(
        title=f"Distribution of {column.replace('_', ' ').title()}",
        column=column,
        data=table_data,
        type="histogram"
    )

    return {
        "title": chart_payload.title,
        "column": chart_payload.column,
        "data": chart_payload.data,
        "type": chart_payload.type
    }


def _build_category_summary_data(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    agg_func: str = "mean",
    dataset_profile: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Builds category vs numeric summary data with ID filtering and semantic awareness.
    """
    if x_column not in df.columns or y_column not in df.columns:
        logger.warning(f"One of columns '{x_column}' or '{y_column}' not found in dataframe")
        return None

    # Check if either column is likely an identifier
    x_series = df[x_column]
    y_series = df[y_column]
    
    if _is_likely_identifier(x_series, x_column) or _is_likely_identifier(y_series, y_column):
        logger.info(f"Skipping summary for columns '{x_column}' and '{y_column}' - one is likely identifier")
        return None

    # Check dataset profile for roles
    if dataset_profile:
        x_profile = next((col for col in dataset_profile.get('columns', []) if col['name'] == x_column), None)
        y_profile = next((col for col in dataset_profile.get('columns', []) if col['name'] == y_column), None)
        
        if x_profile and x_profile.get('role') == 'identifier':
            logger.info(f"Skipping summary - X column '{x_column}' is an identifier")
            return None
        if y_profile and y_profile.get('role') == 'identifier':
            logger.info(f"Skipping summary - Y column '{y_column}' is an identifier")
            return None

    # Drop rows where either column is NaN
    combined_series = pd.concat([x_series, y_series], axis=1).dropna()
    if combined_series.empty:
        logger.warning(f"No valid data for category summary between '{x_column}' and '{y_column}' after dropping NaNs")
        return None

    x_clean = combined_series.iloc[:, 0]
    y_clean = pd.to_numeric(combined_series.iloc[:, 1], errors='coerce')

    # Create valid combined data
    valid_data = pd.concat([x_clean, y_clean], axis=1).dropna()
    if valid_data.empty:
        logger.warning(f"No valid combined data for category summary between '{x_column}' and '{y_column}'")
        return None

    x_final = valid_data.iloc[:, 0]
    y_final = valid_data.iloc[:, 1]

    # Apply aggregation by group
    if agg_func == "sum":
        result = y_final.groupby(x_final).sum()
    elif agg_func == "mean":
        result = y_final.groupby(x_final).mean()
    elif agg_func == "count":
        result = y_final.groupby(x_final).count()
    elif agg_func == "min":
        result = y_final.groupby(x_final).min()
    elif agg_func == "max":
        result = y_final.groupby(x_final).max()
    elif agg_func == "std":
        result = y_final.groupby(x_final).std()
    elif agg_func == "median":
        result = y_final.groupby(x_final).median()
    else:
        # Default to mean
        result = y_final.groupby(x_final).mean()

    # Check if we have enough categories to visualize meaningfully (not too many for readability)
    if len(result) > 20:
        logger.info(f"Too many categories ({len(result)}) for '{x_column}' vs '{y_column}' summary, skipping")
        return None

    categories = [str(idx) for idx in result.index if idx is not None]
    values = [float(val) for val in result.values if pd.notna(val)]

    if not categories:
        logger.warning(f"No valid categories for summary between '{x_column}' and '{y_column}'")
        return None

    table_data = [
        {"category": cat, "agg_value": val}
        for cat, val in zip(categories, values)
    ]

    # Validate and normalize the chart payload
    agg_display = agg_func.title()
    chart_payload = ChartPayload(
        title=f"{agg_display} of {y_column.replace('_', ' ').title()} by {x_column.replace('_', ' ').title()}",
        column=f"agg_{agg_func}_{x_column}",
        data=table_data,
        type="category_summary"
    )

    return {
        "title": chart_payload.title,
        "x_column": x_column,
        "y_column": y_column,
        "data": chart_payload.data,
        "type": chart_payload.type,
        "agg_func": agg_func
    }


def _build_time_series_data(
    df: pd.DataFrame,
    x_column: str,  # datetime column
    y_column: str,  # numeric column
    agg_func: str = "mean",
    dataset_profile: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Builds time series data with datetime validation and identifier exclusion.
    """
    if x_column not in df.columns or y_column not in df.columns:
        logger.warning(f"One of columns '{x_column}' or '{y_column}' not found in dataframe")
        return None

    # Check if X column is datetime-capable
    x_series = df[x_column]
    try:
        x_dt = pd.to_datetime(x_series, errors='coerce')
    except Exception as e:
        logger.warning(f"Cannot convert column '{x_column}' to datetime: {e}")
        return None

    # If X column is likely an identifier, skip
    if _is_likely_identifier(x_series, x_column):
        logger.info(f"Skipping time series - X column '{x_column}' is likely an identifier")
        return None

    # Check dataset profile for roles
    if dataset_profile:
        x_profile = next((col for col in dataset_profile.get('columns', []) if col['name'] == x_column), None)
        y_profile = next((col for col in dataset_profile.get('columns', []) if col['name'] == y_column), None)
        
        if x_profile and x_profile.get('role') == 'identifier':
            logger.info(f"Skipping time series - X column '{x_column}' is an identifier")
            return None
        if y_profile and y_profile.get('role') == 'identifier':
            logger.info(f"Skipping time series - Y column '{y_column}' is an identifier")
            return None

    # Convert Y to numeric
    y_series = df[y_column]
    y_numeric = pd.to_numeric(y_series, errors='coerce')

    # Combine and drop NaN values
    combined = pd.concat([x_dt, y_numeric], axis=1).dropna()
    if combined.empty or len(combined) < 2:
        logger.warning(f"Not enough valid data points for time series between '{x_column}' and '{y_column}'")
        return None

    x_final = combined.iloc[:, 0]
    y_final = combined.iloc[:, 1]

    # Check if time series is meaningful (not just one repeated value)
    if y_final.nunique() < 2:
        logger.info(f"Y column '{y_column}' has less than 2 unique values over time, skipping time series")
        return None

    # Group by date if there are duplicate dates
    if x_final.duplicated().any():
        grouped = y_final.groupby(x_final).agg(agg_func)
        dates = [dt.isoformat() for dt in grouped.index]
        values = [float(val) for val in grouped.values]
    else:
        # Sort by date
        sorted_combined = combined.sort_values(x_column)
        dates = [dt.isoformat() for dt in sorted_combined.iloc[:, 0]]
        values = [float(val) for val in sorted_combined.iloc[:, 1]]

    if not dates:
        logger.warning(f"No valid dates for time series between '{x_column}' and '{y_column}'")
        return None

    table_data = [
        {"date": date, "value": val}
        for date, val in zip(dates, values)
    ]

    # Validate and normalize the chart payload
    chart_payload = ChartPayload(
        title=f"Trend of {y_column.replace('_', ' ').title()} over {x_column.replace('_', ' ').title()}",
        column=f"ts_{x_column}_{y_column}",
        data=table_data,
        type="time_series"
    )

    return {
        "title": chart_payload.title,
        "x_column": x_column,
        "y_column": y_column,
        "data": chart_payload.data,
        "type": chart_payload.type,
        "agg_func": agg_func
    }


def _build_scatter_data(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    dataset_profile: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Builds scatter plot data with identifier exclusion and correlation validation.
    """
    if x_column not in df.columns or y_column not in df.columns:
        logger.warning(f"One of columns '{x_column}' or '{y_column}' not found in dataframe")
        return None

    # Check if either column is likely an identifier
    x_series = df[x_column]
    y_series = df[y_column]
    
    if _is_likely_identifier(x_series, x_column) or _is_likely_identifier(y_series, y_column):
        logger.info(f"Skipping scatter plot for columns '{x_column}' and '{y_column}' - identifier detected")
        return None

    # Check dataset profile for roles
    if dataset_profile:
        x_profile = next((col for col in dataset_profile.get('columns', []) if col['name'] == x_column), None)
        y_profile = next((col for col in dataset_profile.get('columns', []) if col['name'] == y_column), None)
        
        if x_profile and x_profile.get('role') == 'identifier':
            logger.info(f"Skipping scatter - X column '{x_column}' is an identifier")
            return None
        if y_profile and y_profile.get('role') == 'identifier':
            logger.info(f"Skipping scatter - Y column '{y_column}' is an identifier")
            return None

    # Convert both to numeric
    x_numeric = pd.to_numeric(x_series, errors='coerce')
    y_numeric = pd.to_numeric(y_series, errors='coerce')

    # Combine and drop NaN values
    combined = pd.concat([x_numeric, y_numeric], axis=1).dropna()
    if combined.empty or len(combined) < 3:  # Need at least 3 points for meaningful scatter
        logger.warning(f"Not enough valid data points for scatter plot between '{x_column}' and '{y_column}'")
        return None

    x_final = combined.iloc[:, 0]
    y_final = combined.iloc[:, 1]

    # Check for minimal variance (constant or near-constant values)
    x_std = x_final.std()
    y_std = y_final.std()
    
    if pd.isna(x_std) or pd.isna(y_std) or x_std < 0.001 or y_std < 0.001:
        logger.info(f"Low variance in columns '{x_column}' or '{y_column}', skipping scatter plot")
        return None

    # Prepare data
    x_values = [float(val) for val in x_final]
    y_values = [float(val) for val in y_final]

    table_data = [
        {"x": x_val, "y": y_val}
        for x_val, y_val in zip(x_values, y_values)
    ]

    if not x_values or not y_values or len(x_values) != len(y_values):
        logger.warning(f"Mismatched array lengths for scatter plot between '{x_column}' and '{y_column}'")
        return None

    # Validate and normalize the chart payload
    chart_payload = ChartPayload(
        title=f"{x_column.replace('_', ' ').title()} vs {y_column.replace('_', ' ').title()}",
        column=f"scatter_{x_column}_{y_column}",
        data=table_data,
        type="scatter"
    )

    return {
        "title": chart_payload.title,
        "x_column": x_column,
        "y_column": y_column,
        "data": chart_payload.data,
        "type": chart_payload.type
    }


def _build_pie_data(
    df: pd.DataFrame,
    column: str,
    max_categories: int = 10,
    dataset_profile: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Builds pie chart data with cardinality validation and identifier exclusion.
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in dataframe")
        return None

    # Check if this is likely an identifier
    series = df[column]
    if _is_likely_identifier(series, column):
        logger.info(f"Skipping pie chart for column '{column}' - likely identifier")
        return None

    # Check dataset profile for roles
    if dataset_profile:
        col_profile = next((col for col in dataset_profile.get('columns', []) if col['name'] == column), None)
        if col_profile and col_profile.get('role') == 'identifier':
            logger.info(f"Skipping pie chart - column '{column}' is an identifier")
            return None

    # Get value counts
    counts = series.value_counts(dropna=True)

    # For pie charts, limit to a reasonable number of categories for readability
    if len(counts) > max_categories:
        logger.info(f"Too many categories ({len(counts)}) for pie chart of '{column}', only taking top {max_categories}")
        counts = counts.head(max_categories)

    categories = [str(idx) for idx in counts.index if idx is not None]
    values = [int(val) for val in counts.values if pd.notna(val)]

    if not categories:
        logger.warning(f"No valid categories for pie chart of column '{column}'")
        return None

    table_data = [
        {"category": cat, "value": val}
        for cat, val in zip(categories, values)
    ]

    # Validate and normalize the chart payload
    chart_payload = ChartPayload(
        title=f"Distribution of {column.replace('_', ' ').title()}",
        column=column,
        data=table_data,
        type="pie"
    )

    return {
        "title": chart_payload.title,
        "column": column,
        "data": chart_payload.data,
        "type": chart_payload.type
    }


def _build_box_plot_data(
    df: pd.DataFrame,
    x_column: str,  # categorical column
    y_column: str,  # numeric column
    dataset_profile: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Builds box plot data with appropriate validation for identifier exclusion.
    """
    if x_column not in df.columns or y_column not in df.columns:
        logger.warning(f"One of columns '{x_column}' or '{y_column}' not found in dataframe")
        return None

    # Check if either column is likely an identifier
    x_series = df[x_column]
    y_series = df[y_column]
    
    if _is_likely_identifier(x_series, x_column) or _is_likely_identifier(y_series, y_column):
        logger.info(f"Skipping box plot for columns '{x_column}' and '{y_column}' - identifier detected")
        return None

    # Check dataset profile for roles
    if dataset_profile:
        x_profile = next((col for col in dataset_profile.get('columns', []) if col['name'] == x_column), None)
        y_profile = next((col for col in dataset_profile.get('columns', []) if col['name'] == y_column), None)
        
        if x_profile and x_profile.get('role') == 'identifier':
            logger.info(f"Skipping box plot - X column '{x_column}' is an identifier")
            return None
        if y_profile and y_profile.get('role') == 'identifier':
            logger.info(f"Skipping box plot - Y column '{y_column}' is an identifier")
            return None

    # Convert Y to numeric
    y_numeric = pd.to_numeric(y_series, errors='coerce')

    # Combine and drop NaN values
    combined = pd.concat([x_series, y_numeric], axis=1).dropna()
    if combined.empty:
        logger.warning(f"No valid data for box plot between '{x_column}' and '{y_column}'")
        return None

    x_final = combined.iloc[:, 0]
    y_final = combined.iloc[:, 1]

    # Check that X column has multiple categories to compare
    n_unique_x = x_final.nunique()
    if n_unique_x < 2:
        logger.info(f"X column '{x_column}' has less than 2 unique values, skipping box plot")
        return None

    # Limit to reasonable number of categories for readability
    if n_unique_x > 20:
        logger.info(f"X column '{x_column}' has too many categories ({n_unique_x}), skipping box plot")
        return None

    # Group the data for box plot
    grouped = y_final.groupby(x_final)
    
    table_data = []
    for category, values in grouped:
        value_list = [float(v) for v in values if pd.notna(v)]
        if value_list:  # Only add if there are values
            table_data.append({
                "category": str(category),
                "values": value_list
            })

    if not table_data:
        logger.warning(f"No valid data for box plot between '{x_column}' and '{y_column}'")
        return None

    # Validate and normalize the chart payload
    chart_payload = ChartPayload(
        title=f"Distribution of {y_column.replace('_', ' ').title()} by {x_column.replace('_', ' ').title()}",
        column=f"box_{x_column}",
        data=table_data,
        type="box_plot"
    )

    return {
        "title": chart_payload.title,
        "x_column": x_column,
        "y_column": y_column,
        "data": chart_payload.data,
        "type": chart_payload.type
    }


def _build_correlation_heatmap_data(
    df: pd.DataFrame,
    dataset_profile: Dict[str, Any],
    correlation_insights: Optional[List[Dict[str, Any]]] = None
) -> Optional[Dict[str, Any]]:
    """
    Builds correlation heatmap data using the new correlation engine insights.
    Filters out identifiers and low-meaningful correlations.
    """
    if df.empty:
        logger.warning("DataFrame is empty, cannot build correlation heatmap")
        return None

    # Get numeric columns that are not identifiers
    numeric_cols = []
    for col in dataset_profile.get('columns', []):
        if col.get('role') == 'numeric':
            # Check if this column is likely an identifier
            series = df[col['name']]
            if not _is_likely_identifier(series, col['name']):
                numeric_cols.append(col['name'])

    if len(numeric_cols) < 2:
        logger.info(f"Not enough meaningful numeric columns for correlation heatmap ({len(numeric_cols)} found)")
        return None

    # Use only the meaningful numeric columns
    numeric_df = df[numeric_cols]
    
    # Convert to numeric values, handling any potential issues
    for col_name in numeric_df.columns:
        numeric_df[col_name] = pd.to_numeric(numeric_df[col_name], errors='coerce')
    
    # Drop columns that didn't convert properly (ended up with too many NaNs)
    numeric_df = numeric_df.select_dtypes(include=[np.number])
    numeric_cols = [col for col in numeric_cols if col in numeric_df.columns]

    if len(numeric_cols) < 2:
        logger.info(f"After cleaning, not enough meaningful numeric columns for correlation heatmap ({len(numeric_cols)} found)")
        return None

    # Calculate correlation matrix
    try:
        corr_matrix = numeric_df.corr()
        
        # Only include correlations that are meaningful (>0.1 absolute value) or if we have correlation insights
        meaningful_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if pd.notna(corr_val) and abs(corr_val) > 0.1:  # Only meaningful correlations
                    meaningful_corrs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        if not meaningful_corrs:
            logger.info("No meaningful correlations found for heatmap")
            return None

    except Exception as e:
        logger.error(f"Error calculating correlation matrix: {e}")
        return None

    # Prepare data for heatmap
    categories = [str(col) for col in corr_matrix.columns]
    values_matrix = corr_matrix.values.tolist()

    # Ensure valid correlation values in the range [-1, 1]
    for row_idx, row in enumerate(values_matrix):
        for col_idx, val in enumerate(row):
            if pd.isna(val):
                values_matrix[row_idx][col_idx] = 0.0
            else:
                # Clamp correlation values to acceptable range
                values_matrix[row_idx][col_idx] = max(-1.0, min(1.0, val))

    table_data = {
        "categories": categories,
        "values": values_matrix
    }

    # Validate and normalize the chart payload
    chart_payload = ChartPayload(
        title="Correlation Heatmap (Meaningful Relationships Only)",
        column="correlation_matrix",
        data=table_data,
        type="correlation_heatmap"
    )

    return {
        "title": chart_payload.title,
        "data": chart_payload.data,
        "type": chart_payload.type
    }


def build_charts_from_specs(
    df: pd.DataFrame,
    chart_specs,
    dataset_profile: Optional[Dict[str, Any]] = None,
    eda_summary: Optional[Dict[str, Any]] = None,
    max_categories: int = 10,
    max_charts: int = 20,
) -> Dict[str, Any]:
    """
    Intelligent chart suggestion system that considers column roles, semantic tags,
    and meaningful relationships instead of naive heuristics.
    
    Args:
        df: Input DataFrame
        chart_specs: Specifications for what charts to build
        dataset_profile: Dataset profile with column information
        eda_summary: EDA summary with additional insights
        max_categories: Maximum categories for categorical charts
        max_charts: Maximum number of charts to build
        
    Returns:
        Dictionary mapping chart IDs to chart specifications
    """
    if df.empty:
        logger.warning("DataFrame is empty, returning empty charts")
        return {}

    n_rows, n_cols = df.shape
    if n_rows == 0 or n_cols == 0:
        logger.warning(f"Invalid dataframe shape: {n_rows}x{n_cols}")
        return {}

    logger.info(f"Building charts for dataset with {n_rows} rows and {n_cols} columns")

    charts = {}

    if not chart_specs:
        logger.warning("No chart specs provided")
        return charts

    # Process each chart specification
    for spec in chart_specs:
        intent = spec.get('intent')
        chart_id = str(spec.get('id', f'chart_{len(charts)}'))
        
        # Skip if we've reached the maximum charts
        if len(charts) >= max_charts:
            break

        chart_data = None

        try:
            if intent == 'category_count':
                col = spec.get('x_field')
                if col:
                    chart_data = _build_category_count_data(
                        df, 
                        column=col, 
                        max_categories=max_categories,
                        dataset_profile=dataset_profile
                    )

            elif intent == 'histogram':
                col = spec.get('x_field')
                if col:
                    chart_data = _build_histogram_data(
                        df, 
                        column=col,
                        dataset_profile=dataset_profile
                    )

            elif intent == 'category_summary':
                x_col = spec.get('x_field')
                y_col = spec.get('y_field')
                agg_func = spec.get('agg_func', 'mean')
                if x_col and y_col:
                    chart_data = _build_category_summary_data(
                        df, 
                        x_column=x_col, 
                        y_column=y_col, 
                        agg_func=agg_func,
                        dataset_profile=dataset_profile
                    )

            elif intent == 'time_series':
                x_col = spec.get('x_field')
                y_col = spec.get('y_field')
                agg_func = spec.get('agg_func', 'mean')
                if x_col and y_col:
                    chart_data = _build_time_series_data(
                        df, 
                        x_column=x_col, 
                        y_column=y_col, 
                        agg_func=agg_func,
                        dataset_profile=dataset_profile
                    )

            elif intent == 'scatter':
                x_col = spec.get('x_field')
                y_col = spec.get('y_field')
                if x_col and y_col:
                    chart_data = _build_scatter_data(
                        df, 
                        x_column=x_col, 
                        y_column=y_col,
                        dataset_profile=dataset_profile
                    )

            elif intent == 'category_pie':
                col = spec.get('x_field')
                if col:
                    chart_data = _build_pie_data(
                        df, 
                        column=col,
                        max_categories=max_categories,
                        dataset_profile=dataset_profile
                    )

            elif intent == 'box_plot':
                x_col = spec.get('x_field')
                y_col = spec.get('y_field')
                if x_col and y_col:
                    chart_data = _build_box_plot_data(
                        df,
                        x_column=x_col,
                        y_column=y_col,
                        dataset_profile=dataset_profile
                    )

            elif intent == 'correlation_matrix':
                # Use the new correlation heatmap builder that properly filters identifiers
                chart_data = _build_correlation_heatmap_data(
                    df,
                    dataset_profile=dataset_profile,
                    correlation_insights=eda_summary.get('correlation_insights', []) if eda_summary else []
                )

            else:
                logger.warning(f"Unknown chart intent: {intent}")
                continue

            # Add chart data if valid and not already added
            if chart_data and chart_id not in charts:
                charts[chart_id] = chart_data

        except Exception as e:
            logger.error(f"Error building chart with intent '{intent}' and ID '{chart_id}': {e}")
            continue

    logger.info(f"Built {len(charts)} valid charts from {len(chart_specs) if chart_specs else 0} specifications")
    return charts


def build_category_count_charts(
    df: pd.DataFrame,
    chart_specs,
    dataset_profile: Optional[Dict[str, Any]] = None,
    max_categories: int = 10,
    max_charts: int = 20,
) -> Dict[str, Any]:
    """
    Builds multiple category count charts with intelligent filtering and ID exclusion.
    """
    if df.empty:
        logger.warning("DataFrame is empty, returning empty category charts")
        return {}

    category_charts = {}

    if not chart_specs:
        logger.warning("No chart specs provided for category charts")
        return category_charts

    for spec in chart_specs:
        if spec.get("intent") != "category_count":
            continue

        col = spec.get("x_field")
        if not col or col in category_charts:
            continue

        # Check if column is likely an identifier before building chart
        series = df[col]
        if _is_likely_identifier(series, col):
            logger.info(f"Skipping identifier column '{col}' from category count chart")
            continue

        # Check dataset profile for roles
        if dataset_profile:
            col_profile = next((c for c in dataset_profile.get('columns', []) if c['name'] == col), None)
            if col_profile and col_profile.get('role') == 'identifier':
                logger.info(f"Skipping identifier column '{col}' from category count chart")
                continue

        chart_obj = _build_category_count_data(
            df,
            column=col,
            max_categories=max_categories,
            dataset_profile=dataset_profile
        )

        if chart_obj is not None:
            category_charts[col] = chart_obj

        if len(category_charts) >= max_charts:
            break

    logger.info(f"Built {len(category_charts)} category count charts")
    return category_charts
```

## src/viz/simple_renderer.py

```
"""
Simple and reliable chart renderer that creates ready-to-use chart data for the frontend.
"""

import pandas as pd
import numpy as np
import json


def create_chart_data(df, dataset_profile, chart_type, x_col=None, y_col=None, agg_func=None):
    """
    Creates ready-to-use chart data for various chart types.
    """
    chart_data = {
        'type': chart_type,
        'x_col': x_col,
        'y_col': y_col,
        'title': f'{chart_type.title()} Chart',
        'data': []
    }
    
    if chart_type == 'bar':
        if x_col and y_col:
            # Numeric by categorical (e.g., average sales by category)
            if x_col in df.columns and y_col in df.columns:
                grouped = df.groupby(x_col)[y_col].agg(agg_func or 'mean').dropna()
                chart_data['data'] = [{'x': str(idx), 'y': float(val)} for idx, val in grouped.items()]
                chart_data['title'] = f'{agg_func or "Average"} {y_col} by {x_col}'
        elif x_col:
            # Simple category count
            if x_col in df.columns:
                counts = df[x_col].value_counts().head(20)  # Limit for performance
                chart_data['data'] = [{'x': str(idx), 'y': int(val)} for idx, val in counts.items()]
                chart_data['title'] = f'Count of {x_col}'
    
    elif chart_type == 'line':
        if x_col and y_col:
            # Time series or ordered series
            if x_col in df.columns and y_col in df.columns:
                # Convert x to datetime if it looks like a date
                x_series = df[x_col]
                if df[x_col].dtype == 'object':
                    try:
                        x_series = pd.to_datetime(df[x_col], errors='coerce')
                    except:
                        pass  # Keep as is if conversion fails
                
                # Sort by x to get proper line chart
                plot_df = pd.DataFrame({x_col: x_series, y_col: df[y_col]}).dropna()
                plot_df = plot_df.sort_values(x_col)
                
                chart_data['data'] = [{'x': str(row[x_col]), 'y': float(row[y_col])} 
                                     for _, row in plot_df.iterrows() 
                                     if pd.notna(row[x_col]) and pd.notna(row[y_col])]
                chart_data['title'] = f'{y_col} over {x_col}'
    
    elif chart_type == 'scatter':
        if x_col and y_col:
            if x_col in df.columns and y_col in df.columns:
                x_series = pd.to_numeric(df[x_col], errors='coerce').dropna()
                y_series = pd.to_numeric(df[y_col], errors='coerce').dropna()
                
                # Combine x and y series and drop rows with NaN in either column
                combined = pd.DataFrame({x_col: x_series, y_col: y_series}).dropna()
                
                chart_data['data'] = [{'x': float(row[x_col]), 'y': float(row[y_col])} 
                                     for _, row in combined.iterrows()]
                chart_data['title'] = f'{y_col} vs {x_col}'
    
    elif chart_type == 'histogram':
        if x_col:
            if x_col in df.columns:
                # Convert to numeric and drop NaN
                numeric_series = pd.to_numeric(df[x_col], errors='coerce').dropna()
                if len(numeric_series) > 0:
                    # Create bins for histogram
                    counts, bins = np.histogram(numeric_series, bins=min(20, len(numeric_series)//4))
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    
                    chart_data['data'] = [{'x': float(center), 'y': int(count)} 
                                         for center, count in zip(bin_centers, counts)]
                    chart_data['title'] = f'Distribution of {x_col}'
    
    elif chart_type == 'pie':
        if x_col:
            if x_col in df.columns:
                counts = df[x_col].value_counts().head(10)  # Limit for clarity
                chart_data['data'] = [{'label': str(idx), 'value': int(val)} 
                                     for idx, val in counts.items()]
                chart_data['title'] = f'Distribution of {x_col}'
    
    # Add error handling for zero data
    if not chart_data['data']:
        chart_data['data'] = []  # Ensure it's an empty array rather than None
    
    return chart_data


def generate_all_chart_data(df, dataset_profile):
    """
    Generates all possible charts based on the dataset.
    """
    charts = []
    
    # Get column information
    numeric_cols = [col['name'] for col in dataset_profile['columns'] if col['role'] == 'numeric']
    categorical_cols = [col['name'] for col in dataset_profile['columns'] if col['role'] == 'categorical']
    datetime_cols = [col['name'] for col in dataset_profile['columns'] if col['role'] == 'datetime']
    
    # Create bar charts: numeric by categorical
    if numeric_cols and categorical_cols:
        charts.append(create_chart_data(df, dataset_profile, 'bar', 
                                      x_col=categorical_cols[0], y_col=numeric_cols[0], agg_func='mean'))
    
    # Create bar charts: categorical counts
    for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
        charts.append(create_chart_data(df, dataset_profile, 'bar', x_col=col))
    
    # Create line/time series charts
    for dt_col in datetime_cols:
        for num_col in numeric_cols[:2]:  # Limit to first 2 numeric cols per datetime
            charts.append(create_chart_data(df, dataset_profile, 'line', 
                                          x_col=dt_col, y_col=num_col))
    
    # Create scatter plots for numeric vs numeric
    if len(numeric_cols) >= 2:
        charts.append(create_chart_data(df, dataset_profile, 'scatter', 
                                      x_col=numeric_cols[0], y_col=numeric_cols[1]))
    
    # Create histograms for numeric columns
    for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
        charts.append(create_chart_data(df, dataset_profile, 'histogram', x_col=col))
    
    # Create pie charts for categorical columns
    for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
        if dataset_profile['columns'][categorical_cols.index(col)]['unique_count'] <= 10:
            charts.append(create_chart_data(df, dataset_profile, 'pie', x_col=col))
    
    # Create a generic chart for first numeric column if no other charts
    if not charts and numeric_cols:
        charts.append(create_chart_data(df, dataset_profile, 'histogram', x_col=numeric_cols[0]))
    
    return charts
```

## src/viz/eda_visualizer.py

```
"""
Visualization module for EDA insights and key indicators
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any, Optional
import numpy as np

def create_correlation_heatmap(correlations: List[Dict[str, Any]], top_n: int = 10) -> go.Figure:
    """
    Create a correlation heatmap for the strongest correlations
    """
    if not correlations:
        return go.Figure()
    
    # Sort correlations by absolute correlation value and get top N
    sorted_corr = sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)[:top_n]
    
    if not sorted_corr:
        return go.Figure()
    
    # Extract variable names
    variables = set()
    for corr in sorted_corr:
        variables.add(corr['variable1'])
        variables.add(corr['variable2'])
    
    variables = sorted(list(variables))
    n_vars = len(variables)
    
    # Create correlation matrix
    corr_matrix = np.zeros((n_vars, n_vars))
    # Initialize with NaN to indicate no correlation calculated
    corr_matrix[:] = np.nan
    
    # Fill in the matrix
    var_to_idx = {var: i for i, var in enumerate(variables)}
    for corr in sorted_corr:
        i, j = var_to_idx[corr['variable1']], var_to_idx[corr['variable2']]
        corr_matrix[i][j] = corr['correlation']
        corr_matrix[j][i] = corr['correlation']  # Symmetric
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=variables,
        y=variables,
        colorscale='RdBu',
        zmid=0,
        text=np.where(np.isnan(corr_matrix), '', np.round(corr_matrix, 2)),
        texttemplate="%{text}",
        textfont={"size": 12},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Correlation Heatmap",
        xaxis_title="Variables",
        yaxis_title="Variables",
        width=800,
        height=700
    )
    
    return fig


def create_key_indicators_bar(key_indicators: List[Dict[str, Any]], top_n: int = 10) -> go.Figure:
    """
    Create a bar chart showing top key indicators by significance score
    """
    if not key_indicators:
        return go.Figure()
    
    # Get top N indicators
    top_indicators = key_indicators[:top_n]
    
    names = [ind['indicator'] for ind in top_indicators]
    scores = [ind['significance_score'] for ind in top_indicators]
    types = [ind['indicator_type'] for ind in top_indicators]
    
    # Create color mapping for different indicator types
    color_map = {'numeric': '#1f77b4', 'categorical': '#ff7f0e', 'datetime': '#2ca02c'}
    colors = [color_map.get(ind_type, '#7f7f7f') for ind_type in types]
    
    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=scores,
            marker_color=colors,
            text=[f"{score:.2f}" for score in scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Key Indicators by Significance Score",
        xaxis_title="Indicator",
        yaxis_title="Significance Score",
        xaxis_tickangle=-45,
        width=800,
        height=600
    )
    
    return fig


def create_patterns_timeline(trends: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a timeline showing detected trends
    """
    if not trends:
        return go.Figure()
    
    # Create a simple visualization of trend types
    trend_types = [trend['trend_type'] for trend in trends]
    counts = {t: trend_types.count(t) for t in set(trend_types)}
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(counts.keys()),
            y=list(counts.values()),
            marker_color=['#1f77b4' if t == 'increasing' else '#ff7f0e' if t == 'decreasing' else '#2ca02c' for t in counts.keys()]
        )
    ])
    
    fig.update_layout(
        title="Distribution of Time Series Trends",
        xaxis_title="Trend Type",
        yaxis_title="Count",
        width=600,
        height=400
    )
    
    return fig


def create_outliers_visualization(outliers: List[Dict[str, Any]], top_n: int = 10) -> go.Figure:
    """
    Create a visualization showing outlier detection results
    """
    if not outliers:
        return go.Figure()
    
    # Get top N outlier columns
    top_outliers = outliers[:top_n]
    
    names = [out['column'] for out in top_outliers]
    outlier_counts = [out['outlier_count'] for out in top_outliers]
    outlier_percentages = [out['outlier_percentage'] for out in top_outliers]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Outlier Count", "Outlier Percentage"),
        vertical_spacing=0.1
    )
    
    # Add outlier count bar chart
    fig.add_trace(
        go.Bar(x=names, y=outlier_counts, name="Outlier Count", marker_color="#d62728"),
        row=1, col=1
    )
    
    # Add outlier percentage bar chart
    fig.add_trace(
        go.Bar(x=names, y=outlier_percentages, name="Outlier %", marker_color="#9467bd"),
        row=2, col=1
    )
    
    fig.update_layout(
        title="Outlier Detection Results",
        height=600,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Column", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Percentage", row=2, col=1)
    
    # Rotate x-axis labels to prevent overlap
    fig.update_xaxes(tickangle=-45)
    
    return fig


def create_use_cases_visualization(use_cases: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a visualization showing the various detected use cases
    """
    if not use_cases:
        return go.Figure()
    
    # Extract use case names and descriptions
    names = [uc['use_case'][:30] + "..." if len(uc['use_case']) > 30 else uc['use_case'] for uc in use_cases]
    descriptions = [uc['description'][:50] + "..." if len(uc['description']) > 50 else uc['description'] for uc in use_cases]
    
    # Create a simple bar chart showing number of key inputs per use case
    key_input_counts = [len(uc['key_inputs']) for uc in use_cases]
    
    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=key_input_counts,
            text=descriptions,
            hovertemplate='<b>%{x}</b><br>' +
                         'Key Inputs: %{y}<br>' +
                         'Description: %{text}<br>' +
                         '<extra></extra>',
            marker_color="#17becf"
        )
    ])
    
    fig.update_layout(
        title="Dataset Use Cases and Key Inputs",
        xaxis_title="Use Case",
        yaxis_title="Number of Key Inputs",
        xaxis_tickangle=-45,
        width=900,
        height=600
    )
    
    return fig


def create_comprehensive_eda_dashboard(eda_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create all visualizations for the EDA summary
    """
    visualizations = {}
    
    patterns = eda_summary.get('patterns_and_relationships', {})
    key_indicators = eda_summary.get('key_indicators', [])
    use_cases = eda_summary.get('use_cases', [])
    
    # Create correlation heatmap
    if patterns.get('correlations'):
        visualizations['correlation_heatmap'] = create_correlation_heatmap(patterns['correlations']).to_json()
    
    # Create key indicators chart
    if key_indicators:
        visualizations['key_indicators'] = create_key_indicators_bar(key_indicators).to_json()
    
    # Create patterns timeline
    if patterns.get('trends'):
        visualizations['trends'] = create_patterns_timeline(patterns['trends']).to_json()
    
    # Create outliers visualization
    if patterns.get('outliers'):
        visualizations['outliers'] = create_outliers_visualization(patterns['outliers']).to_json()
    
    # Create use cases visualization
    if use_cases:
        visualizations['use_cases'] = create_use_cases_visualization(use_cases).to_json()
    
    return visualizations
```

