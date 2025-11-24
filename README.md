# ML Dashboard Project

An experimental project to build an **AI-powered dashboard generator**.

The idea:  
User uploads a structured dataset (e.g. CSV), and the app will eventually:
- Analyse the data
- Automatically generate KPIs
- Recommend suitable charts
- Render a dashboard layout

At the moment this is a **work in progress**.  
This README documents what has been done so far.

---

## ✅ Status – Day 1

What works right now:

- Project folder created at `dashboard_project/`
- Python virtual environment (`venv/`) set up
- Flask installed and configured
- Basic Flask app (`app.py`) running locally
- HTML template system using `templates/`
- `index.html` rendered as the homepage
- Working file upload form:
  - User can upload a `.csv` file
  - Backend receives the file and confirms upload

What does **not** exist yet (planned next steps):

- No real CSV parsing/analysis yet
- No KPIs or chart generation
- No machine learning models integrated
- No dashboard visualisation (charts) yet

---

## 🧱 Project Structure (current)

```text
dashboard_project/
├── app.py                # Flask app entry point
├── requirements.txt      # Python dependencies (Flask, pandas, etc.)
├── .gitignore            # Ignored files (venv, __pycache__, etc.)
├── venv/                 # Virtual environment (not committed ideally)
├── templates/
│   └── index.html        # Homepage with file upload form
└── src/
    └── data/             # (placeholder for future parser/analyzer modules)





## 🧱 System Architecture & Design (UML / Structural Plan)

This section defines the long-term architecture of the ML Dashboard Generator so functions and modules do not need to be renamed or rewritten as the project grows.

---

# 🥇 1. High-Level Architecture (Layers)

```
[1] Web Layer (Flask routes)
        |
        v
[2] Orchestration Layer (pipeline.py)
        |
        v
[3] Logic Layer (parser, analyser, kpi_generator, chart_selector, layout)
        |
        v
[4] Data Models (shared objects passed between all components)
```

---

# 🧩 2. Core Data Models (Stable Contracts)

These act as long-term structures for passing information between modules.

### 2.1 ColumnProfile
```python
class ColumnProfile:
    name: str
    dtype: str
    role: str            # numeric, categorical, datetime, text
    missing_count: int
    unique_count: int
    stats: NumericStats | None
```

### 2.2 NumericStats
```python
class NumericStats:
    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    sum: float | None
```

### 2.3 DatasetProfile
```python
class DatasetProfile:
    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]
```

### 2.4 KPI
```python
class KPI:
    id: str
    label: str
    value: float | str
    format: str | None
    explanation: str | None
```

### 2.5 ChartSpec
```python
class ChartSpec:
    id: str
    title: str
    chart_type: str       # line, bar, pie
    x_field: str
    y_field: str | None
    agg_func: str | None
```

### 2.6 DashboardLayout
```python
class DashboardLayout:
    kpi_order: list[str]
    chart_order: list[str]
```

### 2.7 DashboardState
```python
class DashboardState:
    profile: DatasetProfile
    kpis: list[KPI]
    charts: list[ChartSpec]
    layout: DashboardLayout
```

---

# 🧠 3. Module Design (Do Not Change Function Signatures)

### ✔ src/data/parser.py
```python
def load_csv(file_storage) -> pd.DataFrame:
    ...
```

### ✔ src/data/analyser.py
```python
def build_dataset_profile(df: pd.DataFrame, max_cols: int = 50) -> DatasetProfile:
    ...
```

### ✔ src/ml/kpi_generator.py
```python
def generate_kpis(df: pd.DataFrame,
                  profile: DatasetProfile) -> list[KPI]:
    ...
```

### ✔ src/ml/chart_selector.py
```python
def suggest_charts(df: pd.DataFrame,
                   profile: DatasetProfile,
                   kpis: list[KPI]) -> list[ChartSpec]:
    ...
```

### ✔ src/visualization/layout.py
```python
def build_layout(kpis: list[KPI],
                 charts: list[ChartSpec]) -> DashboardLayout:
    ...
```

---

# 🧵 4. Core Pipeline (Single Entry Point)

Create `src/core/pipeline.py`:

```python
from src.data.parser import load_csv
from src.data.analyser import build_dataset_profile
from src.ml.kpi_generator import generate_kpis
from src.ml.chart_selector import suggest_charts
from src.visualization.layout import build_layout

def build_dashboard_from_file(file_storage) -> DashboardState:
    df = load_csv(file_storage)
    profile = build_dataset_profile(df)
    kpis = generate_kpis(df, profile)
    charts = suggest_charts(df, profile, kpis)
    layout = build_layout(kpis, charts)
    return DashboardState(
        profile=profile,
        kpis=kpis,
        charts=charts,
        layout=layout
    )
```

This is the master function used by Flask.

---

# 🌐 5. Flask Layer (Thin Web Layer)

`app.py` should eventually look like:

```python
@app.route("/upload", methods=["POST"])
def upload():
    uploaded_file = request.files.get("dataset")
    if not uploaded_file:
        return "No file uploaded", 400

    state = build_dashboard_from_file(uploaded_file)

    return render_template(
        "dashboard.html",
        profile=state.profile,
        kpis=state.kpis,
        charts=state.charts,
        layout=state.layout
    )
```

---

# 🔄 6. Data Flow Diagram

```
User Uploads CSV
        |
        v
+-----------------------+
|  Flask (/upload)      |
+-----------+-----------+
            |
            v
+---------------------------+
| build_dashboard_from_file |
+----+-----------+----------+
     |           |
     v           v
+----------+   +-------------+
| parser   |   | analyser    |
+----------+   +-------------+
                    |
                    v
               +---------+
               | KPIs    |
               +---------+
                    |
                    v
               +---------+
               | Charts  |
               +---------+
                    |
                    v
             +----------------+
             | Layout Builder |
             +----------------+
                    |
                    v
             +----------------+
             | DashboardState |
             +----------------+
                    |
                    v
             HTML Templates
```

---

# 🎯 7. Why This Architecture Matters

- Prevents breaking code when adding new features  
- Ensures clean separation between components  
- Allows testing each module independently  
- Makes future ML integration easy  
- Keeps Flask lightweight  
- Enables scaling to React frontend later  

This is the long-term technical blueprint for the project.
