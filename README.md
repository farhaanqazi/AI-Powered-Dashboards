
# ML Dashboard Project

A Flask-based web application that generates a **data profiling dashboard** from an uploaded CSV file.  
The system analyzes the dataset, computes statistics, identifies column roles, generates KPIs, and displays everything in a clean HTML dashboard.

---

## 🚀 Features

### ✓ CSV Upload
Upload CSV files directly through the web interface.

### ✓ Dataset Summary
- Total rows  
- Total columns  

### ✓ Column Profiling
For each column:
- Name  
- Data type  
- Missing values  
- Unique values  
- Role detection:
  - numeric  
  - datetime  
  - categorical  
  - text  
- Numeric statistics (min, max, mean, std, sum)  
- Top 3 categories (categorical columns only)  

### ✓ KPIs
Automatically computed:
- Numeric columns count  
- Datetime columns count  
- Categorical columns count  
- Text columns count  

### ✓ Dashboard Rendering
The app displays:
1. Dataset Summary  
2. KPIs  
3. Column Profiling Table  

---

## 📐 Project Design Overview

```

```
            +----------------------+
            |      User Uploads    |
            |        CSV File      |
            +----------+-----------+
                       |
                       v
            +----------------------+
            |  Flask Web Server    |
            |     (/upload)        |
            +----------+-----------+
                       |
                       v
            +----------------------+
            |  Core Pipeline       |
            | build_dashboard...   |
            +----------+-----------+
                       |
    -------------------------------------------------
    |                       |                       |
    v                       v                       v
```

+---------------+      +----------------+      +------------------+
|  CSV Parser   |      | Dataset Profiler |    |   KPI Generator  |
| load_csv()    |      | build_dataset... |    | generate_basic...|
+---------------+      +----------------+      +------------------+
|
v
+----------------------+
|   Dashboard State    |
| (profiles + KPIs)    |
+----------+-----------+
|
v
+----------------------+
|   dashboard.html     |
+----------------------+

```

---

## 📁 Project Structure

```

ML-dashboard-project/
├── app.py
├── requirements.txt
├── README.md
├── PROJECT_OVERVIEW.md
├── templates/
│   ├── index.html
│   └── dashboard.html
├── src/
│   ├── core/
│   │   └── pipeline.py
│   ├── data/
│   │   ├── parser.py
│   │   └── analyser.py
│   ├── ml/
│   │   └── kpi_generator.py
│   └── visualization/
│       └── layout.py
└── static/
├── css/
└── js/

````

---

## 🧠 Data Models

### DatasetProfile
```python
{
    "n_rows": int,
    "n_cols": int,
    "columns": [ColumnProfile]
}
````

### ColumnProfile

```python
{
    "name": str,
    "dtype": str,
    "role": "numeric" | "datetime" | "categorical" | "text",
    "missing_count": int,
    "unique_count": int,
    "stats": {
        "min": float | str,
        "max": float | str,
        "mean": float | None,
        "std": float | None,
        "sum": float | None
    } | None,
    "top_categories": [
        {"value": str, "count": int}
    ]
}
```

### KPI

```python
{
    "label": str,
    "value": float | str,
    "format": "integer" | None
}
```

---

## 🔧 Pipeline Flow

```
Upload CSV →
load_csv() →
build_dataset_profile() →
generate_basic_kpis() →
Render dashboard.html
```

Pipeline entry point:

```
src/core/pipeline.py
```

---

## 🛠 Running the App

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Start Flask

```
python app.py
```

### 3. Open browser

```
http://127.0.0.1:5000
```

Upload your CSV → dashboard appears.

---

## 🔮 Next Steps

* Improve text vs categorical detection
* Add chart suggestions
* Render charts using Chart.js
* Build auto-layout engine
* Add ML-based KPI and chart recommendation
* Prepare deployment-ready backend/API

---

## 📜 License

MIT

```

---