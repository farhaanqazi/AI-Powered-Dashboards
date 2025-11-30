
# ML Dashboard Generator

A Flask-based web application that generates **data profiling dashboards** from uploaded CSV files. The system intelligently analyzes datasets, computes statistics, identifies column roles, generates intelligent KPIs, and displays everything in an interactive HTML dashboard.

## 🚀 Features

### Data Profiling
- **Dataset Summary**: Total rows, total columns, type distribution
- **Column Profiling**: For each column - name, data type, missing values, unique values, role detection (numeric, datetime, categorical, text)
- **Numeric Statistics**: min, max, mean, std, sum
- **Top Categories**: Top 3 categories for categorical columns

### Intelligent KPIs
- Automatically identifies important columns based on data behavior
- Highlights numeric columns by variability metrics
- Emphasizes categorical columns by richness and distribution
- Identifies datetime columns for time-based analysis

### Interactive Dashboard
- Multiple chart types: category count, histograms, time series, category summaries
- Grid layout for efficient chart organization
- Clickable KPIs that highlight corresponding charts
- Column profiling table with detailed statistics

### Data Sources
- CSV file upload
- Direct CSV URLs
- Kaggle dataset integration

---

## 🏗️ Architecture

```
User uploads CSV → Flask Server → Core Pipeline → Multiple Data Processors → Dashboard State → Interactive Dashboard
```

### Core Components:
- `app.py` - Flask web server and routing
- `src/core/pipeline.py` - Main orchestration and dashboard building
- `src/data/` - Data parsing and analysis (parser.py, analyser.py)
- `src/ml/` - KPI generation and chart suggestions (kpi_generator.py, chart_selector.py)
- `src/viz/` - Chart rendering (plotly_renderer.py)
- `templates/` - Frontend templates (index.html, dashboard.html)

---

## 📁 Project Structure

```
ml-dashboard/
├── app.py                    # Flask application entry point
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
├── src/                      # Source code modules
│   ├── core/
│   │   └── pipeline.py       # Pipeline orchestration
│   ├── data/
│   │   ├── parser.py         # CSV parsing and loading
│   │   └── analyser.py       # Dataset analysis
│   ├── ml/
│   │   ├── kpi_generator.py  # KPI computation
│   │   └── chart_selector.py # Chart suggestions
│   └── viz/
│       └── plotly_renderer.py # Chart rendering
├── templates/
│   ├── index.html            # Upload interface
│   └── dashboard.html        # Dashboard display
└── static/                   # Static assets (CSS, JS)
```

---

## 🛠️ Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```

### 3. Access the Dashboard
Open `http://127.0.0.1:5000` in your browser

---

## 🧠 Data Models

### DatasetProfile
```python
{
    "n_rows": int,
    "n_cols": int,
    "role_counts": {
        "numeric": int,
        "datetime": int,
        "categorical": int,
        "text": int
    },
    "columns": [ColumnProfile]
}
```

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
    "type": str  # Added for enhanced KPIs
}
```

---

## 🔮 Future Scalability

### Planned Enhancements
- **Advanced Statistical KPIs**: Correlation coefficients, outlier detection, distribution metrics (skewness, kurtosis)
- **Semantic Column Understanding**: NLP-based column name analysis and pattern identification
- **Domain-specific KPI Generators**: Industry-specific metrics for financial, healthcare, e-commerce, etc.
- **Contextual KPIs**: Based on relationships between columns and temporal patterns
- **Statistical Significance Testing**: Tests for significant differences or correlations
- **Automated Insight Generation**: Pattern recognition for generating data insights
- **Advanced Chart Recommendations**: More intelligent chart suggestions based on data characteristics
- **Predictive KPIs**: Forecasting capabilities for time series data
- **Multi-dimensional KPIs**: Based on combinations of categorical columns
- **Anomaly Detection & Alerts**: Automatic flagging of unusual patterns

### Development Roadmap
1. **Enhanced Analytics**: Implement statistical and domain-specific KPIs
2. **Intelligent Visualizations**: Smarter chart type selection and layout
3. **Model Integration**: Potential for simple ML model integration for predictions
4. **Advanced UI**: Interactive filtering, drill-down capabilities, export options

---

## 📋 Development Log

*This section provides a chronological record of development activities. For the most recent updates, see the development log file.*

### Recent Milestones
- Initial dashboard framework with CSV upload and basic profiling
- Column role detection (numeric, datetime, categorical, text)
- Basic KPI generation system
- Chart suggestion engine
- Interactive dashboard with multiple chart types
- Grid layout for organized chart display
- Clickable KPI functionality
- External data source support (URLs and Kaggle)

---

## 📄 License

MIT License