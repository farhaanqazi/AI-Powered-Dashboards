# AI Session Context: ML Dashboard Generator Diagnostic Session

## Project Overview
I'm working with an AI-powered, FastAPI-based dashboard generator that automatically analyzes CSV datasets and creates interactive visualizations. The system features a sophisticated 4-layer analysis engine (profiling, classification, relational analysis, interpretation) and supports both legacy Jinja2 templates and modern React frontend implementations.

## Current Diagnostic Focus
We're systematically diagnosing and resolving bugs that cause the pipeline to fail or produce incorrect results for specific datasets. The diagnostic approach involves analyzing trace logs from both passing and failing dataset runs to identify root causes and implement precise fixes.

## Previously Resolved Issues
- State drift/contract validation failures for datasets with >50 columns
- Inconsistent profile state (n_cols count mismatch)
- Incorrect semantic aggregation (improper KPI generation for non-aggregatable metrics)
- Wide time-series misinterpretation (automatic reshaping of year-column datasets)
- Unresponsive chart rendering (chart resizing issues)

## Technical Architecture
- **Backend**: FastAPI with 4-layer analysis engine (src/analysis/*)
- **Frontend**: Dual implementation (legacy Jinja2 dashboard.html + modern React DashboardPage.jsx)
- **Diagnostics**: Pipeline tracer (src/diagnostics/tracer.py) for detailed execution logging
- **Visualization**: Plotly-based rendering with responsive design
- **Data Processing**: Pandas-based with sampling, reshaping, and semantic analysis

## Current State
The tracer.py diagnostic tool is available for pipeline analysis, logging detailed execution metadata when enabled. Both dashboard implementations exist but may have redundancy. The React frontend fetches data via API calls, while the legacy Jinja2 template uses server-side rendering.

## Next Steps
Awaiting new debug trace from failing dataset to analyze and identify new bug patterns or variations of previously addressed issues, then implement appropriate fixes to ensure robust pipeline performance across all dataset types.