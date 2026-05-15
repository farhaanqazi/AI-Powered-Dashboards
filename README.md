---
title: ML Dashboard Generator
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# ML Dashboard Generator

A FastAPI + React app that ingests a CSV (uploaded, URL-fetched, or pulled from Kaggle), runs a 4-layer analysis pipeline over it, and renders an interactive Plotly dashboard with auto-selected KPIs, charts, and exploratory insights.

The pipeline is **streaming**: large uploads emit phase-by-phase progress events over Server-Sent Events instead of hanging behind a frozen spinner.

## Highlights

- **Streaming upload** ([`POST /api/upload/stream`](main.py#L125)) — SSE-driven progress bar with phase labels (Reading → Profiling → Classifying → Relating → EDA → KPIs → Rendering → Done).
- **Adaptive charting**:
  - Scatter with >10k points falls back to a 2D-histogram density plot to prevent overplotting.
  - Time series with >10k rows auto-resamples on a span-aware rule (`W`/`D`/`H`/`5min`/`30s`).
  - Highly skewed numeric Y axes switch to log scale when `max/median > 100`.
- **4-layer analysis pipeline** ([src/analysis](src/analysis)):
  1. Syntactic profiler — dtypes, cardinalities, ranges.
  2. Semantic classifier — assigns roles (`identifier`, `categorical`, `numeric`, `datetime`, `text`).
  3. Relational analyzer — correlations and inter-column insights.
  4. Interpreter — picks meaningful KPIs and chart specs.
- **EDA module** ([`src/analysis/eda_analyzer.py`](src/analysis/eda_analyzer.py)) — significance-scored key indicators, outlier detection, use-case suggestions, recommendations.
- **React SPA** ([frontend/](frontend)) — Vite + Tailwind + react-plotly.js, with Overview / EDA / Visualizations / Columns tabs.

## Run it

### Local

```bash
# 1. Python backend
python -m venv venv
venv\Scripts\activate              # Windows  (use: source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

# 2. Frontend (one-time build OR dev server with HMR)
cd frontend
npm ci
npm run build                       # writes frontend/dist/ served by FastAPI
# -- or, for hot reload during development:
npm run dev                         # Vite dev server on :5173, proxies /api to :7860

# 3. Backend (serves the built SPA + API)
cd ..
python -m uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

Open <http://localhost:7860>.

### Docker

```bash
docker build -t ml-dashboard .
docker run -p 7860:7860 ml-dashboard
```

### Hugging Face Spaces

`main` branch on the `hf` remote auto-deploys via Docker SDK (see frontmatter at top of this file). Push with `git push hf main`.

## API

All JSON endpoints live under `/api`:

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/upload` | Blocking upload — returns final dashboard payload when the pipeline completes. |
| POST | `/api/upload/stream` | **Streaming upload (SSE).** Emits `{phase, message, percent}` events; final event includes `trace_id` and the full data payload. |
| POST | `/api/load_external` | Load a CSV by URL or Kaggle slug (`username/dataset`). |
| GET | `/api/dashboard` | Fetch the most-recent dashboard payload by `trace_id` (or `most_recent`). |

The SSE stream format is plain `data: {json}\n\n` frames. Example client (see [`frontend/src/services/api.js`](frontend/src/services/api.js)):

```js
await uploadFileStream(file, (evt) => {
  console.log(evt.phase, evt.percent, evt.message);
});
```

## Architecture

```
.
├── main.py                              FastAPI app, endpoints, SSE wiring
├── src/
│   ├── core/pipeline.py                 4-layer orchestrator + generator variant (for SSE)
│   ├── data/parser.py                   CSV/URL/Kaggle loaders + encoding detection
│   ├── analysis/
│   │   ├── layer_1_profiler.py          Syntactic profiling
│   │   ├── layer_2_classifier.py        Semantic role classification
│   │   ├── layer_3_relational.py        Correlation + cross-column analysis
│   │   ├── layer_4_interpreter.py       KPI determination + chart selection
│   │   ├── eda_analyzer.py              Patterns, outliers, indicators, use cases
│   │   └── data_structures.py           DashboardState + EnrichedProfile
│   ├── viz/
│   │   ├── plotly_renderer.py           Chart-spec → data dispatcher
│   │   ├── utils.py                     Per-chart builders (with downsampling/winsorization)
│   │   └── simple_renderer.py           Lightweight render path
│   └── diagnostics/tracer.py            Per-request pipeline tracing
└── frontend/
    ├── package.json                     React 18 + Vite + Tailwind + plotly.js-basic-dist
    └── src/
        ├── App.jsx                      Routes
        ├── dashboardStore.js            Zustand store
        ├── services/api.js              axios client + uploadFileStream (SSE)
        ├── components/
        │   ├── charts/ChartRenderer.jsx Type-aware Plotly wrapper (handles all backend shapes)
        │   ├── dashboard/               Overview / EDA / Visualizations / Columns tabs
        │   ├── upload/UploadPage.jsx    Upload UI with phase progress bar
        │   └── kpi/KPICard.jsx          KPI tile
        └── styles/                      Tailwind layers + design tokens
```

## Stack

- **Backend**: Python 3.9 · FastAPI 0.109 · pandas 2.2 · plotly 5.18 · scipy 1.12 · uvicorn (standard).
- **Frontend**: React 18 · Vite 4 · Tailwind CSS 3 · react-plotly.js (over plotly.js-basic-dist) · axios · zustand · react-router-dom.
- **Container**: Single-stage Docker on `python:3.9-slim` with Node 18 for the frontend build.

## Contributing

1. Fork, branch (`git checkout -b feat/short-description`).
2. Commit with a focused message (one logical change).
3. Push and open a PR. CI on the HF Space rebuilds on every push to `main`.

## License

MIT.
