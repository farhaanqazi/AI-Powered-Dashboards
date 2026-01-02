from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from jinja2 import Environment, FileSystemLoader
import pandas as pd
import io
import logging
import uuid
import requests
from typing import Optional
from pathlib import Path
from datetime import datetime
import json
from threading import Lock

# ---- Internal imports ----
from src.core.pipeline import build_dashboard_from_file, build_dashboard_from_df
from src.data.parser import load_csv_from_url, load_csv_from_kaggle

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- DASHBOARD STATE STORAGE ----------------
dashboard_storage = {}
storage_lock = Lock()

# ---------------- FASTAPI APP ----------------
app = FastAPI()

# ---------------- JINJA SETUP (Diagnostics UI) ----------------
env = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=True
)

def fake_get_flashed_messages(*args, **kwargs):
    return []

env.globals["get_flashed_messages"] = fake_get_flashed_messages
templates = Jinja2Templates(env=env)

# ---------------- MODELS ----------------
class LoadExternalRequest(BaseModel):
    external_source: str

# ---------------- EXCEPTION HANDLERS ----------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(exc)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "error_message": "Request validation failed.",
            "success": False,
        },
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception(exc)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "error_message": f"{type(exc).__name__}: {exc}",
            "success": False,
        },
    )

# =========================================================
# ================= JINJA DIAGNOSTIC UI ===================
# =========================================================

@app.get("/diagnostic-ui", response_class=HTMLResponse)
async def diagnostic_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, dataset: UploadFile = File(...)):
    if not dataset.filename.lower().endswith(".csv"):
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error_message": "Only CSV files allowed.", "success": False},
        )

    contents = await dataset.read()
    file_stream = io.BytesIO(contents)
    state = build_dashboard_from_file(file_stream, original_filename=dataset.filename)

    if not state:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error_message": "Processing failed.", "success": False},
        )

    # Store the dashboard state in memory for API access
    dashboard_data = {
        "dataset_profile": state.dataset_profile,
        "kpis": state.kpis,
        "charts": state.charts,
        "primary_chart": state.primary_chart,
        "category_charts": getattr(state, "category_charts", {}),
        "all_charts": state.all_charts,
        "original_filename": dataset.filename,
        "errors": getattr(state, "errors", []),
        "critical_totals": getattr(state, "critical_totals", {}),
        "critical_full_dataset_aggregates": getattr(state, "critical_full_dataset_aggregates", {}),
        "eda_summary": getattr(state, "eda_summary", {})
    }

    # Store as the most recent for API access
    with storage_lock:
        dashboard_storage['most_recent'] = dashboard_data

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "dataset_profile": state.dataset_profile,
            "kpis": state.kpis,
            "charts": state.charts,
            "primary_chart": state.primary_chart,
            "all_charts": state.all_charts,
            "original_filename": dataset.filename,
            "errors": getattr(state, "errors", []),
            "critical_totals": getattr(state, "critical_totals", {}),
            "critical_full_dataset_aggregates": getattr(state, "critical_full_dataset_aggregates", {}),
            "eda_summary": getattr(state, "eda_summary", {}),
            "success": True,
        },
    )

@app.post("/load_external", response_class=HTMLResponse)
async def load_external(request: Request, external_source: str = Form(...)):
    if external_source.startswith("http"):
        result = load_csv_from_url(external_source)
    else:
        result = load_csv_from_kaggle(external_source)

    if not result.success or result.df is None:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error_message": "Failed to load dataset.", "success": False},
        )

    state = build_dashboard_from_df(result.df)

    # Store the dashboard state in memory for API access
    dashboard_data = {
        "dataset_profile": state.dataset_profile,
        "kpis": state.kpis,
        "charts": state.charts,
        "primary_chart": state.primary_chart,
        "category_charts": getattr(state, "category_charts", {}),
        "all_charts": state.all_charts,
        "original_filename": external_source,
        "errors": getattr(state, "errors", []),
        "critical_totals": getattr(state, "critical_totals", {}),
        "critical_full_dataset_aggregates": getattr(state, "critical_full_dataset_aggregates", {}),
        "eda_summary": getattr(state, "eda_summary", {})
    }

    # Store as the most recent for API access
    with storage_lock:
        dashboard_storage['most_recent'] = dashboard_data

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "dataset_profile": state.dataset_profile,
            "kpis": state.kpis,
            "charts": state.charts,
            "primary_chart": state.primary_chart,
            "all_charts": state.all_charts,
            "original_filename": external_source,
            "errors": getattr(state, "errors", []),
            "critical_totals": getattr(state, "critical_totals", {}),
            "critical_full_dataset_aggregates": getattr(state, "critical_full_dataset_aggregates", {}),
            "eda_summary": getattr(state, "eda_summary", {}),
            "success": True,
        },
    )

# =========================================================
# ======================= API ==============================
# =========================================================

@app.post("/api/upload")
async def api_upload(dataset: UploadFile = File(...)):
    trace_id = str(uuid.uuid4())

    if not dataset.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed.")

    contents = await dataset.read()
    file_stream = io.BytesIO(contents)

    state = build_dashboard_from_file(file_stream, original_filename=dataset.filename)
    if not state:
        raise HTTPException(status_code=500, detail="Dashboard build failed.")

    response_data = {
        "dataset_profile": state.dataset_profile,
        "kpis": state.kpis,
        "charts": state.charts,
        "primary_chart": state.primary_chart,
        "category_charts": getattr(state, "category_charts", {}),
        "all_charts": state.all_charts,
        "original_filename": dataset.filename,
        "errors": getattr(state, "errors", []),
        "critical_totals": getattr(state, "critical_totals", {}),
        "critical_full_dataset_aggregates": getattr(state, "critical_full_dataset_aggregates", {}),
        "eda_summary": getattr(state, "eda_summary", {})
    }

    # Store the dashboard state for later retrieval using the trace_id
    with storage_lock:
        dashboard_storage[trace_id] = response_data

    # Also store as the most recent for simple access
    with storage_lock:
        dashboard_storage['most_recent'] = response_data

    return {
        "status": "success",
        "trace_id": trace_id,
        "data": response_data,
    }

@app.post("/api/load_external")
async def api_load_external(req: LoadExternalRequest):
    trace_id = str(uuid.uuid4())

    if req.external_source.startswith("http"):
        result = load_csv_from_url(req.external_source)
    else:
        result = load_csv_from_kaggle(req.external_source)

    if not result.success or result.df is None:
        raise HTTPException(status_code=400, detail="Failed to load dataset.")

    state = build_dashboard_from_df(result.df)

    response_data = {
        "dataset_profile": state.dataset_profile,
        "kpis": state.kpis,
        "charts": state.charts,
        "primary_chart": state.primary_chart,
        "category_charts": getattr(state, "category_charts", {}),
        "all_charts": state.all_charts,
        "original_filename": req.external_source,
        "errors": getattr(state, "errors", []),
        "critical_totals": getattr(state, "critical_totals", {}),
        "critical_full_dataset_aggregates": getattr(state, "critical_full_dataset_aggregates", {}),
        "eda_summary": getattr(state, "eda_summary", {})
    }

    # Store the dashboard state for later retrieval using the trace_id
    with storage_lock:
        dashboard_storage[trace_id] = response_data

    # Also store as the most recent for simple access
    with storage_lock:
        dashboard_storage['most_recent'] = response_data

    return {
        "status": "success",
        "trace_id": trace_id,
        "data": response_data,
    }

# Add the missing /api/dashboard endpoint
@app.get("/api/dashboard")
async def api_get_dashboard():
    # Get the most recently stored dashboard
    with storage_lock:
        dashboard_data = dashboard_storage.get('most_recent')

    if not dashboard_data:
        raise HTTPException(status_code=404, detail="No dashboard data available")

    # Return the dashboard data directly to match frontend expectations
    return dashboard_data



# =========================================================
# ================== PERSISTENCE TEST =====================
# =========================================================

TEST_FILE = Path("persistence_test.txt")

@app.get("/test-persistence/{action}", response_class=PlainTextResponse)
async def test_persistence(action: str):
    if action == "write":
        content = f"Written at {datetime.utcnow().isoformat()}"
        TEST_FILE.write_text(content)
        return content
    elif action == "read":
        return TEST_FILE.read_text() if TEST_FILE.exists() else "File missing"
    return "Invalid action"

# =========================================================
# ================= STATIC + REACT ========================
# =========================================================

# Legacy static
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- START: New SPA Serving Logic ---

# Mount the React frontend's static assets (js, css, images, etc.)
# This MUST come before the root route
app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="react-assets")

# Serve the index.html for any path that is not an API route or a known file
@app.get("/{full_path:path}", response_class=HTMLResponse)
async def serve_react_app(request: Request, full_path: str):
    # This catch-all route ensures that client-side routing in React works correctly.
    # Any route not matched by your API or other static mounts will serve the React app.
    # FastAPI will match more specific routes (like /api/upload) before this catch-all,
    # so this should only serve the React app for frontend routes like /dashboard, /settings, etc.
    # Exclude API routes and asset files from this catch-all
    if full_path.startswith('api/') or full_path.startswith('assets/') or full_path.endswith(('.js', '.css', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico')):
        # This should be handled by the static mount or API routes, so return 404 if we reach here
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse("frontend/dist/index.html")

# --- END: New SPA Serving Logic ---

# =========================================================
# ================== LOCAL RUN ============================
# =========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
