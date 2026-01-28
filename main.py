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
import time
import uuid
import requests
from typing import Optional
from pathlib import Path
from datetime import datetime
import json
import os
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
    # Validate file type and sanitize filename
    if not dataset.filename:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error_message": "No file provided.", "success": False},
        )

    # Sanitize filename to prevent path traversal
    filename = dataset.filename
    if '..' in filename or filename.startswith('/'):
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error_message": "Invalid filename.", "success": False},
        )

    if not filename.lower().endswith(".csv"):
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
    # Sanitize and validate the external source
    if not external_source or not isinstance(external_source, str):
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error_message": "Invalid external source provided.", "success": False},
        )

    # Remove any potentially dangerous characters or patterns
    external_source = external_source.strip()

    # Validate URL format if it looks like a URL
    if external_source.startswith(("http://", "https://")):
        # Basic URL validation
        import re
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}'  # domain
            r'(?:/[^\s]*)?$'  # optional path
        )
        if not url_pattern.match(external_source):
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error_message": "Invalid URL format.", "success": False},
            )
        result = load_csv_from_url(external_source)
    else:
        # For Kaggle datasets, validate the format (username/dataset)
        if '/' not in external_source or len(external_source.split('/')) != 2:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error_message": "Invalid Kaggle dataset format. Expected 'username/dataset'.", "success": False},
            )
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

    # Validate file type and sanitize filename
    if not dataset.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    # Sanitize filename to prevent path traversal
    filename = dataset.filename
    if '..' in filename or filename.startswith('/'):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if not filename.lower().endswith(".csv"):
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

    # Sanitize and validate the external source
    if not req.external_source or not isinstance(req.external_source, str):
        raise HTTPException(status_code=400, detail="Invalid external source provided.")

    # Remove any potentially dangerous characters or patterns
    external_source = req.external_source.strip()

    # Validate URL format if it looks like a URL
    if external_source.startswith(("http://", "https://")):
        # Basic URL validation
        import re
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}'  # domain
            r'(?:/[^\s]*)?$'  # optional path
        )
        if not url_pattern.match(external_source):
            raise HTTPException(status_code=400, detail="Invalid URL format.")
        result = load_csv_from_url(external_source)
    else:
        # For Kaggle datasets, validate the format (username/dataset)
        if '/' not in external_source or len(external_source.split('/')) != 2:
            raise HTTPException(status_code=400, detail="Invalid Kaggle dataset format. Expected 'username/dataset'.")
        result = load_csv_from_kaggle(external_source)

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

import json
from fastapi.encoders import jsonable_encoder

def sanitize_for_json(obj):
    """Recursively neutralize Jinja2 Undefined, datetimes, sets, etc."""
    from jinja2 import Undefined
    if obj is None:
        return None
    if isinstance(obj, Undefined):
        return None
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(i) for i in obj]
    # Handle numpy/pandas types if present
    if hasattr(obj, 'item'):  # numpy scalars
        try:
            return obj.item()
        except:
            pass
    if str(type(obj)) == "<class 'pandas._libs.tslibs.timestamps.Timestamp'>":
        return str(obj)
    if str(type(obj)).startswith("<class 'numpy."):
        return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
    return obj

# Add the missing /api/dashboard endpoint
@app.get("/api/dashboard")
async def api_get_dashboard():
    try:
        # Get the most recently stored dashboard
        with storage_lock:
            dashboard_data = dashboard_storage.get('most_recent')

        if not dashboard_data:
            return {
                "status": "empty",
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {"hint": "Upload a dataset to generate insights"},
                "kpis": [],
                "charts": [],
                "eda": {},
                "errors": [],
                "message": "Dashboard initializing. Data will appear when pipeline completes.",
                # Return empty versions of all expected fields for React frontend
                "dataset_profile": {},
                "primary_chart": None,
                "category_charts": {},
                "all_charts": [],
                "original_filename": "",
                "critical_totals": {},
                "critical_full_dataset_aggregates": {},
                "eda_summary": {}
            }

        # Sanitize the data to prevent serialization issues
        clean = sanitize_for_json(dashboard_data)

        # Return the complete data structure that React frontend expects
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "columns": clean.get("dataset_profile", {}).get("n_cols", 0),
                "rows": clean.get("dataset_profile", {}).get("n_rows", 0),
                "filename": clean.get("original_filename", "")
            },
            "kpis": clean.get("kpis", []),
            "charts": clean.get("charts", []),
            "eda": clean.get("eda_summary", clean.get("all_charts", {})),
            "errors": clean.get("errors", []),
            "message": None,
            # Include all the fields that the React components expect
            "dataset_profile": clean.get("dataset_profile", {}),
            "primary_chart": clean.get("primary_chart", None),
            "category_charts": clean.get("category_charts", {}),
            "all_charts": clean.get("all_charts", []),
            "original_filename": clean.get("original_filename", ""),
            "critical_totals": clean.get("critical_totals", {}),
            "critical_full_dataset_aggregates": clean.get("critical_full_dataset_aggregates", {}),
            "eda_summary": clean.get("eda_summary", {})
        }

    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}", exc_info=True)
        # Return error state instead of throwing exception
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {"action": "Please refresh the page"},
            "kpis": [],
            "charts": [],
            "eda": {},
            "errors": ["System recovering"],
            "message": "Temporary dashboard issue. Please try uploading a dataset again.",
            # Return empty versions of all expected fields for React frontend
            "dataset_profile": {},
            "primary_chart": None,
            "category_charts": {},
            "all_charts": [],
            "original_filename": "",
            "critical_totals": {},
            "critical_full_dataset_aggregates": {},
            "eda_summary": {}
        }

# Add a root route to serve the React SPA
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse("frontend/dist/index.html")

# Catch-all route for React Router (for client-side routing)
# This should come after all other specific routes
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    return FileResponse("frontend/dist/index.html")



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

# --- START: New SPA Serving Logic with Cache Busting ---

# Serve the React frontend's static assets (js, css, images, etc.)
# This is the recommended approach for serving React SPAs with FastAPI
app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

# Add cache-busting middleware for frontend assets
@app.middleware("http")
async def add_cache_headers(request, call_next):
    response = await call_next(request)

    # Add cache control headers for frontend assets to prevent aggressive caching
    if request.url.path.startswith('/assets/'):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        response.headers["ETag"] = f"\"{int(time.time())}\""  # Add timestamp-based ETag

    # For the main HTML file and other HTML routes, also add cache control
    if request.url.path.endswith('.html') or request.url.path == '/':
        response.headers["Cache-Control"] = "no-cache, no-store, must revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        response.headers["ETag"] = f"\"{int(time.time())}\""  # Add timestamp-based ETag

    # For API responses, add cache control as well
    if request.url.path.startswith('/api/'):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    return response

# --- END: New SPA Serving Logic ---

# =========================================================
# ================== LOCAL RUN ============================
# =========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
