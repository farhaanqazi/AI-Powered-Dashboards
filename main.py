from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
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
import re

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

# ---------------- MODELS ----------------
class LoadExternalRequest(BaseModel):
    external_source: str

# ---------------- EXCEPTION HANDLERS ----------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Request validation failed: {exc.errors()}", exc_info=True)
    return JSONResponse(
        status_code=422,
        content={
            "message": "Request validation failed.",
            "errors": exc.errors()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "message": "An internal server error occurred.",
            "error_type": type(exc).__name__,
            "error_detail": str(exc)
        }
    )

# =========================================================
# ======================= API ==============================

@app.post("/api/upload")
async def api_upload(dataset: UploadFile = File(...)):
    trace_id = str(uuid.uuid4())

    if not dataset.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    filename = dataset.filename
    if '..' in filename or filename.startswith('/'):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed.")

    contents = await dataset.read()
    file_stream = io.BytesIO(contents)

    try:
        state = build_dashboard_from_file(file_stream, original_filename=dataset.filename)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Dashboard build failed: {e}")

    if not state:
        raise HTTPException(status_code=500, detail="Dashboard build failed: returned no state.")

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

    with storage_lock:
        dashboard_storage[trace_id] = response_data
        dashboard_storage['most_recent'] = response_data

    return {
        "status": "success",
        "trace_id": trace_id,
        "data": response_data,
    }

@app.post("/api/load_external")
async def api_load_external(req: LoadExternalRequest):
    trace_id = str(uuid.uuid4())

    if not req.external_source or not isinstance(req.external_source, str):
        raise HTTPException(status_code=400, detail="Invalid external source provided.")

    external_source = req.external_source.strip()

    if external_source.startswith(("http://", "https://")):
        url_pattern = re.compile(
            r'^https?://'  
            r'(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}'  
            r'(?:/[^\s]*)?$'  
        )
        if not url_pattern.match(external_source):
            raise HTTPException(status_code=400, detail="Invalid URL format.")
        result = load_csv_from_url(external_source)
    else:
        if '/' not in external_source or len(external_source.split('/')) != 2:
            raise HTTPException(status_code=400, detail="Invalid Kaggle dataset format. Expected 'username/dataset'.")
        result = load_csv_from_kaggle(external_source)

    if not result.success or result.df is None:
        raise HTTPException(status_code=400, detail="Failed to load dataset.")

    try:
        state = build_dashboard_from_df(result.df)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Dashboard build failed: {e}")

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

    with storage_lock:
        dashboard_storage[trace_id] = response_data
        dashboard_storage['most_recent'] = response_data

    return {
        "status": "success",
        "trace_id": trace_id,
        "data": response_data,
    }

@app.get("/api/dashboard")
async def api_get_dashboard():
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
            "dataset_profile": {},
            "primary_chart": None,
            "category_charts": {},
            "all_charts": [],
            "original_filename": "",
            "critical_totals": {},
            "critical_full_dataset_aggregates": {},
            "eda_summary": {}
        }

    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
        "metadata": {
            "columns": dashboard_data.get("dataset_profile", {}).get("n_cols", 0),
            "rows": dashboard_data.get("dataset_profile", {}).get("n_rows", 0),
            "filename": dashboard_data.get("original_filename", "")
        },
        "kpis": dashboard_data.get("kpis", []),
        "charts": dashboard_data.get("charts", []),
        "eda": dashboard_data.get("eda_summary", {}),
        "errors": dashboard_data.get("errors", []),
        "message": None,
        "dataset_profile": dashboard_data.get("dataset_profile", {}),
        "primary_chart": dashboard_data.get("primary_chart", None),
        "category_charts": dashboard_data.get("category_charts", {}),
        "all_charts": dashboard_data.get("all_charts", []),
        "original_filename": dashboard_data.get("original_filename", ""),
        "critical_totals": dashboard_data.get("critical_totals", {}),
        "critical_full_dataset_aggregates": dashboard_data.get("critical_full_dataset_aggregates", {}),
        "eda_summary": dashboard_data.get("eda_summary", {})
    }

# Serve React SPA
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse("frontend/dist/index.html")

@app.get("/debug-build-files")
async def debug_build_files():
    dist_path = "frontend/dist"
    assets_path = "frontend/dist/assets"

    result = []
    if os.path.exists(dist_path):
        result.append("Files in frontend/dist:")
        result.extend([f"  - {f}" for f in os.listdir(dist_path)])
    else:
        result.append("frontend/dist DOES NOT EXIST")

    if os.path.exists(assets_path):
        result.append("\nFiles in frontend/dist/assets:")
        result.extend([f"  - {f}" for f in os.listdir(assets_path)])
        for subdir in os.listdir(assets_path):
            subdir_path = os.path.join(assets_path, subdir)
            if os.path.isdir(subdir_path):
                result.append(f"\n  Contents of {subdir}/:")
                for item in os.listdir(subdir_path):
                    result.append(f"    - {item}")
    else:
        result.append("frontend/dist/assets DOES NOT EXIST")

    return PlainTextResponse("\n".join(result))

@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    if full_path.startswith("assets/"):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse("frontend/dist/index.html")

# Test persistence
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

# Serve React static assets
app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

# Cache-busting middleware
@app.middleware("http")
async def add_cache_headers(request, call_next):
    response = await call_next(request)

    if request.url.path.startswith('/assets/'):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        response.headers["ETag"] = f"\"{int(time.time())}\""

    if request.url.path.endswith('.html') or request.url.path == '/':
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        response.headers["ETag"] = f"\"{int(time.time())}\""

    if request.url.path.startswith('/api/'):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    return response

# Local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
