from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, FileResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
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

from dotenv import load_dotenv
load_dotenv()

# ---- Internal imports ----
from src.core.pipeline import (
    build_dashboard_from_file,
    build_dashboard_from_df,
    build_dashboard_from_file_generator,
)
from src.data.parser import load_csv_from_url, load_csv_from_kaggle
from src.auth import require_clerk_user, allow_clerk_or_guest
from src.api.schemas import (
    UploadResponse,
    LoadExternalResponse,
    ValidateExternalResponse,
    DashboardResponse,
)
from src.observability.request_id import RequestIDMiddleware
from src.observability.health import build_router as build_health_router
from src.observability.metrics import MetricsMiddleware, build_router as build_metrics_router
from src.observability.tracing import configure_tracing
from src.observability.sentry import configure_sentry

from src.persistence.repository import get_repository

# ---------------- LOGGING ----------------
from src.observability.logging import configure_observability_logging
configure_observability_logging()

if os.environ.get("LOG_FILE_HANDLERS", "false").lower() == "true":
    try:
        from src.logger import configure_logging
        configure_logging()
    except Exception:
        pass

logger = logging.getLogger(__name__)

# ---------------- DASHBOARD STATE STORAGE ----------------
dashboard_storage = {}
storage_lock = Lock()

# ---------------- FASTAPI APP ----------------
# Sentry must init before the app is created so its Starlette/FastAPI
# integrations hook correctly.
configure_sentry()
app = FastAPI()
configure_tracing(app)

from src.config import CORS_ALLOW_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# Middleware execution order is LIFO (last added = outermost). Final order:
# RequestID (outermost) -> Metrics -> CORS (innermost). So add order must be
# CORS, then Metrics, then RequestID.
app.add_middleware(MetricsMiddleware)

app.add_middleware(RequestIDMiddleware)

# ---------------- OBSERVABILITY ROUTES ----------------
# Must be registered before the SPA catch-all (`/{full_path:path}`) so the
# catch-all does not intercept /healthz, /readyz and /metrics.
app.include_router(build_health_router())
app.include_router(build_metrics_router())

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

@app.post("/api/upload", response_model=UploadResponse)
async def api_upload(dataset: UploadFile = File(...), encoding: Optional[str] = Form(None), user=Depends(allow_clerk_or_guest)):
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
        state = build_dashboard_from_file(file_stream, original_filename=dataset.filename, encoding=encoding)
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
        "warnings": getattr(state, "warnings", []),
        "critical_totals": getattr(state, "critical_totals", {}),
        "critical_full_dataset_aggregates": getattr(state, "critical_full_dataset_aggregates", {}),
        "eda_summary": getattr(state, "eda_summary", {})
    }

    get_repository().save(
        user["session_key"], trace_id=trace_id, payload=response_data
    )

    return {
        "status": "success",
        "trace_id": trace_id,
        "data": response_data,
    }

@app.post("/api/upload/stream")
async def api_upload_stream(dataset: UploadFile = File(...), encoding: Optional[str] = Form(None), user=Depends(allow_clerk_or_guest)):
    if not dataset.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    filename = dataset.filename
    if '..' in filename or filename.startswith('/'):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed.")

    contents = await dataset.read()
    trace_id = str(uuid.uuid4())

    def event_source():
        file_stream = io.BytesIO(contents)
        try:
            for event in build_dashboard_from_file_generator(
                file_stream, original_filename=filename, encoding=encoding
            ):
                phase = event.get("phase")
                if phase == "done":
                    state = event.get("state")
                    if state is None:
                        payload = {"phase": "error", "message": "Pipeline returned no state.", "percent": 100}
                        yield f"data: {json.dumps(payload)}\n\n"
                        return
                    response_data = {
                        "dataset_profile": state.dataset_profile,
                        "kpis": state.kpis,
                        "charts": state.charts,
                        "primary_chart": state.primary_chart,
                        "category_charts": getattr(state, "category_charts", {}),
                        "all_charts": state.all_charts,
                        "original_filename": filename,
                        "errors": getattr(state, "errors", []),
                        "warnings": getattr(state, "warnings", []),
                        "critical_totals": getattr(state, "critical_totals", {}),
                        "critical_full_dataset_aggregates": getattr(state, "critical_full_dataset_aggregates", {}),
                        "eda_summary": getattr(state, "eda_summary", {}),
                    }
                    get_repository().save(
                        user["session_key"], trace_id=trace_id, payload=response_data
                    )
                    final = {
                        "phase": "done",
                        "message": event.get("message", "Complete"),
                        "percent": 100,
                        "trace_id": trace_id,
                        "data": response_data,
                    }
                    yield f"data: {json.dumps(final)}\n\n"
                    return
                else:
                    yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.exception("Upload stream pipeline failed")
            err = {"phase": "error", "message": f"Server error: {e}", "percent": 100}
            yield f"data: {json.dumps(err)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/validate_external", response_model=ValidateExternalResponse)
async def api_validate_external(req: LoadExternalRequest, user=Depends(allow_clerk_or_guest)):
    """Cheap pre-flight: confirm the source is reachable and looks like CSV before
    the user is sent to /processing. Returns 200 on success, 400 with a specific
    detail on failure."""
    if not req.external_source or not isinstance(req.external_source, str):
        raise HTTPException(status_code=400, detail="Please enter a URL or Kaggle dataset identifier.")
    src = req.external_source.strip()
    if not src:
        raise HTTPException(status_code=400, detail="Please enter a URL or Kaggle dataset identifier.")

    if src.startswith(("http://", "https://")):
        url_pattern = re.compile(r'^https?://(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s]*)?$')
        if not url_pattern.match(src):
            raise HTTPException(status_code=400, detail="That doesn't look like a valid URL.")
        if 'kaggle.com/datasets/' in src.lower():
            slug = src.lower().split('kaggle.com/datasets/', 1)[1].rstrip('/').split('?')[0]
            suggested = slug if slug.count('/') == 1 else slug.rsplit('/', 1)[0] if '/' in slug else slug
            raise HTTPException(
                status_code=400,
                detail=(
                    f"That's a Kaggle dataset page URL, not a CSV download link. "
                    f"Use the dataset identifier directly — try entering: '{suggested}'"
                ),
            )
        try:
            r = requests.get(src, stream=True, timeout=10, allow_redirects=True)
        except requests.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Couldn't reach that URL: {e}")
        try:
            if r.status_code >= 400:
                raise HTTPException(status_code=400, detail=f"URL returned HTTP {r.status_code}.")
            ctype = (r.headers.get('Content-Type') or '').lower()
            if any(t in ctype for t in ('text/html', 'application/xhtml', 'application/xml', 'text/xml')):
                raise HTTPException(
                    status_code=400,
                    detail=f"URL returned '{ctype.split(';')[0]}', not a CSV. Make sure it points to a raw CSV file.",
                )
            chunk = b''
            for piece in r.iter_content(chunk_size=4096):
                chunk += piece
                if len(chunk) >= 4096:
                    break
            text = chunk[:4096].decode('utf-8', errors='replace').lstrip()
            if text.startswith(('<!DOCTYPE', '<html', '<HTML', '<?xml')):
                raise HTTPException(
                    status_code=400,
                    detail="URL returns HTML, not CSV. Make sure the link points to a raw CSV file (not the page that hosts it).",
                )
        finally:
            r.close()
        return {"ok": True}

    # Kaggle slug path
    parts = src.split('/')
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise HTTPException(
            status_code=400,
            detail="Kaggle dataset must be in the format 'username/dataset' (e.g., 'rtatman/188-million-us-wildfires').",
        )
    return {"ok": True}


@app.post("/api/load_external", response_model=LoadExternalResponse)
async def api_load_external(req: LoadExternalRequest, user=Depends(allow_clerk_or_guest)):
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

    if state is not None:
        state.warnings = result.warnings or []

    response_data = {
        "dataset_profile": state.dataset_profile,
        "kpis": state.kpis,
        "charts": state.charts,
        "primary_chart": state.primary_chart,
        "category_charts": getattr(state, "category_charts", {}),
        "all_charts": state.all_charts,
        "original_filename": req.external_source,
        "errors": getattr(state, "errors", []),
        "warnings": getattr(state, "warnings", []),
        "critical_totals": getattr(state, "critical_totals", {}),
        "critical_full_dataset_aggregates": getattr(state, "critical_full_dataset_aggregates", {}),
        "eda_summary": getattr(state, "eda_summary", {})
    }

    with storage_lock:
        dashboard_storage[trace_id] = response_data
        dashboard_storage[user['session_key']] = response_data

    return {
        "status": "success",
        "trace_id": trace_id,
        "data": response_data,
    }

@app.get("/api/dashboard", response_model=DashboardResponse)
async def api_get_dashboard(user=Depends(allow_clerk_or_guest)):
    with storage_lock:
        dashboard_data = dashboard_storage.get(user['session_key'])

    if not dashboard_data:
        return {
            "status": "empty",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {"hint": "Upload a dataset to generate insights"},
            "kpis": [],
            "charts": [],
            "eda": {},
            "errors": [],
            "warnings": [],
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
        "warnings": dashboard_data.get("warnings", []),
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

@app.get("/assets/{full_path:path}")
async def serve_dynamic_assets(full_path: str):
    filepath = f"frontend/dist/assets/{full_path}"
    if os.path.exists(filepath):
        return FileResponse(filepath)
    raise HTTPException(status_code=404, detail="Asset not found")

# Serve React SPA (handles all frontend routing)
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse("frontend/dist/index.html")

# Serve splash page separately
@app.get("/splash", response_class=HTMLResponse)
async def read_splash():
    return FileResponse("frontend/dist/splash.html")

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

@app.get("/vite.svg")
async def serve_favicon():
    # Return a simple transparent 1x1 pixel as default favicon to avoid 404 errors
    from fastapi.responses import Response
    # Simple SVG favicon
    svg_content = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16"></svg>'
    return Response(content=svg_content, media_type="image/svg+xml")

@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    # Check if it's an API request that should not be handled by SPA
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404)
    # Check if it's an asset request that should not be handled by SPA
    if full_path.startswith("assets/"):
        raise HTTPException(status_code=404)
    # For all other non-API, non-asset paths (SPA routing), serve the main index.html
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
