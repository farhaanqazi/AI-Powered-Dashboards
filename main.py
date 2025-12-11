from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ValidationError
from jinja2 import Environment, FileSystemLoader
import pandas as pd
import io
import os
# Corrected import paths based on the modular structure
from src.core.pipeline import build_dashboard_from_file, build_dashboard_from_df
from src.data.parser import load_csv_from_url, load_csv_from_kaggle, LoadResult # Import LoadResult
from starlette.responses import RedirectResponse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models for Request Validation ---
class UploadFileRequest(BaseModel):
    dataset: UploadFile

class LoadExternalRequest(BaseModel):
    external_source: str

# --- FastAPI App Setup ---
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

# --- Exception Handlers ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Log the validation error details
    logger.error(f"Validation error: {exc}")
    # Return to the index page with an error message
    return templates.TemplateResponse("index.html", {
        "request": request,
        "error_message": f"Request validation failed: {exc.errors()[0]['msg'] if exc.errors() else 'Invalid input'}",
        "success": False
    })

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP error: {exc.detail}")
    # Return to the index page with an error message
    return templates.TemplateResponse("index.html", {
        "request": request,
        "error_message": f"HTTP Error: {exc.detail}",
        "success": False
    })

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unexpected error occurred: {exc}")
    # Return to the index page with a generic error message
    return templates.TemplateResponse("index.html", {
        "request": request,
        "error_message": "An unexpected error occurred. Please try again.",
        "success": False
    })

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Explicitly provide all necessary context variables
    return templates.TemplateResponse("index.html", {
        "request": request,
    })

@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, dataset: UploadFile = File(...)):
    try:
        # Validate file type
        if not dataset.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

        contents = await dataset.read()

        # Create a temporary in-memory file for processing
        file_stream = io.BytesIO(contents)
        file_stream.seek(0)

        # Extract the original filename
        original_filename = dataset.filename

        # Use the central pipeline to build everything
        # build_dashboard_from_file now expects a LoadResult internally, but we call it with the file stream
        # The pipeline will handle the LoadResult internally.
        state = build_dashboard_from_file(file_stream, original_filename=original_filename)

        if state is None:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error_message": "Failed to read CSV file or build dashboard. Please ensure the file is a valid CSV.",
                "success": False
            })

        # If state is returned successfully, pass data to dashboard template
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "profile": state.profile,
                "dataset_profile": state.dataset_profile,
                "kpis": state.kpis,
                "charts": state.charts,
                "primary_chart": state.primary_chart,
                "category_charts": state.category_charts,
                "all_charts": state.all_charts,
                "eda_summary": state.eda_summary,
                "original_filename": original_filename,
                "success": True
            }
        )
    except HTTPException:
        # Re-raise HTTP exceptions to be handled by the global handler
        raise
    except Exception as e:
        logger.exception(f"Error processing uploaded file: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error_message": f"An error occurred while processing the file: {str(e)}",
            "success": False
        })

@app.post("/load_external", response_class=HTMLResponse)
async def load_external(request: Request, external_source: str = Form(...)):
    """Load a dataset from external source (URL or Kaggle)"""
    try:
        # Validate external source format (basic check for URL or Kaggle slug)
        if external_source.startswith("http://") or external_source.startswith("https://"):
            # Basic URL validation could be more robust, e.g., using urllib.parse
            df_load_result = load_csv_from_url(external_source)
        else:
            # Assume it's a Kaggle slug
            df_load_result = load_csv_from_kaggle(external_source)

        # Check if loading was successful using the LoadResult object
        if not df_load_result.success:
            logger.error(f"Failed to load external dataset: {df_load_result.error_code} - {df_load_result.detail}")
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error_message": f"Failed to load dataset from external source: {df_load_result.detail or df_load_result.error_code}",
                "success": False
            })

        # If successful, df_load_result.df contains the DataFrame
        df = df_load_result.df
        if df is None:
             logger.error("Loaded DataFrame is None after successful LoadResult.")
             return templates.TemplateResponse("index.html", {
                "request": request,
                "error_message": "Failed to load dataset from external source (internal error).",
                "success": False
            })

        # Extract a name from the external source for the dashboard title
        if external_source.startswith("http://") or external_source.startswith("https://"):
            import urllib.parse
            parsed_path = urllib.parse.urlparse(external_source).path
            original_filename = parsed_path.split('/')[-1] or external_source
        else:
            # For Kaggle, use the slug or a processed version
            original_filename = external_source.split('/')[-1].replace('-', ' ').title() or "Kaggle Dataset"

        # Use the pipeline to build the dashboard from the loaded DataFrame
        state = build_dashboard_from_df(df)

        if state is None:
            logger.error("Failed to build dashboard from external dataset.")
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error_message": "Failed to build dashboard from the loaded dataset.",
                "success": False
            })

        # If state is returned successfully, pass data to dashboard template
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "profile": state.profile,
                "dataset_profile": state.dataset_profile,
                "kpis": state.kpis,
                "charts": state.charts,
                "primary_chart": state.primary_chart,
                "category_charts": state.category_charts,
                "all_charts": state.all_charts,
                "eda_summary": state.eda_summary,
                "original_filename": original_filename,
                "success": True
            }
        )
    except HTTPException:
        # Re-raise HTTP exceptions to be handled by the global handler
        raise
    except Exception as e:
        logger.exception(f"Error loading external dataset: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error_message": f"An error occurred while loading the external dataset: {str(e)}",
            "success": False
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)