from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, BackgroundTasks
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
from src import config

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
    # More detailed error message
    error_msg = f"An unexpected error occurred: {type(exc).__name__} - {str(exc)}"
    # Return to the index page with a detailed error message
    return templates.TemplateResponse("index.html", {
        "request": request,
        "error_message": error_msg,
        "success": False
    })

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Explicitly provide all necessary context variables
    return templates.TemplateResponse("index.html", {
        "request": request,
    })

from asyncio import TimeoutError as AsyncTimeoutError

def run_pipeline_and_get_state(file_stream, original_filename):
    return build_dashboard_from_file(file_stream, original_filename=original_filename)

@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, background_tasks: BackgroundTasks, dataset: UploadFile = File(...)):
    try:
        # Validate file type
        if not dataset.filename.lower().endswith('.csv'):
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error_message": "Only CSV files are allowed.",
                "success": False
            })

        contents = await dataset.read()

        # Check if file is empty
        if not contents:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error_message": "Uploaded file is empty.",
                "success": False
            })

        # Create a temporary in-memory file for processing
        file_stream = io.BytesIO(contents)
        file_stream.seek(0)

        # Extract the original filename
        original_filename = dataset.filename

        # Add the long-running task to the background
        state = build_dashboard_from_file(file_stream, original_filename=original_filename)

        if state is None:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error_message": "Failed to read CSV file or build dashboard. Please ensure the file is a valid CSV with proper structure.",
                "success": False
            })

        # Validate that all required data exists and is the correct type before passing to template
        # Ensure all data structures are simple, serializable Python types
        profile = state.profile if isinstance(state.profile, list) else []
        dataset_profile = state.dataset_profile if isinstance(state.dataset_profile, dict) else {}
        kpis = state.kpis if isinstance(state.kpis, list) else []
        charts = state.charts if isinstance(state.charts, list) else []
        primary_chart = state.primary_chart if state.primary_chart is not None else {}
        category_charts = state.category_charts if isinstance(state.category_charts, dict) else {}
        all_charts = state.all_charts if isinstance(state.all_charts, list) else []
        eda_summary = state.eda_summary if isinstance(state.eda_summary, dict) else {}
        original_filename = original_filename if original_filename else "Unknown"

        # Validate and sanitize titles to ensure they are proper strings
        def sanitize_chart_titles(chart_list):
            if not chart_list:
                return []
            for chart in chart_list:
                if 'title' in chart and chart['title']:
                    chart['title'] = str(chart['title']).replace('_', ' ').title()
            return chart_list

        # Apply sanitization to all chart title fields
        charts = sanitize_chart_titles(charts)
        all_charts = sanitize_chart_titles(all_charts)

        # If we have a primary chart, make sure its title is a proper string
        if primary_chart and 'title' in primary_chart:
            primary_chart['title'] = str(primary_chart['title']).replace('_', ' ').title()

        # If state is returned successfully, pass data to dashboard template
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "profile": profile,
                "dataset_profile": dataset_profile,
                "kpis": kpis,
                "charts": charts,
                "primary_chart": primary_chart,
                "category_charts": category_charts,
                "all_charts": all_charts,
                "eda_summary": eda_summary,
                "critical_aggregates": state.critical_aggregates if state and hasattr(state, 'critical_aggregates') else {},
                "original_filename": original_filename,
                "success": True
            }
        )
    except HTTPException as e:
        logger.error(f"HTTP error during upload: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error_message": f"HTTP error during upload: {e.detail}",
            "success": False
        })
    except UnicodeDecodeError as e:
        logger.error(f"File encoding error: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error_message": f"File encoding error - try saving your CSV file with UTF-8 encoding: {str(e)}",
            "success": False
        })
    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error_message": "CSV file is empty. Please provide a valid CSV file with data.",
            "success": False
        })
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error_message": f"Error parsing CSV file: {str(e)}. Please check your CSV file format.",
            "success": False
        })
    except Exception as e:
        logger.exception(f"Error processing uploaded file: {e}")
        error_type = type(e).__name__
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error_message": f"Error processing file: {error_type} - {str(e)}",
            "success": False
        })

@app.post("/load_external", response_class=HTMLResponse)
async def load_external(request: Request, background_tasks: BackgroundTasks, external_source: str = Form(...)):
    """Load a dataset from external source (URL or Kaggle)"""
    try:
        if not external_source:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error_message": "No external source provided. Please enter a URL or Kaggle dataset identifier.",
                "success": False
            })

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

        # Add the long-running task to the background
        state = build_dashboard_from_df(df)

        if state is None:
            logger.error("Failed to build dashboard from external dataset.")
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error_message": "Failed to build dashboard from the loaded dataset. The dataset may have structural issues or be incompatible.",
                "success": False
            })

        # Validate that all required data exists and is the correct type before passing to template
        # Ensure all data structures are simple, serializable Python types
        profile = state.profile if isinstance(state.profile, list) else []
        dataset_profile = state.dataset_profile if isinstance(state.dataset_profile, dict) else {}
        kpis = state.kpis if isinstance(state.kpis, list) else []
        charts = state.charts if isinstance(state.charts, list) else []
        primary_chart = state.primary_chart if state.primary_chart is not None else {}
        category_charts = state.category_charts if isinstance(state.category_charts, dict) else {}
        all_charts = state.all_charts if isinstance(state.all_charts, list) else []
        eda_summary = state.eda_summary if isinstance(state.eda_summary, dict) else {}
        original_filename = original_filename if original_filename else "Unknown"

        # Validate and sanitize titles to ensure they are proper strings
        def sanitize_chart_titles(chart_list):
            if not chart_list:
                return []
            for chart in chart_list:
                if 'title' in chart and chart['title']:
                    chart['title'] = str(chart['title']).replace('_', ' ').title()
            return chart_list

        # Apply sanitization to all chart title fields
        charts = sanitize_chart_titles(charts)
        all_charts = sanitize_chart_titles(all_charts)

        # If we have a primary chart, make sure its title is a proper string
        if primary_chart and 'title' in primary_chart:
            primary_chart['title'] = str(primary_chart['title']).replace('_', ' ').title()

        # If state is returned successfully, pass data to dashboard template
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "profile": profile,
                "dataset_profile": dataset_profile,
                "kpis": kpis,
                "charts": charts,
                "primary_chart": primary_chart,
                "category_charts": category_charts,
                "all_charts": all_charts,
                "eda_summary": eda_summary,
                "critical_aggregates": state.critical_aggregates if state and hasattr(state, 'critical_aggregates') else {},
                "original_filename": original_filename,
                "success": True
            }
        )
    except HTTPException as e:
        logger.error(f"HTTP error while loading external dataset: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error_message": f"HTTP error during external load: {e.detail}",
            "success": False
        })
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error while loading external dataset: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error_message": f"Network error loading external dataset: {str(e)}. Please check the URL or your internet connection.",
            "success": False
        })
    except Exception as e:
        logger.exception(f"Error loading external dataset: {e}")
        error_type = type(e).__name__
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error_message": f"Error loading external dataset: {error_type} - {str(e)}",
            "success": False
        })

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
