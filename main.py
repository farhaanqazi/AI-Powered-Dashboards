from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import io
import os
from src.core.pipeline import build_dashboard_from_file, build_dashboard_from_df
from src.data.parser import load_csv_from_url, load_csv_from_kaggle
from starlette.responses import RedirectResponse

app = FastAPI()

# Set up templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, dataset: UploadFile = File(...)):
    try:
        contents = await dataset.read()
        
        # Create a temporary in-memory file for processing
        file_stream = io.BytesIO(contents)
        file_stream.seek(0)
        
        # Use the central pipeline to build everything
        state = build_dashboard_from_file(file_stream)

        if state is None:
            return templates.TemplateResponse("index.html", {
                "request": request, 
                "error": "Failed to read CSV file. Please ensure the file is a valid CSV."
            })

        df = state.df
        dataset_profile = state.dataset_profile
        profile = state.profile
        kpis = state.kpis
        charts = state.charts
        primary_chart = state.primary_chart
        category_charts = state.category_charts
        all_charts = state.all_charts

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
            }
        )
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": f"An error occurred while processing the file: {str(e)}"
        })

@app.post("/load_external", response_class=HTMLResponse)
async def load_external(request: Request, external_source: str = Form(...)):
    """Load a dataset from external source (URL or Kaggle)"""
    try:
        # Decide how to load
        if external_source.startswith("http://") or external_source.startswith("https://"):
            df = load_csv_from_url(external_source)
        else:
            # Treat as Kaggle dataset slug
            df = load_csv_from_kaggle(external_source)

        if df is None:
            return templates.TemplateResponse("index.html", {
                "request": request, 
                "error": "Failed to load dataset from external source. Please check the URL or Kaggle dataset slug."
            })

        state = build_dashboard_from_df(df)

        if state is None:
            return templates.TemplateResponse("index.html", {
                "request": request, 
                "error": "Failed to build dashboard from external dataset."
            })

        df = state.df
        dataset_profile = state.dataset_profile
        profile = state.profile
        kpis = state.kpis
        charts = state.charts
        primary_chart = state.primary_chart
        category_charts = state.category_charts
        all_charts = state.all_charts

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
            }
        )
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": f"An error occurred while loading the external dataset: {str(e)}"
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)