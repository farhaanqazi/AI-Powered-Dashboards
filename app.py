from flask import Flask, render_template, request
from src.core.pipeline import build_dashboard_from_file, build_dashboard_from_df
from src.data.parser import load_csv_from_url, load_csv_from_kaggle

DEBUG = False   # Set to True if you want detailed terminal logging

app = Flask(__name__)


@app.route("/")
def index():
    # This is the main index route that renders the upload page
    # The index.html template contains both file upload and external data loading options
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    uploaded_file = request.files.get("dataset")

    if not uploaded_file:
        return "No file uploaded", 400

    # Use the central pipeline to build everything
    state = build_dashboard_from_file(uploaded_file)

    if state is None:
        return "Failed to read CSV file", 400

    df = state["df"]
    dataset_profile = state["dataset_profile"]
    profile = state["profile"]
    kpis = state["kpis"]
    charts = state["charts"]
    primary_chart = state["primary_chart"]
    category_charts = state["category_charts"]
    all_charts = state["all_charts"]

    if DEBUG:
        print("\n=== Data Preview ===")
        print(df.head())

    return render_template(
        "dashboard.html",
        profile=profile,
        dataset_profile=dataset_profile,
        kpis=kpis,
        charts=charts,
        primary_chart=primary_chart,
        category_charts=category_charts,
        all_charts=all_charts,
    )


@app.route("/load_external", methods=["POST"])
def load_external():
    """
    Load a dataset either from:
    - a direct CSV URL (http/https)
    - or a Kaggle dataset slug (owner/dataset)
    """
    source = request.form.get("external_source", "").strip()

    if not source:
        return "No source provided", 400

    # Decide how to load
    if source.startswith("http://") or source.startswith("https://"):
        df = load_csv_from_url(source)
    else:
        # Treat as Kaggle dataset slug
        df = load_csv_from_kaggle(source)

    if df is None:
        return "Failed to load dataset from external source.", 400

    state = build_dashboard_from_df(df)

    if state is None:
        return "Failed to build dashboard from external dataset.", 500

    df = state["df"]
    dataset_profile = state["dataset_profile"]
    profile = state["profile"]
    kpis = state["kpis"]
    charts = state["charts"]
    primary_chart = state["primary_chart"]
    category_charts = state["category_charts"]
    all_charts = state["all_charts"]

    if DEBUG:
        print("\n=== External Data Preview ===")
        print(df.head())

    return render_template(
        "dashboard.html",
        profile=profile,
        dataset_profile=dataset_profile,
        kpis=kpis,
        charts=charts,
        primary_chart=primary_chart,
        category_charts=category_charts,
        all_charts=all_charts,
    )


if __name__ == "__main__":
    app.run(debug=True)
