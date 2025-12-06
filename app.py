from flask import Flask, render_template, request, flash, redirect, url_for
import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from src.core.pipeline import build_dashboard_from_file, build_dashboard_from_df
from src.data.parser import load_csv_from_url, load_csv_from_kaggle

# Create Flask app with proper configuration
def create_app():
    app = Flask(__name__)

    # Configuration from environment variables
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    # Remove MAX_CONTENT_LENGTH to avoid upload size issues in certain environments
    # The upload size will be limited by the hosting environment instead

    # Setup logging
    if not app.debug:
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
    else:
        app.logger.setLevel(logging.DEBUG)

    return app

app = create_app()

@app.route("/")
def index():
    # This is the main index route that renders the upload page
    # The index.html template contains both file upload and external data loading options
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    try:
        uploaded_file = request.files.get("dataset")

        if not uploaded_file or not uploaded_file.filename:
            app.logger.error("No file uploaded")
            flash("No file uploaded. Please select a CSV file to upload.", "error")
            return redirect(url_for('index'))

        if not uploaded_file.filename.lower().endswith('.csv'):
            app.logger.error(f"Invalid file type: {uploaded_file.filename}")
            flash("Invalid file type. Please upload a CSV file.", "error")
            return redirect(url_for('index'))

        # Use the central pipeline to build everything
        state = build_dashboard_from_file(uploaded_file)

        if state is None:
            app.logger.error("Failed to read CSV file")
            flash("Failed to read CSV file. Please ensure the file is a valid CSV.", "error")
            return redirect(url_for('index'))

        df = state.df
        dataset_profile = state.dataset_profile
        profile = state.profile
        kpis = state.kpis
        charts = state.charts
        primary_chart = state.primary_chart
        category_charts = state.category_charts
        all_charts = state.all_charts

        if app.config['DEBUG']:
            app.logger.info("\n=== Data Preview ===")
            app.logger.info(df.head())

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

    except Exception as e:
        app.logger.exception("Error processing uploaded file")
        flash(f"An error occurred while processing the file: {str(e)}", "error")
        return redirect(url_for('index'))


@app.route("/load_external", methods=["POST"])
def load_external():
    """
    Load a dataset either from:
    - a direct CSV URL (http/https)
    - or a Kaggle dataset slug (owner/dataset)
    """
    try:
        source = request.form.get("external_source", "").strip()

        if not source:
            app.logger.error("No source provided")
            flash("No source provided. Please enter a CSV URL or Kaggle dataset slug.", "error")
            return redirect(url_for('index'))

        # Decide how to load
        if source.startswith("http://") or source.startswith("https://"):
            app.logger.info(f"Loading CSV from URL: {source}")
            df = load_csv_from_url(source)
        else:
            # Treat as Kaggle dataset slug
            app.logger.info(f"Loading Kaggle dataset: {source}")
            df = load_csv_from_kaggle(source)

        if df is None:
            app.logger.error(f"Failed to load dataset from external source: {source}")
            flash("Failed to load dataset from external source. Please check the URL or Kaggle dataset slug.", "error")
            return redirect(url_for('index'))

        state = build_dashboard_from_df(df)

        if state is None:
            app.logger.error("Failed to build dashboard from external dataset")
            flash("Failed to build dashboard from external dataset.", "error")
            return redirect(url_for('index'))

        df = state.df
        dataset_profile = state.dataset_profile
        profile = state.profile
        kpis = state.kpis
        charts = state.charts
        primary_chart = state.primary_chart
        category_charts = state.category_charts
        all_charts = state.all_charts

        if app.config['DEBUG']:
            app.logger.info("\n=== External Data Preview ===")
            app.logger.info(df.head())

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

    except Exception as e:
        app.logger.exception(f"Error loading external dataset: {source}")
        flash(f"An error occurred while loading the external dataset: {str(e)}", "error")
        return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=app.config['DEBUG'])
