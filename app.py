from flask import Flask, render_template, request
from src.core.pipeline import build_dashboard_from_file  # use central pipeline

DEBUG = False   # Set to True if you want detailed terminal logging

app = Flask(__name__)

@app.route("/")
def index():
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

    # Optional debug logging
    if DEBUG:
        print("\n=== Data Preview ===")
        print(df.head())

        print("\n=== Dataset Summary ===")
        print(f"Rows: {dataset_profile['n_rows']}, Columns: {dataset_profile['n_cols']}")

        print("\n=== First 3 Column Profiles ===")
        for col in dataset_profile["columns"][:3]:
            print(col)

    # Render dashboard
    return render_template(
        "dashboard.html",
        profile=profile,
        dataset_profile=dataset_profile,
        kpis=kpis
    )

if __name__ == "__main__":
    app.run(debug=True)
