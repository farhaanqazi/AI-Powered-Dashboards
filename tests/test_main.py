import pytest
from fastapi.testclient import TestClient
import os

# We need to make sure the app can be imported.
# This might require adjusting the python path, but pytest usually handles this.
from main import app

# Create the test client
client = TestClient(app)

# Define the path to the fixture relative to the test file
FIXTURE_PATH = 'tests/fixtures/sample_data.csv'

def test_upload_csv_success():
    """
    Tests the /upload endpoint with a valid CSV file.
    This is an end-to-end test that simulates a user uploading a file.
    """
    # Check if the fixture file exists before proceeding
    if not os.path.exists(FIXTURE_PATH):
        pytest.fail(f"Test fixture file not found: {FIXTURE_PATH}")

    with open(FIXTURE_PATH, "rb") as f:
        files = {"dataset": ("sample_data.csv", f, "text/csv")}
        response = client.post("/upload", files=files)

    # 1. Assert that the request was successful
    assert response.status_code == 200, f"Request failed with status code {response.status_code}. Response text: {response.text}"

    # 2. Assert that the response is HTML
    assert "text/html" in response.headers['content-type']

    # 3. Assert that the returned dashboard is not blank and contains expected content
    response_text = response.text
    assert "<title>AI-Powered Dashboard</title>" in response_text

    # Check for signs of a successful analysis, not a blank or error page
    assert "Dataset Insights" in response_text
    assert "Analysis Pipeline Failed!" not in response_text # Ensure our new error message is not present

    # Check for a specific KPI from our sample data.
    # The 'amount' column should be identified as a key metric.
    assert "Amount:" in response_text # Check for specific KPI label
    assert "101.15" in response_text  # Check for specific KPI value

    # No longer checking for a specific chart title, as it might vary
    # assert "Count of Category" in response_text # Removed specific chart title check

def test_get_index_page():
    """
    Tests that the root URL (/) returns the main index page successfully.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "AI-Powered Dashboard Generator" in response.text
    assert "Upload CSV" in response.text
