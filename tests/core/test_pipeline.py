import pandas as pd
import pytest
from src.core.pipeline import build_dashboard_from_df

# Define the path to the fixture relative to the test file
# This makes the test runner find the file correctly
FIXTURE_PATH = 'tests/fixtures/sample_data.csv'

def test_pipeline_end_to_end_from_dataframe():
    """
    Tests the full pipeline from a DataFrame to a final DashboardState.
    This acts as an integration test to ensure all components work together.
    """
    # 1. Load the sample data from the fixture
    try:
        df = pd.read_csv(FIXTURE_PATH)
    except FileNotFoundError:
        pytest.fail(f"Test fixture not found at {FIXTURE_PATH}. Ensure the path is correct relative to the project root.")

    # 2. Run the full pipeline
    state = build_dashboard_from_df(df)

    # 3. Assert the results to ensure the pipeline ran successfully
    assert state is not None, "The pipeline returned a None state, indicating a critical failure."
    
    # Assert that the error list is empty, proving no silent failures occurred
    assert state.errors is not None, "The state object is missing the 'errors' attribute."
    assert len(state.errors) == 0, f"The pipeline produced errors: {state.errors}"

    # Assert that the dashboard is NOT blank
    assert state.kpis, "The pipeline did not generate any KPIs."
    assert len(state.kpis) > 0, "KPI list is empty."
    
    assert state.all_charts, "The pipeline did not generate any charts."
    assert len(state.all_charts) > 0, "Chart list is empty."

    # Assert that column roles were inferred correctly
    profile = state.dataset_profile
    assert profile is not None
    
    column_roles = {col['name']: col['role'] for col in profile['columns']}
    
    assert column_roles.get('record_id') == 'identifier'
    assert column_roles.get('transaction_date') == 'datetime'
    assert column_roles.get('amount') == 'numeric'
    assert column_roles.get('category') == 'categorical'
    assert column_roles.get('description') == 'text'
