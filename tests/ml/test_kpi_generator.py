import pytest
import pandas as pd
from src.ml.kpi_generator import generate_kpis, _calculate_significance_score_from_profile

@pytest.fixture
def sample_df_for_kpi():
    """Provides a sample DataFrame for KPI tests."""
    data = {
        'order_id': range(100),
        'revenue': [x * 10.5 for x in range(100)],
        'product_category': ['A', 'B'] * 50,
        'status_code': [200] * 100, # Constant value
    }
    return pd.DataFrame(data)

def test_generate_kpis_ignores_identifiers(sample_df_for_kpi):
    """
    Tests that generate_kpis correctly ignores columns marked as 'identifier'.
    """
    df = sample_df_for_kpi
    # Create a profile where 'order_id' is correctly identified
    dataset_profile = {
        "n_rows": 100,
        "n_cols": 4,
        "columns": [
            {"name": "order_id", "role": "identifier", "unique_count": 100, "stats": {"count": 100}},
            {"name": "revenue", "role": "numeric", "unique_count": 100, "stats": {"count": 100, "std": 300.0, "mean": 500.0}},
            {"name": "product_category", "role": "categorical", "unique_count": 2, "stats": {"count": 100}},
            {"name": "status_code", "role": "numeric", "unique_count": 1, "stats": {"count": 100, "std": 0.0, "mean": 200.0}},
        ]
    }
    
    kpis = generate_kpis(df, dataset_profile)
    
    kpi_labels = [kpi['label'] for kpi in kpis]
    
    # Assert that the identifier is NOT in the KPIs
    assert 'order_id' not in kpi_labels
    # Assert that the constant value column is also not a KPI
    assert 'status_code' not in kpi_labels
    # Assert that a meaningful metric IS in the KPIs
    assert 'revenue' in kpi_labels

def test_significance_score_prioritizes_monetary_and_variance():
    """
    Tests if _calculate_significance_score_from_profile gives a higher score
    to a numeric column with high variance and a 'monetary' tag.
    """
    # Profile for a high-variance, monetary column
    monetary_profile = {
        "name": "revenue",
        "role": "numeric",
        "unique_count": 100,
        "stats": {"count": 100, "std": 500.0, "mean": 1000.0}
    }
    
    # Profile for a low-variance, non-semantic column
    other_numeric_profile = {
        "name": "item_count",
        "role": "numeric",
        "unique_count": 5,
        "stats": {"count": 100, "std": 1.5, "mean": 3.0}
    }

    monetary_score = _calculate_significance_score_from_profile(monetary_profile, ["monetary"])
    other_score = _calculate_significance_score_from_profile(other_numeric_profile, [])

    print(f"Monetary Score: {monetary_score}, Other Score: {other_score}")
    assert monetary_score > other_score

def test_generate_kpis_handles_malformed_profile():
    """
    Tests that the generate_kpis function handles a malformed column profile
    gracefully without crashing (verifies the recent bugfix).
    """
    df = pd.DataFrame({'col1': [1, 2, 3]})
    # A profile missing the 'name' key in one of its columns
    dataset_profile = {
        "n_rows": 3, "n_cols": 1,
        "columns": [
            {"role": "numeric"} # Missing 'name'
        ]
    }

    # This call should not raise an exception
    try:
        kpis = generate_kpis(df, dataset_profile)
        # We expect it to produce an empty list of KPIs because it skips the malformed entry
        assert kpis == []
    except Exception as e:
        pytest.fail(f"generate_kpis raised an unexpected exception with malformed profile: {e}")
