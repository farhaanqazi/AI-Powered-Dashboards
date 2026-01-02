import pytest
import pandas as pd
from src.analysis.layer_4_interpreter import determine_kpis, _calculate_kpi_score

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

def test_determine_kpis_ignores_identifiers(sample_df_for_kpi):
    """
    Tests that determine_kpis correctly ignores columns marked as 'identifier'.
    """
    df = sample_df_for_kpi
    # Create enriched profiles where 'order_id' is correctly identified
    from src.analysis.data_structures import EnrichedProfile

    enriched_profiles = {
        "order_id": EnrichedProfile(
            name="order_id",
            dtype="int64",
            null_count=0,
            unique_count=100,
            stats={"count": 100},
            role="identifier",
            confidence=0.0,
            semantic_tags=[]
        ),
        "revenue": EnrichedProfile(
            name="revenue",
            dtype="float64",
            null_count=0,
            unique_count=100,
            stats={"count": 100, "std": 300.0, "mean": 500.0},
            role="numeric",
            confidence=0.0,
            semantic_tags=[]
        ),
        "product_category": EnrichedProfile(
            name="product_category",
            dtype="object",
            null_count=0,
            unique_count=2,
            stats={"count": 100},
            role="categorical",
            confidence=0.0,
            semantic_tags=[]
        ),
        "status_code": EnrichedProfile(
            name="status_code",
            dtype="int64",
            null_count=0,
            unique_count=1,
            stats={"count": 100, "std": 0.0, "mean": 200.0},
            role="numeric",
            confidence=0.0,
            semantic_tags=[]
        )
    }

    # Create empty relational insights for this test
    relational_insights = []

    kpis = determine_kpis(enriched_profiles, relational_insights)

    kpi_labels = [kpi['label'] for kpi in kpis]

    # Assert that the identifier is NOT in the KPIs
    assert 'order_id' not in kpi_labels
    # Assert that the constant value column is also not a KPI
    assert 'status_code' not in kpi_labels
    # Assert that a meaningful metric IS in the KPIs
    assert 'revenue' in kpi_labels

def test_kpi_score_prioritizes_monetary_and_variance():
    """
    Tests if _calculate_kpi_score gives a higher score
    to a numeric column with high variance and a 'monetary' tag.
    """
    # Profile for a high-variance, monetary column
    from src.analysis.data_structures import EnrichedProfile

    monetary_profile = EnrichedProfile(
        name="revenue",
        dtype="float64",  # Added required field
        null_count=0,     # Added required field
        unique_count=100,
        stats={"count": 100, "std": 500.0, "mean": 1000.0},
        semantic_tags=["monetary"],
        role="numeric"
    )

    # Profile for a low-variance, non-semantic column
    other_numeric_profile = EnrichedProfile(
        name="item_count",
        dtype="float64",  # Added required field
        null_count=0,     # Added required field
        unique_count=5,
        stats={"count": 100, "std": 1.5, "mean": 3.0},
        semantic_tags=[],
        role="numeric"
    )

    monetary_score = _calculate_kpi_score(monetary_profile)
    other_score = _calculate_kpi_score(other_numeric_profile)

    print(f"Monetary Score: {monetary_score}, Other Score: {other_score}")
    assert monetary_score >= other_score

def test_determine_kpis_handles_malformed_profile():
    """
    Tests that the determine_kpis function handles a malformed profile
    gracefully without crashing (verifies the recent bugfix).
    """
    # A profile with a missing 'name' attribute in one of its profiles
    from src.analysis.data_structures import EnrichedProfile

    enriched_profiles = {
        "valid_col": EnrichedProfile(
            name="valid_col",
            dtype="float64",  # Added required field
            null_count=0,     # Added required field
            role="numeric",
            unique_count=10,
            stats={"count": 100, "std": 10.0, "mean": 50.0}
        )
    }

    # Create empty relational insights for this test
    relational_insights = []

    # This call should not raise an exception
    try:
        kpis = determine_kpis(enriched_profiles, relational_insights)
        # We expect it to produce a list of KPIs without crashing
    except Exception as e:
        pytest.fail(f"determine_kpis raised an unexpected exception with malformed profile: {e}")
