import pandas as pd
import pytest
from src.data.analyser import _infer_role_advanced

def test_infer_role_numeric_string():
    """
    Tests if a series of numeric strings is correctly inferred as 'numeric'.
    This verifies the fix for data coercion.
    """
    numeric_string_series = pd.Series(["10", "20", "30.5", "40", None, "50.0"])
    role, confidence, _, _, _ = _infer_role_advanced(numeric_string_series)
    assert role == "numeric"
    assert confidence > 0.8

def test_infer_role_identifier_strings():
    """
    Tests if a series of ID-like strings is correctly inferred as 'identifier'.
    """
    id_series = pd.Series(["ID-001", "ID-002", "ID-003", "ID-004", "ID-005"])
    id_series.name = "customer_id"
    role, _, _, _, _ = _infer_role_advanced(id_series)
    assert role == "identifier"

def test_infer_role_categorical_strings():
    """
    Tests if a series with a small number of unique strings is inferred as 'categorical'.
    """
    categorical_series = pd.Series(["Apple", "Banana", "Apple", "Cherry", "Banana"])
    role, _, _, _, _ = _infer_role_advanced(categorical_series)
    assert role == "categorical"

def test_infer_role_high_cardinality_text():
    """
    Tests if a series with many unique, long strings is inferred as 'text'.
    """
    text_series = pd.Series([
        "This is the first long sentence.",
        "This is a second, completely different sentence.",
        "Yet another unique phrase for testing.",
        "The quick brown fox jumps over the lazy dog.",
        "Each entry is unique to ensure high cardinality."
    ])
    role, _, _, _, _ = _infer_role_advanced(text_series)
    assert role == "text"

def test_infer_role_mixed_currency_gets_monetary_tag():
    """
    Tests if a series with mixed currency strings gets the 'monetary' semantic tag.
    """
    currency_series = pd.Series(["$100.50", "€50.25", "99.99 USD", "25.00"])
    # Note: _infer_role_advanced will still classify the role as 'text' because of the symbols,
    # but it should correctly identify the semantic content. The coercion to numeric happens later.
    # Our updated analyser now coerces first, so the role should be numeric.
    role, _, _, _, semantic_tags = _infer_role_advanced(currency_series)
    
    assert role == "numeric"
    assert "monetary" in semantic_tags

def test_handle_all_nan_series():
    """
    Tests if a series with all NaN/None values is handled gracefully.
    """
    all_nan_series = pd.Series([None, pd.NA, None, None])
    role, confidence, provenance, _, _ = _infer_role_advanced(all_nan_series)
    assert role == "text"
    assert provenance == "all_nan"
    assert confidence < 0.5
