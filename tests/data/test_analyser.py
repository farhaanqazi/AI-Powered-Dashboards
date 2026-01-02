import pandas as pd
import pytest
from src.utils.identifier_detector import is_likely_identifier_with_confidence

def test_identify_identifier_with_confidence():
    """
    Tests if the identifier detection function works properly.
    """
    id_series = pd.Series(["ID-001", "ID-002", "ID-003", "ID-004", "ID-005"])
    id_series.name = "customer_id"
    is_id, signal, confidence = is_likely_identifier_with_confidence(id_series, "customer_id")
    assert is_id == True
    assert confidence >= 0.6

def test_identify_identifier_basic():
    """
    Tests if a series of ID-like strings is correctly identified as an identifier.
    """
    id_series = pd.Series(["ID-001", "ID-002", "ID-003", "ID-004", "ID-005"])
    id_series.name = "customer_id"
    is_id, signal, confidence = is_likely_identifier_with_confidence(id_series, "customer_id")
    assert is_id == True

def test_identify_non_identifier():
    """
    Tests if a series with categorical values is correctly identified as non-identifier.
    """
    categorical_series = pd.Series(["Apple", "Banana", "Apple", "Cherry", "Banana"])
    is_id, signal, confidence = is_likely_identifier_with_confidence(categorical_series, "fruit_type")
    assert is_id == False

def test_identify_uuid_pattern():
    """
    Tests if a series with UUID patterns is correctly identified as identifier.
    """
    uuid_series = pd.Series([
        "550e8400-e29b-41d4-a716-446655440000",
        "550e8400-e29b-41d4-a716-446655440001",
        "550e8400-e29b-41d4-a716-446655440002",
        "550e8400-e29b-41d4-a716-446655440003",
        "550e8400-e29b-41d4-a716-446655440004"
    ])
    uuid_series.name = "user_id"
    is_id, signal, confidence = is_likely_identifier_with_confidence(uuid_series, "user_id")
    assert is_id == True
    assert confidence >= 0.8

def test_identify_sequential_numbers():
    """
    Tests if a series with sequential numbers is correctly identified as identifier.
    """
    seq_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    seq_series.name = "row_id"
    is_id, signal, confidence = is_likely_identifier_with_confidence(seq_series, "row_id")
    assert is_id == True

def test_handle_all_nan_series():
    """
    Tests if a series with all NaN/None values is handled gracefully.
    """
    all_nan_series = pd.Series([None, pd.NA, None, None])
    is_id, signal, confidence = is_likely_identifier_with_confidence(all_nan_series, "test_col")
    assert is_id == False
    assert confidence < 0.5
