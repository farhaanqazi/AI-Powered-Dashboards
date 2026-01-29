"""
Centralized identifier detection utility to be used across all modules
"""
import re
import pandas as pd
from typing import Tuple


def is_likely_identifier_with_confidence(s: pd.Series, name: str) -> Tuple[bool, str, float]:
    """
    Check if a series is likely an identifier with confidence scoring.
    """
    n_total = len(s)
    if n_total == 0:
        return False, "empty", 0.0

    n_unique = s.nunique()
    unique_ratio = n_unique / n_total if n_total > 0 else 0.0

    detection_signals = {}

    # Signal 1: High cardinality (potential ID)
    if unique_ratio > 0.98:
        detection_signals["very_high_cardinality"] = min(0.95, unique_ratio)
    elif unique_ratio > 0.95:
        detection_signals["high_cardinality"] = min(0.85, unique_ratio * 0.9)
    elif unique_ratio > 0.90:
        detection_signals["moderate_cardinality"] = unique_ratio * 0.6

    # Signal 2: Sequential numeric pattern (common in internal IDs)
    if pd.api.types.is_numeric_dtype(s):
        numeric_vals = pd.to_numeric(s, errors='coerce').dropna()
        if len(numeric_vals) > 5:  # Need at least 5 values to check sequence
            sorted_vals = numeric_vals.sort_values()
            diffs = sorted_vals.diff().dropna()
            if len(diffs) > 0:
                # Check for mostly constant differences (sequential IDs)
                unique_diffs = diffs.unique()
                if len(unique_diffs) == 1 and abs(unique_diffs[0] - 1) < 0.01:  # Step of 1
                    detection_signals["sequential_step1"] = min(0.95, len(numeric_vals) / max(len(numeric_vals), 10))
                elif len(unique_diffs) <= 3 and diffs.std() < diffs.mean() * 0.1:  # Low variance in steps
                    detection_signals["sequential_low_variance"] = min(0.85, diffs.mean() * 0.7)

    # Signal 3: UUID pattern
    if s.dtype == 'object':
        sample = s.dropna().head(20).astype(str)
        uuid_matches = 0
        for val in sample:
            # Check for UUID v4 pattern (with case insensitivity)
            if re.match(r'^[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}$', val, re.IGNORECASE):
                uuid_matches += 1
        if len(sample) > 0:
            uuid_ratio = uuid_matches / len(sample)
            if uuid_ratio > 0.5:  # More than 50% are UUIDs
                detection_signals["uuid_pattern"] = uuid_ratio

    # Signal 4: Name-based detection (semantic heuristics)
    name_lower = name.lower()
    id_keywords = [
        "id", "uuid", "guid", "key", "code", "sku", "passport", "licence", "license", # Strong identifiers
        "account_number", "customer_number", "order_number", "product_number", "item_number", "transaction_number", # Number-based identifiers
        "invoice_number", "booking_id", "session_id", "token_id", "hash_id", # Specific IDs
    ]

    matching_keywords = [kw for kw in id_keywords if kw in name_lower]
    if matching_keywords:
        # Calculate confidence based on how many keywords match and their position in name
        # Stronger boost for presence of dedicated identifier keywords
        keyword_confidence = min(0.9, len(matching_keywords) * 0.2 + (0.1 if any(k in ["id", "uuid", "key"] for k in matching_keywords) else 0))
        detection_signals["name_pattern"] = min(1.0, keyword_confidence)

    # Calculate overall confidence based on signal strengths and weights
    if detection_signals:
        # Weight different signals appropriately
        weights = {
            "uuid_pattern": 1.0,              # Highest confidence for UUIDs
            "sequential_step1": 0.98,         # Very high confidence for clear sequential patterns
            "very_high_cardinality": 0.92,    # High confidence for extremely high uniqueness
            "sequential_low_variance": 0.88,  # High confidence for sequential patterns with low variance
            "high_cardinality": 0.85,         # Good confidence for high uniqueness
            "name_pattern": 0.70,             # Slightly reduced weight for name patterns, now more specific
            "moderate_cardinality": 0.4       # Lower confidence for moderate uniqueness
        }

        max_confidence = 0
        best_signal = ""

        for signal, score in detection_signals.items():
            weight = weights.get(signal, 0.6)  # Default weight of 0.6
            weighted_score = score * weight
            if weighted_score > max_confidence:
                max_confidence = min(1.0, weighted_score)
                best_signal = signal

        # Consider it an identifier if confidence exceeds threshold
        is_identifier = max_confidence > 0.65 # Slightly raise threshold for stronger evidence

        return is_identifier, best_signal, max_confidence

    return False, "no_signals", 0.0