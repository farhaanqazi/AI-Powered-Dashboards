"""
Centralized identifier detection utility to be used across all modules
"""
import re
import pandas as pd
from typing import Tuple


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