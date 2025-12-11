"""
Centralized identifier detection utility to be used across all modules
"""
import re
import pandas as pd
from typing import Tuple


def is_likely_identifier(series: pd.Series, name: str = "") -> bool:
    """
    Determine if a series is likely an identifier based on multiple heuristics.
    """
    n_total = len(series)
    if n_total == 0:
        return False

    n_unique = series.nunique()
    unique_ratio = n_unique / n_total if n_total > 0 else 0.0

    # Check for high cardinality (potential ID)
    if unique_ratio > 0.98:
        # Check if it's numeric (potential sequential ID)
        if pd.api.types.is_numeric_dtype(series):
            numeric_vals = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_vals) > 5:  # Need at least 5 values to check sequence
                sorted_vals = numeric_vals.sort_values()
                diffs = sorted_vals.diff().dropna()
                if len(diffs) > 0:
                    # Check for mostly constant differences (sequential IDs)
                    unique_diffs = diffs.unique()
                    if len(unique_diffs) == 1 and abs(unique_diffs[0] - 1) < 0.01:  # Step of 1
                        return True
        # Check for UUID patterns in string values
        if series.dtype == 'object':
            sample = series.dropna().head(20).astype(str)
            uuid_matches = 0
            for val in sample:
                # Check for UUID v4 pattern (with case insensitivity)
                if re.match(r'^[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}$', val, re.IGNORECASE):
                    uuid_matches += 1
            if uuid_matches / len(sample) > 0.5:  # More than 50% are UUIDs
                return True

    # Check for ID-like names
    name_lower = name.lower()
    id_keywords = [
        "id", "uuid", "guid", "key", "code", "no", "number", "index",
        "account", "user", "customer", "product", "item", "order",
        "transaction", "invoice", "booking", "session", "token", "hash"
    ]

    if any(keyword in name_lower for keyword in id_keywords):
        return True

    return False


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
        "id", "uuid", "guid", "key", "code", "no", "number", "index",
        "account", "user", "customer", "product", "item", "order",
        "transaction", "invoice", "booking", "session", "token", "hash"
    ]

    matching_keywords = [kw for kw in id_keywords if kw in name_lower]
    if matching_keywords:
        # Calculate confidence based on how many keywords match and their position in name
        keyword_confidence = min(0.8, len(matching_keywords) * 0.3)
        # Boost confidence if important keywords are found
        important_keywords = ["id", "uuid", "key", "code", "account", "user", "customer"]
        important_matches = sum(1 for kw in matching_keywords if kw in important_keywords)
        keyword_confidence += important_matches * 0.15
        detection_signals["name_pattern"] = min(1.0, keyword_confidence)

    # Calculate overall confidence based on signal strengths and weights
    if detection_signals:
        # Weight different signals appropriately
        weights = {
            "uuid_pattern": 1.0,              # Highest confidence for UUIDs
            "sequential_step1": 0.95,         # High confidence for clear sequential patterns
            "very_high_cardinality": 0.9,     # High confidence for extremely high uniqueness
            "sequential_low_variance": 0.85,  # High confidence for sequential patterns
            "high_cardinality": 0.8,          # Good confidence for high uniqueness
            "name_pattern": 0.75,             # Good confidence for name patterns
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
        is_identifier = max_confidence > 0.6

        return is_identifier, best_signal, max_confidence

    return False, "no_signals", 0.0