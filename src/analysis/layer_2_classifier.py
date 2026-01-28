"""
Layer 2: Semantic Classifier

Responsibilities:
-   Consumes the `SyntacticProfile` from Layer 1.
-   Applies heuristics and rules to infer the semantic role of each column.
-   Assigns semantic tags.
-   Produces an `EnrichedProfile`.
"""
import logging
import pandas as pd
from typing import Dict, List, Any, Tuple
import re

from src.analysis.data_structures import SyntacticProfile, EnrichedProfile
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_bool_dtype
from src.utils.identifier_detector import is_likely_identifier_with_confidence

logger = logging.getLogger(__name__)

def run_semantic_classification(
    profiles: Dict[str, SyntacticProfile],
    df: pd.DataFrame
) -> Dict[str, EnrichedProfile]:
    """
    Performs Layer 2 analysis: assigns semantic roles to profiled columns.
    """
    enriched_profiles: Dict[str, EnrichedProfile] = {}

    for name, profile in profiles.items():
        role = "unknown"
        semantic_tags = []

        # --- Role Inference Hierarchy ---
        # The order of these checks is critical.

        # 1. Handle unambiguous data types first.
        if is_bool_dtype(df[name]):
            role = "boolean"
        elif is_datetime64_any_dtype(df[name]):
            role = "datetime"

        # 2. **CRITICAL FIX**: Check for identifiers *before* checking for generic numeric types.
        # This prevents numeric IDs from being misclassified as aggregatable measures.
        elif is_likely_identifier_with_confidence(df[name], name)[0]:
            role = "identifier"

        # 3. Check for generic numerics if it's not an identifier.
        elif is_numeric_dtype(df[name]):
            role = "numeric"

        # 4. For 'object' types, perform more detailed checks.
        elif profile.dtype == 'object':
            # Attempt to parse as datetime
            if pd.to_datetime(df[name], errors='coerce').notna().mean() > 0.85: # Increased confidence to 85%
                role = "datetime"
            # Check for low-cardinality strings
            elif profile.unique_count < 50 and profile.unique_count / profile.stats['count'] < 0.2: # Explicit count limit and lower ratio
                role = "categorical"
            # Fallback for high-cardinality potential categorical (e.g., more than 50 unique values but still a small ratio)
            elif profile.unique_count / profile.stats['count'] < 0.05 and profile.unique_count > 50:
                role = "categorical"
            # Otherwise, it's high-cardinality free text.
            else:
                role = "text"

        # 5. Fallback for any other types.
        else:
            role = "text"

        # --- Semantic Tagging ---
        # Apply semantic tags only if the role is numeric, to avoid tagging text fields with currency symbols
        if role == 'numeric':
            # Check for monetary symbols more robustly by looking at string representations
            sample_values = df[name].dropna().astype(str).sample(min(len(df[name].dropna()), 100), random_state=42) # Sample up to 100 values
            if any(re.search(r'[\$\€\£]', val) for val in sample_values): # Corrected symbols
                semantic_tags.append('monetary')

        enriched_profiles[name] = EnrichedProfile(
            role=role,
            semantic_tags=semantic_tags,
            **profile.__dict__
        )

    logger.info(f"Layer 2: Semantic classification complete. Assigned roles to {len(enriched_profiles)} columns.")
    return enriched_profiles
