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

from src.analysis.data_structures import SyntacticProfile, EnrichedProfile
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_bool_dtype

logger = logging.getLogger(__name__)

def _is_likely_identifier(profile: SyntacticProfile, series: pd.Series) -> bool:
    """
    Detects if a column is likely an identifier based on its profile and data.
    """
    # High uniqueness is a prerequisite.
    # Use a slightly lower threshold here as this check runs before others.
    unique_ratio = profile.unique_count / profile.stats['count'] if profile.stats['count'] > 0 else 0
    if unique_ratio < 0.9:
        return False

    # Strong signal: Name contains 'id', 'key', 'code', etc.
    name_lower = profile.name.lower()
    id_name_keywords = ["id", "key", "uuid", "guid", "code", "token", "hash", "number"]
    if any(keyword in name_lower for keyword in id_name_keywords):
        return True
        
    # Strong signal: Looks like a UUID
    if profile.dtype == 'object':
        sample = series.dropna().head(20).astype(str)
        uuid_pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
        if not sample.empty and sample.str.match(uuid_pattern).mean() > 0.5:
            return True

    # If it's 100% unique and numeric, it's very likely an ID.
    if unique_ratio == 1.0 and is_numeric_dtype(series):
        return True

    return False

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
        elif _is_likely_identifier(profile, df[name]):
            role = "identifier"

        # 3. Check for generic numerics if it's not an identifier.
        elif is_numeric_dtype(df[name]):
            role = "numeric"
        
        # 4. For 'object' types, perform more detailed checks.
        elif profile.dtype == 'object':
            try:
                # Attempt to parse as datetime
                if pd.to_datetime(df[name], errors='coerce', infer_datetime_format=True).notna().mean() > 0.7:
                    role = "datetime"
                # Check for low-cardinality strings
                elif profile.unique_count / profile.stats['count'] < 0.5 and profile.unique_count < 100:
                    role = "categorical"
                # Otherwise, it's high-cardinality free text.
                else:
                    role = "text"
            except Exception:
                role = "text" # Fallback
        
        # 5. Fallback for any other types.
        else:
            role = "text"

        # --- Semantic Tagging ---
        if role == 'numeric':
            col_str_sample = df[name].dropna().head(100).to_string().lower()
            if any(sym in col_str_sample for sym in ['$', '€', '£']):
                semantic_tags.append('monetary')

        enriched_profiles[name] = EnrichedProfile(
            role=role,
            semantic_tags=semantic_tags,
            **profile.__dict__
        )

    logger.info(f"Layer 2: Semantic classification complete. Assigned roles to {len(enriched_profiles)} columns.")
    return enriched_profiles