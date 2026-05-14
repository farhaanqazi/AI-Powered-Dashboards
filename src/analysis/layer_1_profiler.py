"""
Layer 1: Syntactic Profiler

Responsibilities:
-   Interfaces directly with the raw DataFrame.
-   Computes objective, factual metadata for each column (dtype, null count, basic stats, etc.).
-   Produces a structured `SyntacticProfile` without interpretation.
"""
import logging
import pandas as pd
from typing import Dict, List, Any

from src.analysis.data_structures import SyntacticProfile

logger = logging.getLogger(__name__)

def run_syntactic_profiling(df: pd.DataFrame, max_cols: int = 50) -> Dict[str, SyntacticProfile]:
    """
    Performs Layer 1 analysis: raw, objective profiling of a DataFrame.
    
    Args:
        df: The input DataFrame to be profiled.
        max_cols: The maximum number of columns to profile for performance reasons.

    Returns:
        A dictionary mapping column names to their `SyntacticProfile` objects.
    """
    if df.empty:
        logger.warning("Layer 1: Input DataFrame is empty, returning empty syntactic profile.")
        return {}

    n_cols_original = df.shape[1]
    columns_to_profile = df.columns[:min(max_cols, n_cols_original)]
    
    profiles: Dict[str, SyntacticProfile] = {}

    for col_name in columns_to_profile:
        s = df[col_name]

        if s.isna().all():
            logger.info(f"Layer 1: Skipping column '{col_name}' as it contains only null values.")
            continue
        
        # --- Basic Facts ---
        dtype = str(s.dtype)
        missing_count = int(s.isna().sum())
        unique_count = int(s.nunique())
        
        stats = {"count": int(len(s.dropna()))}
        top_categories = []

        # --- Data-type specific stats ---
        if pd.api.types.is_numeric_dtype(s):
            s_clean = s.dropna()
            if len(s_clean) > 0:
                try:
                    stats.update({
                        "min": float(s_clean.min()),
                        "max": float(s_clean.max()),
                        "mean": float(s_clean.mean()),
                        "std": float(s_clean.std()) if len(s_clean) > 1 else 0.0,
                        "median": float(s_clean.median()),
                        "q25": float(s_clean.quantile(0.25)),
                        "q75": float(s_clean.quantile(0.75)),
                        "sum": float(s_clean.sum()),
                        "variance": float(s_clean.var()) if len(s_clean) > 1 else 0.0,
                    })
                except (TypeError, ValueError) as e:
                    logger.warning(f"Layer 1: Could not calculate numeric stats for column '{col_name}': {e}")

        # Always calculate value counts for potential categorical display
        try:
            value_counts = s.value_counts(dropna=True)
            if not value_counts.empty:
                top_categories = [
                    {"value": str(idx), "count": int(cnt)}
                    for idx, cnt in value_counts.head(10).items()
                ]
        except Exception as e:
             logger.warning(f"Layer 1: Could not calculate value counts for column '{col_name}': {e}")

        profiles[col_name] = SyntacticProfile(
            name=col_name,
            dtype=dtype,
            null_count=missing_count,
            unique_count=unique_count,
            stats=stats,
            top_categories=top_categories
        )

    logger.info(f"Layer 1: Syntactic profiling complete for {len(profiles)} columns.")
    return profiles