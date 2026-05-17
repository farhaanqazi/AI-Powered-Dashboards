"""
Layer 3: Relational Analyzer

Responsibilities:
-   Consumes the `EnrichedProfile` from Layer 2 and the DataFrame.
-   Analyzes relationships *between* columns (e.g., correlation).
-   Produces a list of `RelationalInsight` objects.
"""
import logging
import pandas as pd
from typing import Dict, List, Any
from scipy.stats import pearsonr

from src.analysis.data_structures import EnrichedProfile, RelationalInsight
from src.contract.role_router import is_correlatable
from src import config as _cfg

logger = logging.getLogger(__name__)

def run_relational_analysis(
    df: pd.DataFrame, 
    enriched_profiles: Dict[str, EnrichedProfile]
) -> List[RelationalInsight]:
    """
    Performs Layer 3 analysis to find significant relationships between columns.

    Args:
        df: The input DataFrame.
        enriched_profiles: The output from Layer 2, containing semantic info for each column.

    Returns:
        A list of RelationalInsight objects describing found relationships.
    """
    insights: List[RelationalInsight] = []
    
    # --- 1. Correlation Analysis for Numeric Columns ---
    
    # Identify numeric *measures* with meaningful variance. The role router
    # excludes identifiers and year columns: numeric by storage, not meaning.
    numeric_cols_for_corr = [
        profile.name for profile in enriched_profiles.values()
        if is_correlatable(profile) and profile.stats.get('std', 0) > 0.001
    ]
    
    logger.info(f"Layer 3: Found {len(numeric_cols_for_corr)} numeric columns with variance for correlation analysis.")

    # Calculate pairwise correlations
    for i, col1_name in enumerate(numeric_cols_for_corr):
        for j, col2_name in enumerate(numeric_cols_for_corr[i+1:]):
            # Drop NA's for the pair to align them
            aligned_df = df[[col1_name, col2_name]].dropna()
            
            if len(aligned_df) < 20: # Need a reasonable number of samples for a stable correlation
                continue

            try:
                corr, p_value = pearsonr(aligned_df[col1_name], aligned_df[col2_name])
                
                # Only report moderate to strong correlations that are statistically significant
                if pd.isna(corr) or abs(corr) < _cfg.MIN_CORRELATION or p_value > 0.05:
                    continue
                
                strength = "strong" if abs(corr) >= 0.7 else "moderate"
                
                insights.append(RelationalInsight(
                    type="correlation",
                    columns=[col1_name, col2_name],
                    details={
                        "correlation_coefficient": corr,
                        "p_value": p_value,
                        "strength": strength,
                        "sample_size": len(aligned_df)
                    }
                ))
            except Exception as e:
                logger.warning(f"Layer 3: Could not calculate correlation for '{col1_name}' and '{col2_name}': {e}")

    # --- (Future) 2. Add other relational analyses here ---
    # For example, categorical-vs-numeric analysis (ANOVA-like) or time-series pair detection.

    logger.info(f"Layer 3: Relational analysis complete. Found {len(insights)} significant insights.")
    return insights