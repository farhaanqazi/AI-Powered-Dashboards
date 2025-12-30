"""
Layer 4: Interpreter / Strategist

Responsibilities:
-   Consumes the outputs from all previous layers.
-   Makes strategic decisions about what to display.
-   Scores and selects the top KPIs.
-   Selects and prioritizes the most appropriate charts.
-   Produces the final `ActionPlan` (e.g., lists of KPIs and Chart Specs).
"""
import logging
from typing import Dict, List, Any

from src.analysis.data_structures import EnrichedProfile, RelationalInsight

logger = logging.getLogger(__name__)

def _calculate_kpi_score(profile: EnrichedProfile) -> float:
    """
    Calculates a significance score for a column based on its enriched profile.
    """
    score = 0.0
    
    # Identifiers and low-variance columns are not good KPIs.
    if profile.role == 'identifier' or profile.stats.get('std', 0) < 0.01:
        return 0.0

    # Boost score for key roles
    if profile.role == 'numeric':
        score += 0.3
    if profile.role == 'categorical':
        score += 0.2
    if profile.role == 'datetime':
        score += 0.2
        
    # Boost for semantic meaning
    if 'monetary' in profile.semantic_tags:
        score += 0.4
    if 'rating' in profile.semantic_tags or 'quantity' in profile.semantic_tags:
        score += 0.3
        
    # Boost for variability (Coefficient of Variation)
    mean = profile.stats.get('mean')
    std = profile.stats.get('std')
    if mean and std and mean != 0:
        cv = abs(std / mean)
        score += min(0.3, cv)

    return min(1.0, score)

def determine_kpis(
    enriched_profiles: Dict[str, EnrichedProfile],
    relational_insights: List[RelationalInsight],
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Scores and selects the most important KPIs from the analysis results.
    """
    scored_kpis = []

    # 1. Generate KPIs from individual column profiles
    for name, profile in enriched_profiles.items():
        score = _calculate_kpi_score(profile)
        if score > 0.1: # Only consider KPIs with a minimum significance
            kpi_value = "N/A"
            if profile.role == 'numeric' and profile.stats.get('mean') is not None:
                mean_val = profile.stats.get('mean', 0.0)
                std_val = profile.stats.get('std', 0.0)
                kpi_value = f"{mean_val:.2f} (±{std_val:.2f})"
            elif profile.role in ['categorical', 'text'] and profile.top_categories:
                top_cat = profile.top_categories[0]
                kpi_value = f"Top: '{top_cat['value']}'"
            
            scored_kpis.append({
                "label": name,
                "value": kpi_value,
                "type": profile.role,
                "score": score
            })

    # 2. Generate KPIs from relational insights (e.g., correlations)
    for insight in relational_insights:
        if insight.type == 'correlation':
            details = insight.details
            scored_kpis.append({
                "label": f"Corr: {insight.columns[0]} & {insight.columns[1]}",
                "value": f"{details['correlation_coefficient']:.2f}",
                "type": "correlation",
                "score": abs(details['correlation_coefficient']) # Score is the strength of correlation
            })
            
    # 3. Sort all potential KPIs by score and return the top K
    sorted_kpis = sorted(scored_kpis, key=lambda x: x['score'], reverse=True)
    
    logger.info(f"Layer 4: Determined {len(sorted_kpis)} potential KPIs, returning top {top_k}.")
    return sorted_kpis[:top_k]

def select_charts(
    enriched_profiles: Dict[str, EnrichedProfile],
    relational_insights: List[RelationalInsight],
    max_charts: int = 20
) -> List[Dict[str, Any]]:
    """
    Selects and prioritizes a list of appropriate charts based on the analysis.
    """
    chart_specs = []

    # --- 1. Suggest charts based on individual column roles ---
    numerics = [p for p in enriched_profiles.values() if p.role == 'numeric']
    categoricals = [p for p in enriched_profiles.values() if p.role == 'categorical' and p.unique_count <= 50]
    datetimes = [p for p in enriched_profiles.values() if p.role == 'datetime']

    for p in numerics:
        chart_specs.append({
            "id": f"dist_{p.name}", "title": f"Distribution of {p.name}", "chart_type": "histogram",
            "intent": "distribution", "x_field": p.name, "priority": 2
        })

    for p in categoricals:
        chart_specs.append({
            "id": f"count_{p.name}", "title": f"Count of {p.name}", "chart_type": "bar",
            "intent": "category_count", "x_field": p.name, "priority": 1
        })

    # --- 2. Suggest charts based on relational insights ---
    for insight in relational_insights:
        if insight.type == 'correlation':
            col1, col2 = insight.columns
            chart_specs.append({
                "id": f"scatter_{col1}_{col2}", "title": f"{col1} vs. {col2}", "chart_type": "scatter",
                "intent": "scatter", "x_field": col1, "y_field": col2, "priority": 1
            })

    # Suggest time-series charts if datetime and numeric columns exist
    if datetimes and numerics:
        dt_col = datetimes[0] # Use the first datetime column found
        # Plot against the top 3 most significant numerics (excluding identifiers)
        top_numerics = sorted([p for p in numerics if p.role != 'identifier'], key=_calculate_kpi_score, reverse=True)
        for num_col in top_numerics[:3]:
            chart_specs.append({
                "id": f"timeseries_{dt_col.name}_{num_col.name}", 
                "title": f"Trend of {num_col.name} over {dt_col.name}", 
                "chart_type": "line", "intent": "time_series", "x_field": dt_col.name, 
                "y_field": num_col.name, "priority": 0 # Highest priority
            })

    # --- 3. Prioritize and truncate the list ---
    sorted_charts = sorted(chart_specs, key=lambda x: x.get('priority', 99))
    
    logger.info(f"Layer 4: Selected {len(sorted_charts)} potential charts, returning top {max_charts}.")
    return sorted_charts[:max_charts]