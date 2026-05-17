"""
EDA Analysis Module

This module performs Exploratory Data Analysis to generate insights about the dataset,
including key indicators, patterns, relationships, and recommendations.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from scipy.stats import pearsonr
from collections import Counter
from src.analysis.data_structures import EnrichedProfile

logger = logging.getLogger(__name__)

def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    """
    Safely get a field from either a dict-like object or an attribute-based object.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def run_eda_analysis(df: pd.DataFrame, enriched_profiles: Dict[str, EnrichedProfile], relational_insights: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Performs comprehensive EDA analysis on the dataset.

    Args:
        df: Input DataFrame
        enriched_profiles: Profiles of columns with semantic roles
        relational_insights: Insights about relationships between columns

    Returns:
        Dictionary containing EDA insights
    """
    errors: List[str] = []

    try:
        # Generate key indicators
        try:
            key_indicators = _generate_key_indicators(df, enriched_profiles)
        except Exception as e:
            errors.append(f"key_indicators_failed: {e}")
            key_indicators = []

        # Identify patterns and relationships
        try:
            patterns_and_relationships = _identify_patterns_and_relationships(df, enriched_profiles, relational_insights)
        except Exception as e:
            errors.append(f"patterns_and_relationships_failed: {e}")
            patterns_and_relationships = {}

        # Generate potential use cases
        try:
            use_cases = _generate_use_cases(enriched_profiles)
        except Exception as e:
            errors.append(f"use_cases_failed: {e}")
            use_cases = []

        # Generate recommendations
        try:
            recommendations = _generate_recommendations(enriched_profiles, patterns_and_relationships)
        except Exception as e:
            errors.append(f"recommendations_failed: {e}")
            recommendations = []

        # Generate critical totals if applicable
        try:
            critical_totals = _generate_critical_totals(df, enriched_profiles)
        except Exception as e:
            errors.append(f"critical_totals_failed: {e}")
            critical_totals = {}

        eda_summary = {
            "key_indicators": key_indicators,
            "patterns_and_relationships": patterns_and_relationships,
            "use_cases": use_cases,
            "recommendations": recommendations,
            "critical_totals": critical_totals,
            "errors": errors,
        }

        logger.info(f"EDA analysis completed with {len(key_indicators)} key indicators, "
                   f"{len(patterns_and_relationships.get('correlations', []))} correlations, "
                   f"and {len(use_cases)} use cases identified.")

        if errors:
            logger.warning(f"EDA analysis completed with {len(errors)} errors: {errors}")

        return eda_summary

    except Exception as e:
        logger.error(f"Error during EDA analysis: {e}")
        return {
            "key_indicators": [],
            "patterns_and_relationships": {},
            "use_cases": [],
            "recommendations": [],
            "critical_totals": {},
            "errors": [f"eda_fatal_error: {e}"],
        }


def _generate_key_indicators(df: pd.DataFrame, enriched_profiles: Dict[str, EnrichedProfile]) -> List[Dict[str, Any]]:
    """
    Generate key indicators based on the dataset characteristics.
    """
    indicators = []
    
    # Add indicators for numeric columns
    for col_name, profile in enriched_profiles.items():
        if profile.role == 'numeric':
            stats = profile.stats
            if 'mean' in stats:
                indicators.append({
                    "indicator": f"Average {col_name}",
                    "description": f"The average value of {col_name} is {stats['mean']:.2f}",
                    "value": stats['mean'],
                    "type": "average"
                })

            if 'std' in stats and stats['std'] > 0:
                indicators.append({
                    "indicator": f"Variability in {col_name}",
                    "description": f"{col_name} has a standard deviation of {stats['std']:.2f}",
                    "value": stats['std'],
                    "type": "variability"
                })
    
    # Add dataset-level indicators
    indicators.extend([
        {
            "indicator": "Dataset Size",
            "description": f"Dataset contains {len(df)} rows and {len(enriched_profiles)} columns",
            "value": f"{len(df)} rows × {len(enriched_profiles)} cols",
            "type": "dataset_size"
        },
        {
            "indicator": "Most Frequent Category",
            "description": _get_most_frequent_category_info(enriched_profiles),
            "value": _get_most_frequent_value(enriched_profiles),
            "type": "dominant_category"
        }
    ])

    return indicators


def _get_most_frequent_category_info(enriched_profiles: Dict[str, EnrichedProfile]) -> str:
    """Helper to get info about the most frequent category."""
    for col_name, profile in enriched_profiles.items():
        if profile.role == 'categorical' and profile.top_categories:
            top_cat = profile.top_categories[0]
            return f"The most frequent category in {col_name} is '{top_cat['value']}' with {top_cat['count']} occurrences"
    return "No categorical columns found"


def _get_most_frequent_value(enriched_profiles: Dict[str, EnrichedProfile]) -> str:
    """Helper to get the most frequent value."""
    for col_name, profile in enriched_profiles.items():
        if profile.role == 'categorical' and profile.top_categories:
            top_cat = profile.top_categories[0]
            return f"{top_cat['value']} ({top_cat['count']} times)"
    return "N/A"


def _identify_patterns_and_relationships(df: pd.DataFrame, enriched_profiles: Dict[str, EnrichedProfile],
                                       relational_insights: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Identify patterns and relationships in the data.
    """
    patterns = {
        "correlations": [],
        "trends": [],
        "outliers": [],
        "anomalies": []
    }

    # Extract correlations from relational insights
    for insight in relational_insights:
        insight_type = _get_field(insight, "type")
        if insight_type == 'correlation':
            details = _get_field(insight, "details", {}) or {}
            columns = _get_field(insight, "columns", []) or []
            if len(columns) >= 2 and 'correlation_coefficient' in details:
                patterns['correlations'].append({
                    "variable1": columns[0],
                    "variable2": columns[1],
                    "correlation": details.get('correlation_coefficient'),
                    "p_value": details.get('p_value', None),
                    "strength": details.get('strength', 'moderate')
                })

    # Identify outliers in numeric columns
    for col_name, profile in enriched_profiles.items():
        if profile.role == 'numeric':
            # A column's ROLE can be 'numeric' (heuristic or AI-arbitrated)
            # while its raw values are still strings. Quantiles on an object
            # series return strings → `Q3 - Q1` is `str - str` and the whole
            # patterns block dies. Coerce defensively; skip if not real numbers.
            series = pd.to_numeric(df[col_name], errors='coerce').dropna()
            if len(series) > 10:  # Need sufficient data points
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = series[(series < lower_bound) | (series > upper_bound)]
                if len(outliers) > 0:
                    patterns['outliers'].append({
                        "column": col_name,
                        "outlier_count": len(outliers),
                        "outlier_percentage": (len(outliers) / len(series)) * 100
                    })

    return patterns


def _generate_use_cases(enriched_profiles: Dict[str, EnrichedProfile]) -> List[Dict[str, Any]]:
    """
    Generate potential use cases based on column roles.
    """
    use_cases = []

    # Identify potential business use cases based on column types
    has_numeric = any(p.role == 'numeric' for p in enriched_profiles.values())
    has_categorical = any(p.role == 'categorical' for p in enriched_profiles.values())
    has_datetime = any(p.role == 'datetime' for p in enriched_profiles.values())

    if has_numeric and has_categorical:
        use_cases.append({
            "use_case": "Segmentation Analysis",
            "description": "Analyze how numeric metrics vary across different categories",
            "key_inputs": [name for name, profile in enriched_profiles.items()
                          if profile.role in ['numeric', 'categorical']]
        })

    if has_datetime and has_numeric:
        use_cases.append({
            "use_case": "Time Series Analysis",
            "description": "Track numeric metrics over time to identify trends",
            "key_inputs": [name for name, profile in enriched_profiles.items()
                          if profile.role in ['datetime', 'numeric']]
        })

    if has_numeric:
        use_cases.append({
            "use_case": "Performance Metrics Dashboard",
            "description": "Monitor key numeric metrics and KPIs",
            "key_inputs": [name for name, profile in enriched_profiles.items()
                          if profile.role == 'numeric']
        })

    return use_cases


def _generate_recommendations(enriched_profiles: Dict[str, EnrichedProfile], patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate recommendations based on the analysis.
    """
    recommendations = []

    # Recommend visualizations based on column types
    numeric_cols = [name for name, profile in enriched_profiles.items()
                   if profile.role == 'numeric']
    categorical_cols = [name for name, profile in enriched_profiles.items()
                       if profile.role == 'categorical']

    if len(numeric_cols) >= 2:
        recommendations.append({
            "title": "Correlation Analysis",
            "description": f"Consider analyzing correlations between numeric columns: {', '.join(numeric_cols[:3])}",
            "priority": "high"
        })

    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        recommendations.append({
            "title": "Group Comparisons",
            "description": f"Compare numeric metrics across categories: {categorical_cols[0] if categorical_cols else 'N/A'}",
            "priority": "medium"
        })

    # Add recommendation if there are many correlations
    if len(patterns.get('correlations', [])) > 3:
        recommendations.append({
            "title": "Deep Dive into Relationships",
            "description": "Several strong correlations detected - investigate causation vs correlation",
            "priority": "high"
        })

    return recommendations


def _generate_critical_totals(df: pd.DataFrame, enriched_profiles: Dict[str, EnrichedProfile]) -> Dict[str, Any]:
    """
    Generate critical totals that might be important for business metrics.
    """
    critical_totals = {}

    # Look for columns that might represent monetary values
    for col_name, profile in enriched_profiles.items():
        if profile.role == 'numeric':
            # Check if column name suggests it's a monetary value
            name_lower = col_name.lower()
            monetary_indicators = ['amount', 'revenue', 'cost', 'price', 'fee', 'charge', 'payment', 'income', 'expense', 'profit']

            if any(indicator in name_lower for indicator in monetary_indicators):
                # role can be 'numeric' on a string-valued column; coerce
                # before summing or float() blows up (same class of bug as
                # the outlier IQR path).
                numeric = pd.to_numeric(df[col_name], errors='coerce').dropna()
                if numeric.empty:
                    continue
                critical_totals[f"total_{col_name}"] = float(numeric.sum())

    return critical_totals


def create_key_indicators_bar_chart(key_indicators: List[Dict[str, Any]], top_n: int = 10) -> Dict[str, Any]:
    """
    Create a simple chart representation for key indicators (for frontend display)
    """
    if not key_indicators:
        return {"type": "bar", "data": [], "layout": {}}

    # Get top N indicators
    top_indicators = key_indicators[:top_n]

    names = [ind['indicator'] for ind in top_indicators]
    values = [ind.get('value', 0) for ind in top_indicators]
    types = [ind.get('type', 'other') for ind in top_indicators]

    # Create a simple representation that can be used by frontend
    chart_data = {
        "type": "bar",
        "x": names,
        "y": values,
        "labels": names,
        "values": values,
        "types": types
    }

    return chart_data
