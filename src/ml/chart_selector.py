import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from scipy.stats import chi2_contingency
import math
import re
from src.utils.identifier_detector import is_likely_identifier, is_likely_identifier_with_confidence as centralized_is_likely_identifier_with_confidence

logger = logging.getLogger(__name__)

def _analyze_column_for_viz_from_profile(col_profile: Dict[str, Any], semantic_tags: List[str] = []) -> Dict[str, Any]:
    """
    Analyze a column's characteristics for appropriate visualization selection using dataset profile.
    This function avoids re-analyzing the raw series data.

    Args:
        col_profile: The column profile from dataset_profile
        semantic_tags: List of semantic tags associated with the column

    Returns:
        Dictionary with analysis results including suitability for different chart types
    """
    # Get information from the column profile
    role = col_profile.get("role", "unknown")
    
    # Safely get stats, n_total, n_unique, and missing_count
    stats = col_profile.get("stats") or {}
    n_total = stats.get("count", 0)
    n_unique = col_profile.get("unique_count", 0)
    missing_count = col_profile.get("missing_count", 0)

    unique_ratio = n_unique / n_total if n_total > 0 else 0.0
    missing_ratio = missing_count / n_total if n_total > 0 else 0.0

    # Get statistics from profile
    stats = col_profile.get("stats", {})

    analysis = {
        "role": role,
        "semantic_tags": semantic_tags,
        "n_total": n_total,
        "n_unique": n_unique,
        "unique_ratio": unique_ratio,
        "missing_ratio": missing_ratio,
        "suitable_charts": [],
        "stats": stats,
        "confidence": 0.7  # Default medium confidence
    }

    # Add role-specific visualization logic based on profile information
    if role == "numeric":
        # Use statistics from the profile
        std_val = stats.get("std", 0.0) if stats else 0.0
        mean_val = stats.get("mean", None)

        # Determine suitable charts for numeric data based on profile characteristics
        if std_val > 0.001 and mean_val is not None:  # Has meaningful variance
            analysis["suitable_charts"].extend(["histogram", "box_plot", "scatter", "line"])
            analysis["confidence"] = 0.9  # High confidence for well-behaved numeric data
        else:
            analysis["suitable_charts"].append("summary_stat")  # No meaningful variance
            analysis["confidence"] = 0.3  # Low confidence for constant data
    elif role in ["categorical", "text"]:
        # Determine suitable charts for categorical data based on cardinality from profile
        if n_unique <= 2:
            # Binary categorical - pie, bar charts are suitable
            analysis["suitable_charts"].extend(["bar", "pie", "donut"])
            analysis["confidence"] = 0.8  # High confidence for binary categorical
        elif n_unique <= 10:
            # Low cardinality categorical - bar is excellent, pie is okay
            analysis["suitable_charts"].extend(["bar", "pie", "stacked_bar"])
            analysis["confidence"] = 0.85  # High confidence for low-cardinality categorical
        elif n_unique <= 50:
            # Medium cardinality - bar charts are still appropriate
            analysis["suitable_charts"].extend(["bar", "horizontal_bar", "top_categories"])
            analysis["confidence"] = 0.7  # Medium-high confidence
        else:
            # High cardinality - only show top categories or summary, avoid pie charts
            analysis["suitable_charts"].extend(["top_categories", "summary_stat"])
            analysis["confidence"] = 0.4  # Lower confidence due to high cardinality
    elif role == "datetime":
        analysis["suitable_charts"].extend(["line", "time_series", "calendar_heatmap", "bar"])
        analysis["confidence"] = 0.9  # High confidence for datetime (good for time series)

    # Apply confidence adjustments based on data quality from profile
    if missing_ratio > 0.5:
        analysis["confidence"] *= 0.6  # Reduce confidence for high missingness
    elif missing_ratio > 0.2:
        analysis["confidence"] *= 0.8  # Moderate reduction for moderate missingness

    if unique_ratio > 0.98:  # Potential identifier based on profile
        analysis["confidence"] *= 0.4  # Significant reduction if likely an identifier

    analysis["confidence"] = max(0.1, min(1.0, analysis["confidence"]))  # Clamp to [0.1, 1.0]

    return analysis


def generate_chart_signature(chart_spec: dict) -> str:
    """
    Generate a unique signature for a chart specification based on its key properties.

    Args:
        chart_spec: A dictionary representing a chart specification

    Returns:
        A unique string signature for the chart
    """
    chart_type = chart_spec.get('chart_type', '')
    x_field = chart_spec.get('x_field', '')
    y_field = chart_spec.get('y_field', '')
    agg_func = chart_spec.get('agg_func', '')

    # Create a hashable signature string
    signature = f"{chart_type}:{x_field}:{y_field}:{agg_func}"
    return signature


def deduplicate_charts(charts: List[dict]) -> List[dict]:
    """
    Remove semantically duplicate charts from the list based on their signatures.

    Args:
        charts: List of chart specifications

    Returns:
        List of unique chart specifications
    """
    seen_signatures = set()
    unique_charts = []

    for chart in charts:
        signature = generate_chart_signature(chart)
        if signature not in seen_signatures:
            seen_signatures.add(signature)
            unique_charts.append(chart)
        else:
            logger.debug(f"Removed duplicate chart with signature: {signature}")

    return unique_charts


def _is_likely_identifier_with_confidence(s: pd.Series, name: str) -> Tuple[bool, str, float]:
    """
    Check if a series is likely an identifier with confidence scoring.

    Args:
        s: The pandas Series to analyze
        name: The column name

    Returns:
        Tuple of (is_identifier, detection_method, confidence_score)
    """
    # Use the centralized identifier detector with confidence
    is_id, method, confidence = centralized_is_likely_identifier_with_confidence(s, name)
    return is_id, method, confidence


def _is_likely_identifier(s: pd.Series, name: str) -> bool:
    """
    Simplified function to check if a series is likely an identifier.
    Uses the confidence-based function internally but returns only a boolean.
    """
    is_id, _, _ = _is_likely_identifier_with_confidence(s, name)
    return is_id


def _is_multi_value_field(series: pd.Series, delimiter_chars: List[str] = [',', ';', '|', '/', ' | ']) -> Tuple[bool, str, float]:
    """
    Detect if a field contains multiple values separated by delimiters.
    
    Args:
        series: The pandas Series to analyze
        delimiter_chars: List of potential delimiter characters
        
    Returns:
        Tuple of (is_multi_value, primary_delimiter, confidence_score)
    """
    if series.dtype != 'object' and not str(series.dtype).startswith('string'):
        return False, "", 0.0

    n_total = len(series)
    if n_total == 0:
        return False, "", 0.0

    sample_size = min(100, n_total)
    sample_values = series.dropna().head(sample_size).astype(str)

    if len(sample_values) == 0:
        return False, "", 0.0

    delimiter_scores = {}

    for delim in delimiter_chars:
        # Count how many values contain the delimiter
        contains_delim = sample_values.str.contains(delim, na=False, regex=False)
        delimiter_scores[delim] = contains_delim.sum() / len(sample_values)

    if delimiter_scores:
        # Get the delimiter with highest ratio
        best_delimiter = max(delimiter_scores, key=delimiter_scores.get)
        best_ratio = delimiter_scores[best_delimiter]
        
        # Additional validation: see if splitting creates multiple meaningful parts
        if best_ratio > 0.1:  # At least 10% of values contain the delimiter
            sample_with_delim = sample_values[sample_values.str.contains(best_delimiter, na=False, regex=False)]
            if len(sample_with_delim) > 0:
                split_result = sample_with_delim.str.split(best_delimiter).apply(len)
                # Ensure split_result is a pandas Series before calling .mean()
                if hasattr(split_result, 'mean'):
                    avg_parts = split_result.mean()
                else:
                    # Convert to Series if needed
                    split_series = pd.Series(split_result) if not isinstance(split_result, pd.Series) else split_result
                    avg_parts = split_series.mean()

                if avg_parts > 1.5:  # On average more than 1 part after splitting
                    # Confidence is based on both ratio and average number of parts
                    confidence = min(1.0, best_ratio * avg_parts * 0.7)
                    return True, best_delimiter, confidence

    return False, "", 0.0


def _is_meaningful_for_correlation(series1: pd.Series, series2: pd.Series,
                                 df: pd.DataFrame, col1: str, col2: str) -> bool:
    """
    Determine if two columns are meaningful to correlate together.

    Args:
        series1, series2: The two series to evaluate
        df: The dataframe they come from
        col1, col2: Their respective column names

    Returns:
        True if correlation between these columns would be meaningful
    """
    # If either series is likely an identifier, correlation is not meaningful
    if _is_likely_identifier(series1, col1) or _is_likely_identifier(series2, col2):
        return False

    # Check if both are numeric (requirement for standard correlation)
    if not (pd.api.types.is_numeric_dtype(series1) and pd.api.types.is_numeric_dtype(series2)):
        return False

    # Check if both have sufficient variance to be meaningful
    clean_s1 = pd.to_numeric(series1, errors='coerce').dropna()
    clean_s2 = pd.to_numeric(series2, errors='coerce').dropna()

    if len(clean_s1) < 3 or len(clean_s2) < 3:
        return False  # Need at least 3 points for meaningful correlation

    std1 = clean_s1.std()
    std2 = clean_s2.std()

    if pd.isna(std1) or std1 < 0.001 or pd.isna(std2) or std2 < 0.001:
        return False  # One or both have very little variance

    # Align indices to ensure same data points
    aligned_df = pd.concat([clean_s1, clean_s2], axis=1).dropna()

    if len(aligned_df) < 3:  # Need at least 3 aligned points for meaningful correlation
        return False

    s1_aligned = aligned_df.iloc[:, 0]
    s2_aligned = aligned_df.iloc[:, 1]

    if len(s1_aligned) == 0 or len(s2_aligned) == 0 or len(s1_aligned) != len(s2_aligned):
        return False

    try:
        correlation = s1_aligned.corr(s2_aligned)
        # Only consider correlations meaningful if abs(correlation) > 0.1
        # and correlation is a valid number
        return abs(correlation) > 0.1 and not pd.isna(correlation) and np.isfinite(correlation)
    except Exception as e:
        logger.debug(f"Error calculating correlation between {col1} and {col2}: {e}")
        return False


def _suggest_appropriate_charts_for_columns(df: pd.DataFrame, dataset_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Suggest charts based on dataset profile information, avoiding re-analysis of raw data.

    Args:
        df: The input DataFrame (still needed for some operations)
        dataset_profile: Profile containing column information including roles and semantic tags

    Returns:
        List of chart specifications with appropriate chart types and fields
    """
    charts = []
    columns = dataset_profile.get("columns", []) # Handle case where 'columns' might be missing
    if not columns:
        logger.warning("No columns found in dataset profile for chart selection.")
        return []

    # Group columns by their roles for appropriate chart suggestions using profile data
    numeric_cols = []
    categorical_cols = []
    datetime_cols = []
    identifier_cols = []
    text_cols = []

    for col in columns:
        role = col.get("role", "unknown") # Handle case where 'role' might be missing
        unique_count = col.get("unique_count", 0)
        col_name = col.get("name", "") # Handle case where 'name' might be missing

        if not col_name:
            logger.warning(f"Column found without a name: {col}")
            continue

        # Check if this is an identifier based on profile role (SSOT)
        if role == "identifier":
            identifier_cols.append(col)
            continue

        if role == "numeric":
            # If the analyser classified it as numeric, trust it. No need to re-check unique_ratio.
            numeric_cols.append(col)
        elif role in ["categorical", "text"]:
            if unique_count <= 50:  # Low cardinality categorical/text
                categorical_cols.append(col)
            else:
                # High cardinality - treat as text
                text_cols.append(col)
        elif role == "datetime":
            datetime_cols.append(col)
        else:
            # Default to treating as text if unknown (this shouldn't happen with a good analyser)
            text_cols.append(col)

    logger.info(f"Column classification from profile: {len(numeric_cols)} numeric, "
                f"{len(categorical_cols)} categorical, "
                f"{len(datetime_cols)} datetime, "
                f"{len(identifier_cols)} identifiers, "
                f"{len(text_cols)} text")

    # 1. Distribution charts for meaningful numeric variables (excluding identifiers)
    for col in numeric_cols:
        col_name = col["name"]

        # Skip if this looks like an identifier (double-check)
        if any(id_col["name"] == col_name for id_col in identifier_cols):
            continue

        # Get stats from profile to check if it's constant or nearly constant
        stats = col.get("stats", {})
        std_val = stats.get("std", 0.0) if stats else 0.0

        # Skip if nearly constant based on profile
        if std_val < 0.001:
            continue

        # Suggest distribution chart (histogram) if meaningful based on profile
        count = stats.get("count", 0) if stats else 0
        if count > 5 and std_val > 0.001:
            charts.append({
                "id": f"dist_{col_name}",
                "title": f"Distribution of {col_name.replace('_', ' ').title()}",
                "chart_type": "histogram",
                "intent": "distribution",
                "x_field": col_name,
                "y_field": None,
                "agg_func": "count",
                "priority": 1
            })

        # Suggest box plot if there are enough data points and meaningful variance
        if count > 10 and std_val > 0.001:
            charts.append({
                "id": f"box_{col_name}",
                "title": f"Box Plot of {col_name.replace('_', ' ').title()}",
                "chart_type": "box",
                "intent": "box_plot",
                "x_field": None,
                "y_field": col_name,
                "agg_func": None,
                "priority": 2
            })

    # 2. Time series charts for datetime + numeric combinations (excluding identifiers)
    for dt_col in datetime_cols:
        for num_col in numeric_cols:  # Only numeric non-identifier columns
            dt_name = dt_col["name"]
            num_name = num_col["name"]

            # Skip if either is an identifier
            if any(id_col["name"] == dt_name for id_col in identifier_cols) or \
               any(id_col["name"] == num_name for id_col in identifier_cols):
                continue

            # Get stats from profile
            dt_count = dt_col.get("stats", {}).get("count", 0)
            num_count = num_col.get("stats", {}).get("count", 0)

            if dt_count > 2 and num_count > 2:  # Need at least 3 points for meaningful time series
                charts.append({
                    "id": f"ts_{dt_name}_{num_name}",
                    "title": f"Trend of {num_name.replace('_', ' ').title()} Over {dt_name.replace('_', ' ').title()}",
                    "chart_type": "line",
                    "intent": "time_series",
                    "x_field": dt_name,
                    "y_field": num_name,
                    "agg_func": "mean",  # Use mean aggregation for potential duplicate dates
                    "priority": 1
                })

    # 3. Categorical charts for low-cardinality categorical variables (excluding identifiers)
    for col in categorical_cols:
        col_name = col["name"]

        # Skip if this looks like an identifier
        if any(id_col["name"] == col_name for id_col in identifier_cols):
            continue

        # Get value counts from profile
        unique_count = col.get("unique_count", 0)
        missing_count = col.get("missing_count", 0)
        total_count = col.get("stats", {}).get("count", 0)
        valid_count = total_count - missing_count

        # Suggest bar chart if not too many categories
        if valid_count > 1 and unique_count <= 20:  # Not too many categories for a readable bar chart
            charts.append({
                "id": f"cat_{col_name}",
                "title": f"Count of {col_name.replace('_', ' ').title()}",
                "chart_type": "bar",
                "intent": "category_count",
                "x_field": col_name,
                "y_field": None,
                "agg_func": "count",
                "priority": 1
            })

        # Suggest pie chart if not too many categories (max 10 for readability)
        if valid_count > 1 and unique_count <= 10:
            charts.append({
                "id": f"pie_{col_name}",
                "title": f"Distribution of {col_name.replace('_', ' ').title()}",
                "chart_type": "pie",
                "intent": "category_pie",
                "x_field": col_name,
                "y_field": None,
                "agg_func": "count",
                "priority": 2
            })

    # 4. Scatter plots for meaningful numeric-numeric relationships (excluding identifiers)
    numeric_non_id = [col for col in numeric_cols if not any(id_col["name"] == col["name"] for id_col in identifier_cols)]

    for i, col1 in enumerate(numeric_non_id):
        for j, col2 in enumerate(numeric_non_id[i+1:], i+1):  # Avoid duplicate pairs
            # Get stats from profile to check if correlation would be meaningful
            stats1 = col1.get("stats", {})
            stats2 = col2.get("stats", {})

            std1 = stats1.get("std", 0.0) if stats1 else 0.0
            std2 = stats2.get("std", 0.0) if stats2 else 0.0
            count1 = stats1.get("count", 0) if stats1 else 0
            count2 = stats2.get("count", 0) if stats2 else 0

            # Only suggest scatter if both have meaningful variance and sufficient data
            if std1 > 0.001 and std2 > 0.001 and count1 > 2 and count2 > 2:
                charts.append({
                    "id": f"scatter_{col1['name']}_{col2['name']}",
                    "title": f"{col1['name'].replace('_', ' ').title()} vs {col2['name'].replace('_', ' ').title()}",
                    "chart_type": "scatter",
                    "intent": "scatter",
                    "x_field": col2["name"],
                    "y_field": col1["name"],
                    "agg_func": None,
                    "priority": 3
                })

    # 5. Correlation heatmap only for meaningful numeric columns (excluding identifiers)
    meaningful_numeric_cols = [col for col in numeric_cols
                              if not any(id_col["name"] == col["name"] for id_col in identifier_cols)]

    if len(meaningful_numeric_cols) >= 2:
        # Check if we have meaningful correlations (>0.1 absolute value) based on profile
        has_meaningful_corrs = False
        for i, col1 in enumerate(meaningful_numeric_cols):
            for j, col2 in enumerate(meaningful_numeric_cols[i+1:], i+1):
                # Since we can't get correlations from profile directly, we'll assume that
                # if both are numeric and not identifiers, they might have meaningful correlations
                stats1 = col1.get("stats", {})
                stats2 = col2.get("stats", {})

                std1 = stats1.get("std", 0.0) if stats1 else 0.0
                std2 = stats2.get("std", 0.0) if stats2 else 0.0
                count1 = stats1.get("count", 0) if stats1 else 0
                count2 = stats2.get("count", 0) if stats2 else 0

                if std1 > 0.001 and std2 > 0.001 and count1 > 10 and count2 > 10:
                    has_meaningful_corrs = True
                    break
            if has_meaningful_corrs:
                break

        if has_meaningful_corrs:
            charts.append({
                "id": "correlation_heatmap",
                "title": "Correlation Heatmap (Meaningful Numeric Variables)",
                "chart_type": "heatmap",
                "intent": "correlation_matrix",
                "x_field": "variables",
                "y_field": "variables",
                "agg_func": "correlation",
                "priority": 2
            })

    # 6. Group by charts: meaningful categorical vs numeric relationships (excluding identifiers)
    numeric_non_id = [col for col in numeric_cols if not any(id_col["name"] == col["name"] for id_col in identifier_cols)]

    for cat_col in categorical_cols:
        for num_col in numeric_non_id:  # Only numeric non-identifier columns
            cat_name = cat_col["name"]
            num_name = num_col["name"]

            # Skip if categorical column has too many unique values (would create unreadable chart)
            if cat_col["unique_count"] > 20:
                continue

            # Get stats from profile
            cat_count = cat_col.get("stats", {}).get("count", 0)
            num_count = num_col.get("stats", {}).get("count", 0)
            cat_unique = cat_col.get("unique_count", 0)

            if cat_count > 5 and cat_unique >= 2 and num_count > 5:  # Enough data and categories
                charts.append({
                    "id": f"group_bar_{cat_name}_{num_name}",
                    "title": f"Avg {num_name.replace('_', ' ').title()} by {cat_name.replace('_', ' ').title()}",
                    "chart_type": "bar",  # Bar chart showing average by category
                    "intent": "group_comparison",
                    "x_field": cat_name,
                    "y_field": num_name,
                    "agg_func": "mean",
                    "priority": 2
                })

                # Also suggest a box plot for distribution comparison if not too many categories
                if cat_unique <= 10:
                    charts.append({
                        "id": f"group_box_{cat_name}_{num_name}",
                        "title": f"Distribution of {num_name.replace('_', ' ').title()} by {cat_name.replace('_', ' ').title()}",
                        "chart_type": "box",
                        "intent": "distribution_by_category",
                        "x_field": cat_name,
                        "y_field": num_name,
                        "agg_func": None,
                        "priority": 3
                    })

    # Sort charts by priority (lower number means higher priority)
    charts.sort(key=lambda x: x.get("priority", 999))

    # Validation step: ensure that the suggested x_field and y_field names correspond to actual columns in the dataset_profile
    validated_charts = []
    profile_column_names = {col["name"] for col in dataset_profile.get("columns", [])}

    for chart in charts:
        x_field = chart.get("x_field")
        y_field = chart.get("y_field")

        # Check if both fields exist in the profile (if they're not None)
        x_valid = x_field is None or x_field in profile_column_names
        y_valid = y_field is None or y_field in profile_column_names

        if x_valid and y_valid:
            validated_charts.append(chart)
        else:
            logger.warning(f"Filtered out chart '{chart.get('title')}' due to non-existent fields: x='{x_field}', y='{y_field}'")

    charts = validated_charts

    # Limit number of charts to prevent overwhelming the user (max 20 charts)
    max_charts = min(20, len(charts))
    return charts[:max_charts]


def suggest_charts(df: pd.DataFrame, dataset_profile: Dict[str, Any], kpis: List[Dict[str, Any]] = []) -> List[Dict[str, Any]]:
    """
    Intelligent chart suggestion system that considers column roles, semantic tags,
    and meaningful relationships instead of naive heuristics.

    Args:
        df: Input DataFrame
        dataset_profile: Dataset profile with column roles and semantic information
        kpis: List of KPIs to consider for chart suggestions (optional)

    Returns:
        List of chart specifications tailored to the dataset's meaningful characteristics
    """
    if df.empty:
        logger.warning("Empty dataframe provided to chart selector")
        return []

    n_rows, n_cols = df.shape
    if n_rows == 0 or n_cols == 0:
        logger.warning(f"Invalid dataframe shape: {n_rows}x{n_cols}")
        return []

    logger.info(f"Suggesting charts for dataset with {n_rows} rows and {n_cols} columns")

    try:
        suggested_charts = _suggest_appropriate_charts_for_columns(df, dataset_profile)

        logger.info(f"Suggested {len(suggested_charts)} meaningful charts before deduplication")

        # Apply deduplication to remove semantically duplicate charts
        deduplicated_charts = deduplicate_charts(suggested_charts)

        logger.info(f"Returned {len(deduplicated_charts)} charts after deduplication")

        return deduplicated_charts

    except Exception as e:
        logger.error(f"Error in chart suggestion: {e}")
        import traceback
        traceback.print_exc()

        # Fallback: return minimal charts based on simple heuristics
        fallback_charts = []

        # At minimum, suggest one histogram for a meaningful numeric column based on profile
        for col in dataset_profile.get("columns", []):
            if col.get("role") == "numeric":
                col_name = col["name"]

                # Check if it's an identifier based on profile
                if col.get("role") != "identifier":
                    # Get stats from profile
                    stats = col.get("stats", {})
                    count = stats.get("count", 0) if stats else 0
                    std_val = stats.get("std", 0.0) if stats else 0.0

                    # Only suggest if it has meaningful variance
                    if count > 5 and std_val > 0.001:
                        fallback_charts.append({
                            "id": f"hist_{col_name}",
                            "title": f"Distribution of {col_name.replace('_', ' ').title()}",
                            "chart_type": "histogram",
                            "intent": "distribution",
                            "x_field": col_name,
                            "y_field": None,
                            "agg_func": "count",
                            "priority": 1
                        })
                        break  # Only add one fallback histogram

        # Apply deduplication to fallback charts as well
        deduplicated_fallback_charts = deduplicate_charts(fallback_charts)

        logger.info(f"Fallback: generated {len(deduplicated_fallback_charts)} charts after deduplication")
        return deduplicated_fallback_charts
