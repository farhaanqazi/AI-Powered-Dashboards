import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from scipy.stats import chi2_contingency
import math
import re

logger = logging.getLogger(__name__)

def _analyze_column_for_viz(series: pd.Series, role: str, semantic_tags: List[str] = []) -> Dict[str, Any]:
    """
    Analyze a column's characteristics for appropriate visualization selection.
    
    Args:
        series: The pandas Series to analyze
        role: The semantic role of the column ('numeric', 'categorical', 'datetime', etc.)
        semantic_tags: List of semantic tags associated with the column
        
    Returns:
        Dictionary with analysis results including suitability for different chart types
    """
    n_total = len(series)
    if n_total == 0:
        return {"suitable_charts": [], "stats": {}, "confidence": 0.0}
    
    n_unique = series.nunique(dropna=True)
    unique_ratio = n_unique / n_total if n_total > 0 else 0.0
    missing_ratio = series.isna().sum() / n_total if n_total > 0 else 0.0
    
    analysis = {
        "role": role,
        "semantic_tags": semantic_tags,
        "n_total": n_total,
        "n_unique": n_unique,
        "unique_ratio": unique_ratio,
        "missing_ratio": missing_ratio,
        "suitable_charts": [],
        "stats": {},
        "confidence": 0.7  # Default medium confidence
    }
    
    # Add role-specific statistics
    if role == "numeric":
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        if len(numeric_series) > 0:
            analysis["stats"] = {
                "min": float(numeric_series.min()) if pd.notna(numeric_series.min()) else None,
                "max": float(numeric_series.max()) if pd.notna(numeric_series.max()) else None,
                "mean": float(numeric_series.mean()) if pd.notna(numeric_series.mean()) else None,
                "std": float(numeric_series.std()) if len(numeric_series) > 1 and pd.notna(numeric_series.std()) else 0.0,
                "median": float(numeric_series.median()) if pd.notna(numeric_series.median()) else None,
                "q25": float(numeric_series.quantile(0.25)) if len(numeric_series) > 0 and pd.notna(numeric_series.quantile(0.25)) else None,
                "q75": float(numeric_series.quantile(0.75)) if len(numeric_series) > 0 and pd.notna(numeric_series.quantile(0.75)) else None,
                "skewness": float(numeric_series.skew()) if len(numeric_series) > 2 and pd.notna(numeric_series.skew()) else 0.0,
                "kurtosis": float(numeric_series.kurtosis()) if len(numeric_series) > 3 and pd.notna(numeric_series.kurtosis()) else 0.0
            }
            # Determine suitable charts for numeric data based on characteristics
            std_val = analysis["stats"]["std"]
            if std_val > 0.001:  # Has meaningful variance
                analysis["suitable_charts"].extend(["histogram", "box_plot", "scatter", "line"])
                analysis["confidence"] = 0.9  # High confidence for well-behaved numeric data
            else:
                analysis["suitable_charts"].append("summary_stat")  # No meaningful variance
                analysis["confidence"] = 0.3  # Low confidence for constant data
    elif role in ["categorical", "text"]:
        # Determine suitable charts for categorical data based on cardinality
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
    
    # Apply confidence adjustments based on data quality
    if missing_ratio > 0.5:
        analysis["confidence"] *= 0.6  # Reduce confidence for high missingness
    elif missing_ratio > 0.2:
        analysis["confidence"] *= 0.8  # Moderate reduction for moderate missingness

    if unique_ratio > 0.98:  # Potential identifier
        analysis["confidence"] *= 0.4  # Significant reduction if likely an identifier
        
    analysis["confidence"] = max(0.1, min(1.0, analysis["confidence"]))  # Clamp to [0.1, 1.0]
    
    return analysis


def _is_likely_identifier_with_confidence(s: pd.Series, name: str) -> Tuple[bool, str, float]:
    """
    Check if a series is likely an identifier with confidence scoring.
    
    Args:
        s: The pandas Series to analyze
        name: The column name
        
    Returns:
        Tuple of (is_identifier, detection_method, confidence_score)
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


def _is_likely_identifier_with_confidence(s: pd.Series, name: str) -> Tuple[bool, str, float]:
    """
    Check if a series is likely an identifier with confidence scoring.

    Args:
        s: The pandas Series to analyze
        name: The column name

    Returns:
        Tuple of (is_identifier, detection_method, confidence_score)
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


def _is_likely_identifier(s: pd.Series, name: str) -> bool:
    """
    Simplified function to check if a series is likely an identifier.
    Uses the confidence-based function internally but returns only a boolean.
    """
    is_id, _, confidence = _is_likely_identifier_with_confidence(s, name)
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
                avg_parts = sample_with_delim.str.split(best_delimiter).apply(len).mean()
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
        return abs(correlation) > 0.1 if not pd.isna(correlation) else False
    except Exception:
        return False


def _suggest_appropriate_charts_for_columns(df: pd.DataFrame, dataset_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Suggest charts based on column analysis and semantic understanding with identifier filtering.
    
    Args:
        df: The input DataFrame
        dataset_profile: Profile containing column information including roles and semantic tags
        
    Returns:
        List of chart specifications with appropriate chart types and fields
    """
    charts = []
    columns = dataset_profile["columns"]
    
    # Group columns by their roles for appropriate chart suggestions
    numeric_cols = []
    categorical_cols = []
    datetime_cols = []
    identifier_cols = []
    text_cols = []

    for col in columns:
        role = col["role"]
        unique_count = col.get("unique_count", 0)
        
        # First check if it looks like an identifier regardless of role
        series_sample = df[col["name"]].head(100)  # Sample for checking
        is_id, _, id_conf = _is_likely_identifier_with_confidence(series_sample, col["name"])

        if is_id and id_conf > 0.6:  # Higher threshold to be more conservative
            identifier_cols.append(col)
        elif role == "numeric":
            # Verify that it's truly meaningful numeric data (not an ID disguised as a number)
            series = df[col["name"]]
            unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
            if unique_ratio < 0.95:  # Exclude columns that are almost all unique (likely IDs)
                numeric_cols.append(col)
        elif role in ["categorical", "text"]:
            # Check if it's a multi-value text field
            series = df[col["name"]]
            is_multi, delimiter, multi_conf = _is_multi_value_field(series)
            
            if is_multi and multi_conf > 0.3:
                # Multi-value fields should be treated specially if not too many unique combinations
                if unique_count <= 50:
                    categorical_cols.append(col)
            elif unique_count <= 50:  # Low cardinality categorical/text
                categorical_cols.append(col)
            else:
                # High cardinality - treat as text
                text_cols.append(col)
        elif role == "datetime":
            datetime_cols.append(col)
        else:
            # Default to treating as text if unknown
            text_cols.append(col)
    
    logger.info(f"Column classification: {len(numeric_cols)} numeric, "
                f"{len(categorical_cols)} categorical, "
                f"{len(datetime_cols)} datetime, "
                f"{len(identifier_cols)} identifiers, "
                f"{len(text_cols)} text")
    
    # 1. Distribution charts for meaningful numeric variables (excluding identifiers)
    for col in numeric_cols:
        col_name = col["name"]
        series = df[col_name]
        
        # Skip if this looks like an identifier (double-check)
        if any(id_col["name"] == col_name for id_col in identifier_cols):
            continue
            
        # Skip if constant or nearly constant
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        if len(numeric_series) > 1:
            std_val = numeric_series.std()
            if pd.notna(std_val) and std_val < 0.001:  # Nearly constant
                continue
        
        # Suggest distribution chart (histogram) if meaningful
        if len(numeric_series) > 5 and numeric_series.std() > 0.001:
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
        if len(numeric_series) > 10 and numeric_series.std() > 0.001:
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
                
            datetime_series = pd.to_datetime(df[dt_name], errors='coerce').dropna()
            numeric_series = pd.to_numeric(df[num_name], errors='coerce')
            
            # Align the series
            aligned = pd.concat([datetime_series, numeric_series], axis=1).dropna()
            
            if len(aligned) > 2:  # Need at least 3 points for meaningful time series
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
        series = df[col_name]
        
        # Skip if this looks like an identifier
        if any(id_col["name"] == col_name for id_col in identifier_cols):
            continue
            
        # Get value counts for the series
        value_counts = series.value_counts(dropna=True)
        
        # Suggest bar chart if not too many categories
        if len(value_counts) > 1 and len(value_counts) <= 20:  # Not too many categories for a readable bar chart
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
        if len(value_counts) > 1 and len(value_counts) <= 10:
            charts.append({
                "id": f"pie_{col_name}",
                "title": f"Distribution of {col_name.replace('_', ' ').title()}",
                "chart_type": "pie",
                "intent": "category_distribution",
                "x_field": col_name,
                "y_field": None,
                "agg_func": "count",
                "priority": 2
            })
    
    # 4. Scatter plots for meaningful numeric-numeric relationships (excluding identifiers)
    numeric_non_id = [col for col in numeric_cols if not any(id_col["name"] == col["name"] for id_col in identifier_cols)]

    for i, col1 in enumerate(numeric_non_id):
        for j, col2 in enumerate(numeric_non_id[i+1:], i+1):  # Avoid duplicate pairs
            series1 = df[col1["name"]]
            series2 = df[col2["name"]]

            # Check if this correlation would be meaningful before suggesting scatter
            if _is_meaningful_for_correlation(series1, series2, df, col1["name"], col2["name"]):
                charts.append({
                    "id": f"scatter_{col1['name']}_{col2['name']}",
                    "title": f"{col1['name'].replace('_', ' ').title()} vs {col2['name'].replace('_', ' ').title()}",
                    "chart_type": "scatter",
                    "intent": "correlation",
                    "x_field": col2["name"],
                    "y_field": col1["name"],
                    "agg_func": None,
                    "priority": 3
                })

    # 5. Correlation heatmap only for meaningful numeric columns (excluding identifiers)
    meaningful_numeric_cols = [col for col in numeric_cols
                              if not any(id_col["name"] == col["name"] for id_col in identifier_cols)]
    
    if len(meaningful_numeric_cols) >= 2:
        # Extract data for only meaningful numeric columns
        column_names = [col["name"] for col in meaningful_numeric_cols]
        numeric_data = df[column_names]
        
        # Only include truly numeric columns (filter out any remaining non-numeric data)
        numeric_data = numeric_data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) >= 2:
            # Compute correlation matrix
            corr_matrix = numeric_data.corr()
            
            # Only add heatmap if we have meaningful correlations (>0.1 absolute value)
            has_meaningful_corrs = False
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if pd.notna(corr_val) and abs(corr_val) > 0.1:
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

            series_cat = df[cat_name]
            series_num = pd.to_numeric(df[num_name], errors='coerce')

            # Create grouped statistics
            grouped = pd.concat([series_cat, series_num], axis=1).dropna()

            if len(grouped) > 5 and grouped[cat_name].nunique() >= 2:  # Enough data and categories
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
                if grouped[cat_name].nunique() <= 10:
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

        logger.info(f"Suggested {len(suggested_charts)} meaningful charts")

        return suggested_charts

    except Exception as e:
        logger.error(f"Error in chart suggestion: {e}")
        import traceback
        traceback.print_exc()

        # Fallback: return minimal charts based on simple heuristics
        fallback_charts = []

        # At minimum, suggest one histogram for a meaningful numeric column
        for col in dataset_profile.get("columns", []):
            if col.get("role") == "numeric":
                col_name = col["name"]
                series = df[col_name]
                
                # Only suggest if it's not an identifier
                if not _is_likely_identifier(series, col_name):
                    numeric_data = pd.to_numeric(series, errors='coerce').dropna()
                    if len(numeric_data) > 5 and numeric_data.std() > 0.001:
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

        logger.info(f"Fallback: generated {len(fallback_charts)} charts")
        return fallback_charts