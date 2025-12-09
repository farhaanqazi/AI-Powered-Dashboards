import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, List, Any, Tuple, Optional
from scipy.stats import pearsonr, spearmanr
import math

logger = logging.getLogger(__name__)


def _is_likely_identifier(series: pd.Series, name: str = "") -> bool:
    """
    Determine if a series is likely an identifier based on multiple heuristics.
    """
    n_total = len(series)
    if n_total == 0:
        return False

    n_unique = series.nunique()
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
    if pd.api.types.is_numeric_dtype(series):
        numeric_vals = pd.to_numeric(series, errors='coerce').dropna()
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
    if series.dtype == 'object':
        sample = series.dropna().head(20).astype(str)
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
        # Calculate confidence based on how many keywords match
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

        return is_identifier

    return False


def _has_meaningful_variance(series: pd.Series, threshold: float = 0.001) -> bool:
    """
    Check if a numeric series has meaningful variance for correlation analysis.

    Args:
        series: The numeric series to evaluate
        threshold: Minimum standard deviation to be considered meaningful

    Returns:
        True if the series has meaningful variance
    """
    clean_series = pd.to_numeric(series, errors='coerce').dropna()
    if len(clean_series) < 3:  # Need at least 3 points for meaningful variance
        return False

    std_val = clean_series.std()
    mean_val = clean_series.mean()

    if pd.isna(std_val) or std_val < threshold:
        return False  # Very low variance

    # Additional check: if std is tiny compared to mean, it might be nearly constant
    if pd.notna(mean_val) and abs(mean_val) > threshold and std_val/abs(mean_val) < 0.01:
        return False  # Very low coefficient of variation

    return True


def _is_stable_correlation(series1: pd.Series, series2: pd.Series, stability_threshold: float = 0.1) -> bool:
    """
    Check if the correlation between two series is stable and meaningful.
    
    Args:
        series1, series2: The two series to evaluate
        stability_threshold: Minimum correlation value to be considered stable
        
    Returns:
        True if correlation is stable and above threshold
    """
    clean_s1 = pd.to_numeric(series1, errors='coerce').dropna()
    clean_s2 = pd.to_numeric(series2, errors='coerce').dropna()
    
    # Align series to have the same indices
    aligned_df = pd.concat([clean_s1, clean_s2], axis=1).dropna()
    
    if len(aligned_df) < 10:  # Need at least 10 aligned points for stable correlation
        return False
    
    s1_aligned = aligned_df.iloc[:, 0]
    s2_aligned = aligned_df.iloc[:, 1]
    
    try:
        # Calculate Pearson correlation
        pearson_corr, p_value = pearsonr(s1_aligned, s2_aligned) if len(s1_aligned) > 3 else (0.0, 1.0)
        
        # If correlation is weak, it might be spurious
        if pd.isna(pearson_corr) or abs(pearson_corr) < stability_threshold:
            return False
            
        # Only return True if both have meaningful variance
        return _has_meaningful_variance(s1_aligned) and _has_meaningful_variance(s2_aligned)
    except Exception:
        return False


def _identify_meaningful_correlations(df: pd.DataFrame, columns: List[Dict[str, Any]], 
                                    min_correlation: float = 0.1, min_variance: float = 0.001) -> List[Dict[str, Any]]:
    """
    Identify meaningful correlations between numeric columns, excluding identifiers.
    
    Args:
        df: Input DataFrame
        columns: List of column profiles including roles and names
        min_correlation: Minimum absolute correlation value to be considered meaningful
        min_variance: Minimum variance for columns to be considered meaningful
        
    Returns:
        List of meaningful correlations as dictionaries
    """
    # Filter to only numeric columns that are not likely identifiers
    numeric_cols = []
    for col_info in columns:
        if col_info.get("role") == "numeric":
            col_name = col_info["name"]
            series = df[col_name]
            
            # Check if this is likely an identifier
            if not _is_likely_identifier(series, col_name):
                # Additional validation: check for sufficient variance
                if _has_meaningful_variance(series, min_variance):
                    numeric_cols.append(col_name)
    
    if len(numeric_cols) < 2:
        logger.info(f"Not enough meaningful numeric columns for correlation analysis (found {len(numeric_cols)})")
        return []
    
    logger.info(f"Analyzing correlations for {len(numeric_cols)} meaningful numeric columns")
    
    correlations = []
    
    # Compute pair-wise correlations for meaningful numeric columns only
    for i, col1_name in enumerate(numeric_cols):
        for j, col2_name in enumerate(numeric_cols[i+1:], i+1):  # Avoid duplicate pairs
            series1 = df[col1_name]
            series2 = df[col2_name]
            
            # Clean the data before correlation
            clean_s1 = pd.to_numeric(series1, errors='coerce').dropna()
            clean_s2 = pd.to_numeric(series2, errors='coerce').dropna()
            
            # Align series to have the same indices
            aligned_df = pd.concat([clean_s1, clean_s2], axis=1).dropna()
            
            if len(aligned_df) < 3:  # Need at least 3 aligned points
                continue
                
            s1_aligned = aligned_df.iloc[:, 0]
            s2_aligned = aligned_df.iloc[:, 1]
            
            if len(s1_aligned) == 0 or len(s2_aligned) == 0 or len(s1_aligned) != len(s2_aligned):
                continue
            
            try:
                # Calculate Pearson correlation
                correlation, p_value = pearsonr(s1_aligned, s2_aligned)
                
                if pd.isna(correlation):
                    continue
                    
                abs_corr = abs(correlation)
                
                # Only include if correlation is meaningful
                if abs_corr >= min_correlation:
                    # Determine correlation strength
                    strength = "weak"
                    if abs_corr >= 0.7:
                        strength = "strong"
                    elif abs_corr >= 0.5:
                        strength = "moderate"
                    elif abs_corr >= 0.3:
                        strength = "moderate_weak"

                    # Determine correlation type
                    correlation_type = "positive" if correlation > 0 else "negative"

                    # Get basic statistics for the relationship
                    slope, intercept = np.polyfit(s1_aligned, s2_aligned, 1) if len(s1_aligned) > 1 else (0, 0)

                    correlations.append({
                        "variable1": col1_name,
                        "variable2": col2_name,
                        "correlation": float(correlation),
                        "abs_correlation": float(abs_corr),
                        "strength": strength,
                        "type": correlation_type,
                        "p_value": float(p_value),
                        "sample_size": int(len(aligned_df)),
                        "slope": float(slope),
                        "intercept": float(intercept)
                    })
                    
            except Exception as e:
                logger.warning(f"Error calculating correlation between {col1_name} and {col2_name}: {e}")
                continue
    
    logger.info(f"Identified {len(correlations)} meaningful correlations")
    return correlations


def _identify_cross_type_relationships(df: pd.DataFrame, columns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Identify meaningful relationships between different column types (numerical vs categorical).
    
    Args:
        df: Input DataFrame
        columns: List of column profiles including roles and names
        
    Returns:
        List of meaningful cross-type relationships as dictionaries
    """
    numeric_cols = []
    categorical_cols = []
    
    # Separate numeric and categorical columns, filtering out identifiers
    for col_info in columns:
        col_name = col_info["name"]
        series = df[col_name]
        
        # Skip if likely an identifier
        if _is_likely_identifier(series, col_name):
            continue
            
        if col_info.get("role") == "numeric":
            if _has_meaningful_variance(series):
                numeric_cols.append(col_name)
        elif col_info.get("role") in ["categorical", "text"]:
            if col_info.get("unique_count", 0) <= 50:  # Limit to low-cardinality categorical
                categorical_cols.append(col_name)
    
    relationships = []
    
    # Analyze numeric vs categorical relationships using ANOVA-like approach
    for num_col in numeric_cols:
        for cat_col in categorical_cols:
            series_num = pd.to_numeric(df[num_col], errors='coerce').dropna()
            series_cat = df[cat_col].dropna()
            
            # Align series
            aligned_df = pd.concat([series_num, series_cat], axis=1).dropna()
            
            if len(aligned_df) < 10:  # Need sufficient data
                continue
                
            aligned_num = aligned_df.iloc[:, 0]
            aligned_cat = aligned_df.iloc[:, 1]
            
            # Check if categorical has sufficient different values to be meaningful
            n_unique_cats = aligned_cat.nunique()
            if n_unique_cats < 2 or n_unique_cats > 20:  # Not meaningful if too few or too many categories
                continue
                
            try:
                # Group by category and calculate statistics
                grouped = aligned_num.groupby(aligned_cat)
                group_means = grouped.mean()
                group_sizes = grouped.count()
                
                # Calculate overall statistics
                overall_mean = aligned_num.mean()
                
                # Calculate between-group and within-group variance (ANOVA-like)
                between_sum_sq = sum(group_sizes * ((group_means - overall_mean) ** 2))
                within_sum_sq = sum([
                    ((aligned_num[aligned_cat == cat] - group_means[cat]) ** 2).sum() 
                    for cat in group_means.index
                ])
                
                # Calculate effect size (eta-squared - variance explained by group membership)
                total_sum_sq = between_sum_sq + within_sum_sq
                eta_squared = between_sum_sq / total_sum_sq if total_sum_sq > 0 else 0
                
                # Calculate significance using basic approximation (for larger samples)
                f_stat = (between_sum_sq / (n_unique_cats - 1)) / (within_sum_sq / (len(aligned_num) - n_unique_cats)) if within_sum_sq > 0 and (len(aligned_num) - n_unique_cats) > 0 else 0
                # This is a simplified approximation; for exact p-values, we'd need scipy.stats.f
                
                # Only include if there's meaningful variance explained
                if eta_squared >= 0.02:  # At least 2% variance explained
                    relationships.append({
                        "numeric_variable": num_col,
                        "categorical_variable": cat_col,
                        "effect_size": float(eta_squared),
                        "group_means": {str(cat): float(mean_val) for cat, mean_val in group_means.items()},
                        "group_sizes": {str(cat): int(size) for cat, size in group_sizes.items()},
                        "f_statistic": float(f_stat) if not pd.isna(f_stat) else 0.0,
                        "sample_size": int(len(aligned_num)),
                        "n_groups": int(n_unique_cats)
                    })
            except Exception as e:
                logger.warning(f"Error analyzing relationship between {num_col} and {cat_col}: {e}")
                continue
    
    logger.info(f"Identified {len(relationships)} meaningful cross-type relationships")
    return relationships


def _detect_spurious_correlations(df: pd.DataFrame, columns: List[Dict[str, Any]], 
                                correlations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detect spurious correlations in the already computed correlations.
    
    Args:
        df: Input DataFrame
        columns: List of column profiles
        correlations: List of computed correlation dictionaries
        
    Returns:
        List of potentially spurious correlations
    """
    spurious_correlations = []
    
    # Check for correlations involving likely identifiers (shouldn't happen if filtered properly, but double-check)
    for corr in correlations:
        var1 = corr["variable1"]
        var2 = corr["variable2"]
        
        series1 = df[var1]
        series2 = df[var2]
        
        is_id1 = _is_likely_identifier(series1, var1)
        is_id2 = _is_likely_identifier(series2, var2)
        
        if is_id1 or is_id2:
            spurious_correlations.append({
                "variables": [var1, var2],
                "correlation": corr["correlation"],
                "reason": "at_least_one_is_identifier",
                "confidence": 0.9
            })
    
    # Check for correlations between nearly constant variables
    for corr in correlations:
        var1 = corr["variable1"]
        var2 = corr["variable2"]
        
        series1 = df[var1]
        series2 = df[var2]
        
        # Check if either variable has very low variance
        has_low_var1 = not _has_meaningful_variance(series1)
        has_low_var2 = not _has_meaningful_variance(series2)
        
        if has_low_var1 or has_low_var2:
            reason = f"{'first' if has_low_var1 else 'second'} variable has low variance"
            spurious_correlations.append({
                "variables": [var1, var2],
                "correlation": corr["correlation"],
                "reason": reason,
                "confidence": 0.8
            })
    
    # Check for correlations with very small sample sizes
    for corr in correlations:
        if corr.get("sample_size", 0) < 10:
            spurious_correlations.append({
                "variables": [corr["variable1"], corr["variable2"]],
                "correlation": corr["correlation"],
                "reason": "insufficient sample size",
                "confidence": 0.7
            })
    
    logger.info(f"Detected {len(spurious_correlations)} potentially spurious correlations")
    return spurious_correlations


def analyze_correlations(df: pd.DataFrame, dataset_profile: Dict[str, Any], 
                       min_correlation: float = 0.1, 
                       min_variance: float = 0.001) -> Dict[str, Any]:
    """
    Comprehensive correlation analysis with proper filtering to avoid spurious correlations.
    
    Args:
        df: Input DataFrame
        dataset_profile: Dataset profile containing column information
        min_correlation: Minimum absolute correlation value to be considered meaningful
        min_variance: Minimum variance for columns to be considered meaningful
        
    Returns:
        Dictionary containing correlation analysis results including:
        - meaningful_correlations: List of meaningful correlations
        - cross_type_relationships: Relationships between different column types
        - spurious_correlations: Identified spurious correlations
        - summary_stats: Summary statistics about correlation analysis
    """
    if df.empty:
        logger.warning("Empty dataframe provided to correlation analysis")
        return {
            "meaningful_correlations": [],
            "cross_type_relationships": [],
            "spurious_correlations": [],
            "summary_stats": {"total_analyzed_pairs": 0, "meaningful_pairs": 0, "spurious_pairs": 0}
        }
    
    columns = dataset_profile.get("columns", [])
    if not columns:
        logger.warning("No columns found in dataset profile for correlation analysis")
        return {
            "meaningful_correlations": [],
            "cross_type_relationships": [],
            "spurious_correlations": [],
            "summary_stats": {"total_analyzed_pairs": 0, "meaningful_pairs": 0, "spurious_pairs": 0}
        }
    
    logger.info(f"Starting correlation analysis for {len(columns)} columns")
    
    # Perform meaningful correlation analysis
    meaningful_correlations = _identify_meaningful_correlations(
        df, columns, min_correlation, min_variance
    )
    
    # Identify cross-type relationships
    cross_type_relationships = _identify_cross_type_relationships(df, columns)
    
    # Detect spurious correlations
    spurious_correlations = _detect_spurious_correlations(df, columns, meaningful_correlations)
    
    # Create summary statistics
    n_numeric_cols = len([col for col in columns if col.get("role") == "numeric"])
    n_meaningful_numeric = len([col for col in columns 
                               if col.get("role") == "numeric" and 
                               not _is_likely_identifier(df[col["name"]], col["name"]) and
                               _has_meaningful_variance(df[col["name"]])])
    
    # Calculate total possible pairs for numeric columns (n*(n-1)/2)
    total_possible_pairs = (n_meaningful_numeric * (n_meaningful_numeric - 1)) // 2
    
    summary_stats = {
        "total_possible_pairs": total_possible_pairs,
        "total_analyzed_pairs": len(meaningful_correlations),
        "meaningful_pairs": len(meaningful_correlations),
        "spurious_pairs_detected": len(spurious_correlations),
        "analyzed_numeric_columns": n_meaningful_numeric,
        "original_numeric_columns": n_numeric_cols,
        "min_correlation_threshold": min_correlation,
        "min_variance_threshold": min_variance
    }
    
    logger.info(f"Correlation analysis completed: {len(meaningful_correlations)} meaningful relationships found out of {total_possible_pairs} possible pairs")
    
    return {
        "meaningful_correlations": meaningful_correlations,
        "cross_type_relationships": cross_type_relationships,
        "spurious_correlations": spurious_correlations,
        "summary_stats": summary_stats
    }


def generate_correlation_insights(correlation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate insights from the correlation analysis results.
    
    Args:
        correlation_results: Results from the analyze_correlations function
        
    Returns:
        List of insight dictionaries
    """
    insights = []
    
    # Insight from meaningful correlations
    meaningful_correlations = correlation_results.get("meaningful_correlations", [])
    if meaningful_correlations:
        # Find strongest correlations
        sorted_corrs = sorted(meaningful_correlations, key=lambda x: abs(x["correlation"]), reverse=True)
        strong_corrs = [c for c in sorted_corrs if abs(c["correlation"]) >= 0.7]
        
        if strong_corrs:
            insights.append({
                "type": "strong_correlation",
                "title": f"Strong Correlations Detected ({len(strong_corrs)} found)",
                "description": f"Identified {len(strong_corrs)} strongly correlated variable pairs (>0.7 correlation)",
                "details": [{"variables": [c["variable1"], c["variable2"]], "correlation": c["correlation"]} for c in strong_corrs[:5]],  # Limit to top 5
                "confidence": 0.9
            })
        
        # Find moderate correlations
        moderate_corrs = [c for c in meaningful_correlations if 0.5 <= abs(c["correlation"]) < 0.7]
        if moderate_corrs:
            insights.append({
                "type": "moderate_correlation",
                "title": f"Moderate Correlations Detected ({len(moderate_corrs)} found)",
                "description": f"Identified {len(moderate_corrs)} moderately correlated variable pairs (0.5-0.7 correlation)",
                "details": [{"variables": [c["variable1"], c["variable2"]], "correlation": c["correlation"]} for c in moderate_corrs[:5]],
                "confidence": 0.8
            })
    
    # Insight from cross-type relationships
    cross_relationships = correlation_results.get("cross_type_relationships", [])
    if cross_relationships:
        high_impact_relations = [r for r in cross_relationships if r["effect_size"] >= 0.15]  # High effect size
        
        if high_impact_relations:
            insights.append({
                "type": "cross_type_relationship",
                "title": f"Strong Cross-Type Relationships ({len(high_impact_relations)} found)",
                "description": f"Identified {len(high_impact_relations)} categorical variables that strongly influence numeric variables",
                "details": [{"numeric": r["numeric_variable"], "categorical": r["categorical_variable"], "effect_size": r["effect_size"]} for r in high_impact_relations[:5]],
                "confidence": 0.85
            })
    
    # Insight from spurious correlations detection
    spurious_correlations = correlation_results.get("spurious_correlations", [])
    if spurious_correlations:
        insights.append({
            "type": "data_quality_warning",
            "title": f"Spurious Correlations Flagged ({len(spurious_correlations)} found)",
            "description": f"Flagged {len(spurious_correlations)} potentially spurious correlations that may not be meaningful",
            "details": [{"variables": c["variables"], "correlation": c["correlation"], "reason": c["reason"]} for c in spurious_correlations[:5]],
            "confidence": 0.7
        })
    
    # Summary insight
    summary_stats = correlation_results.get("summary_stats", {})
    meaningful_pairs = summary_stats.get("meaningful_pairs", 0)
    analyzed_cols = summary_stats.get("analyzed_numeric_columns", 0)
    
    if analyzed_cols > 1:
        insights.append({
            "type": "correlation_summary",
            "title": f"Correlation Analysis Summary",
            "description": f"Analyzed {analyzed_cols} meaningful numeric columns and found {meaningful_pairs} significant relationships",
            "details": summary_stats,
            "confidence": 0.8
        })
    
    logger.info(f"Generated {len(insights)} correlation insights")
    return insights