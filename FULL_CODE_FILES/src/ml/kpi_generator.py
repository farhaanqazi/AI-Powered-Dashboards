"""
Advanced KPI generator with robust column analysis and semantic understanding.
Addresses core issues with misidentification of IDs, inappropriate correlations,
and meaningless KPI scoring that plagued the original implementation.
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats

logger = logging.getLogger(__name__)

def _semantic_column_analysis(df: pd.DataFrame, column_name: str) -> List[str]:
    """
    Analyze column name semantically to identify potential meanings
    with improved accuracy and confidence scoring.
    """
    col_lower = column_name.lower()

    # Define semantic categories and their patterns with confidence scores
    semantic_patterns = {
        'monetary': {
            'patterns': [
                r'price', r'cost', r'revenue', r'sales', r'amount', r'value',
                r'income', r'expense', r'salary', r'wage', r'fee', r'charge',
                r'payment', r'profit', r'budget', r'funding', r'investment',
                r'capital', r'dividend', r'tip', r'rate', r'premium', r'total'
            ],
            'confidence_boost': 2.0
        },
        'time': {
            'patterns': [
                r'time', r'date', r'hour', r'minute', r'second', r'day',
                r'week', r'month', r'year', r'season', r'period', r'interval',
                r'morning', r'evening', r'night', r'afternoon', r'duration'
            ],
            'confidence_boost': 1.5
        },
        'identifier': {
            'patterns': [
                r'id', r'identifier', r'code', r'number', r'num', r'index',
                r'key', r'uid', r'uuid', r'pk', r'ssn', r'pin', r'isbn',
                r'product', r'item', r'account', r'customer', r'user', r'client',
                r'order', r'transaction', r'invoice'
            ],
            'confidence_boost': -1.0  # Negative because identifiers should not be KPIs
        },
        'geographic': {
            'patterns': [
                r'country', r'city', r'state', r'province', r'county', r'address',
                r'location', r'coord', r'longitude', r'latitude', r'lat', r'lon',
                r'zip', r'postal', r'area', r'region', r'zone', r'neighborhood',
                r'continent', r'address'
            ],
            'confidence_boost': 1.2
        },
        'demographic': {
            'patterns': [
                r'age', r'gender', r'sex', r'race', r'ethnicity', r'nationality',
                r'education', r'occupation', r'marital', r'family',
                r'children', r'birth', r'death', r'life'
            ],
            'confidence_boost': 1.5
        },
        'rating': {
            'patterns': [
                r'rating', r'score', r'grade', r'rank', r'level', r'point',
                r'quality', r'satisfaction', r'review', r'feedback'
            ],
            'confidence_boost': 1.8
        },
        'quantity': {
            'patterns': [
                r'count', r'quantity', r'qty', r'volume', r'weight', r'height',
                r'width', r'length', r'depth', r'size', r'area', r'population',
                r'frequency', r'number', r'amount'
            ],
            'confidence_boost': 1.5
        },
        'percentage': {
            'patterns': [
                r'percent', r'percentage', r'pct', r'ratio', r'proportion',
                r'rate', r'fraction', r'part', r'discount', r'tax', r'interest',
                r'change'
            ],
            'confidence_boost': 1.5
        }
    }

    # Identify semantic categories with confidence scores
    identified_categories = []
    total_patterns_found = 0
    
    for category, data in semantic_patterns.items():
        category_matches = 0
        for pattern in data['patterns']:
            if re.search(pattern, col_lower):
                category_matches += 1
                total_patterns_found += 1
        if category_matches > 0:
            identified_categories.append(category)

    return identified_categories


def _is_likely_identifier(series: pd.Series, uniqueness_threshold: float = 0.95) -> bool:
    """
    Robustly detect if a series is likely an identifier based on multiple heuristics.
    """
    n_total = len(series)
    if n_total == 0:
        return False

    n_unique = series.nunique(dropna=True)
    unique_ratio = n_unique / n_total if n_total > 0 else 0.0

    # Check if column name suggests it's an ID
    name_lower = (series.name or "").lower()
    id_name_tokens = [
        "id", "identifier", "uuid", "guid", "key", "account", "user", "customer",
        "client", "booking", "transaction", "order", "invoice", "code", "number"
    ]
    looks_like_id_name = any(token in name_lower for token in id_name_tokens)

    # Check for sequential numeric patterns (common in internal IDs)
    if pd.api.types.is_numeric_dtype(series):
        numeric_values = pd.to_numeric(series, errors='coerce').dropna()
        if len(numeric_values) > 2:
            sorted_values = numeric_values.sort_values()
            diffs = sorted_values.diff().dropna()
            # If mostly step of 1, this is likely an internal sequential ID
            sequential_ratio = (diffs == 1).mean() if len(diffs) > 0 else 0.0
            if sequential_ratio > 0.8:
                return True

    # Check for UUID patterns in string values
    if series.dtype == 'object':
        sample_values = series.dropna().head(min(50, len(series))).astype(str)
        uuid_matches = 0
        for val in sample_values:
            if re.match(r'^[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}$', val, re.IGNORECASE):
                uuid_matches += 1
        uuid_confidence = uuid_matches / len(sample_values) if len(sample_values) > 0 else 0
        
        if uuid_confidence > 0.5:
            return True

    # Use combination of uniqueness and name/context clues
    if unique_ratio > uniqueness_threshold:
        if looks_like_id_name:
            return True
        elif unique_ratio > 0.99:  # Very high uniqueness might indicate ID anyway
            return True

    return False


def _calculate_outliers(series: pd.Series, method: str = 'iqr') -> int:
    """
    Calculate outliers in a numeric series using IQR method (or other methods).
    """
    series = pd.to_numeric(series, errors='coerce').dropna()
    if len(series) < 4:  # Need at least 4 points for meaningful outlier detection
        return 0

    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR if pd.notna(IQR) and IQR != 0 else Q1 - 1.5
        upper_bound = Q3 + 1.5 * IQR if pd.notna(IQR) and IQR != 0 else Q3 + 1.5
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return len(outliers)

    return 0


def _calculate_distribution_metrics(series: pd.Series) -> Tuple[float, float]:
    """
    Calculate distribution metrics: skewness and kurtosis using pandas.
    Returns (skewness, kurtosis) or (0, 0) if not enough data.
    """
    series = pd.to_numeric(series, errors='coerce').dropna()
    if len(series) < 4:  # Need at least 4 points for meaningful distribution metrics
        return 0.0, 0.0

    try:
        skewness = series.skew()
        kurtosis = series.kurtosis()
        # Ensure valid values
        skewness = float(skewness) if pd.notna(skewness) and np.isfinite(skewness) else 0.0
        kurtosis = float(kurtosis) if pd.notna(kurtosis) and np.isfinite(kurtosis) else 0.0
        return skewness, kurtosis
    except Exception as e:
        logger.warning(f"Error calculating distribution metrics: {e}")
        return 0.0, 0.0


def _calculate_correlations(df: pd.DataFrame) -> List[Tuple[float, float, str, str]]:
    """
    Calculate correlation matrix for meaningful numeric columns with filtering to avoid spurious correlations.
    Returns list of correlations as (absolute_corr, corr_value, col1, col2).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Filter to only meaningful numeric columns (not IDs/codes)
    meaningful_numeric_cols = []
    for col in numeric_cols:
        series = df[col]
        
        # Skip if it's likely an identifier
        if _is_likely_identifier(series):
            continue
            
        # Skip if it has very low variance (essentially constant)
        std_val = pd.to_numeric(series, errors='coerce').std()
        if pd.isna(std_val) or std_val < 0.001:  # Essentially constant
            continue
            
        # Skip if it has too many unique values that look like codes/IDs
        unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
        if unique_ratio > 0.95 and len(series) > 10:
            # Check if these high-cardinality values are mostly integer-like (indicating IDs)
            numeric_values = pd.to_numeric(series, errors='coerce')
            integer_ratio = (numeric_values == numeric_values.round()).sum() / len(numeric_values) if len(numeric_values) > 0 else 0
            if integer_ratio > 0.9:
                continue  # Likely ID field, skip
                
        meaningful_numeric_cols.append(col)

    if len(meaningful_numeric_cols) < 2:
        logger.info(f"Not enough meaningful numeric columns to calculate correlations (found {len(meaningful_numeric_cols)})")
        return []

    corr_matrix = df[meaningful_numeric_cols].corr()

    # Get pairs with highest absolute correlation (excluding self-correlations)
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            
            # Only include if correlation is valid and meaningful (not too close to 0)
            if pd.notna(corr_value) and abs(corr_value) > 0.1:  # Only correlations above threshold
                correlations.append((abs(corr_value), corr_value, col1, col2))

    # Sort by absolute correlation value (highest first)
    correlations.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"Found {len(correlations)} meaningful correlation pairs")
    return correlations


def _calculate_significance_score(series: pd.Series, semantic_categories: List[str]) -> float:
    """
    Calculate a significance score for a column based on statistical properties and semantic meaning.
    Higher scores indicate more important KPIs.
    """
    if series.empty or series.isna().all():
        return 0.0

    # Check if this is likely an identifier (should have low significance)
    if _is_likely_identifier(series):
        return 0.05  # Very low score for identifiers

    score = 0.0

    # Statistical significance factors
    n_valid = len(series.dropna())
    if n_valid == 0:
        return 0.0

    # Variability score: meaningful metrics tend to have variability
    if pd.api.types.is_numeric_dtype(series):
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        if len(numeric_series) > 1:
            std_dev = numeric_series.std()
            mean_val = numeric_series.mean()
            
            if pd.notna(std_dev) and pd.notna(mean_val) and mean_val != 0:
                cv = abs(std_dev / mean_val)  # Coefficient of variation
                score += min(0.5, cv)  # Cap at 0.5 to prevent extreme scores
            elif pd.notna(std_dev):
                score += min(0.5, std_dev / 10)  # If mean is 0, use raw std deviation capped

    # Uniqueness score: avoid both too unique (IDs) and too uniform (constants)
    n_unique = series.nunique(dropna=True)
    unique_ratio = n_unique / n_valid if n_valid > 0 else 0.0

    # Score peaks at medium uniqueness, penalizes extreme uniqueness (like IDs) or low uniqueness (like constants)
    if 0.05 <= unique_ratio <= 0.90:  # Good range for meaningful categorical variables
        uniqueness_bonus = 0.2
        # Further boost if it's in the sweet spot (not too many, not too few categories)
        if 2 <= n_unique <= 20:  # Ideal range for categorical KPIs
            uniqueness_bonus += 0.1
        score += uniqueness_bonus
    elif unique_ratio > 0.95:  # Probably an ID, reduce score
        score -= 0.3

    # Semantic significance boosts
    for semantic_cat in semantic_categories:
        if semantic_cat in ['monetary', 'rating', 'quantity']:
            score += 0.3  # High importance semantic categories
        elif semantic_cat in ['demographic', 'percentage']:
            score += 0.2  # Medium importance
        elif semantic_cat in ['time']:
            score += 0.15  # Time fields are often important

    # Distribution shape significance (for numeric fields)
    if pd.api.types.is_numeric_dtype(series):
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        if len(numeric_series) >= 4:  # Need enough points for distribution metrics
            skewness, kurtosis = _calculate_distribution_metrics(numeric_series)
            if abs(skewness) > 1.0:  # Highly skewed - might be important for analysis
                score += 0.1
            if abs(kurtosis) > 1.0:  # Heavy or light-tailed - might be important
                score += 0.1

    # Outlier significance
    if pd.api.types.is_numeric_dtype(series):
        outlier_count = _calculate_outliers(series)
        if outlier_count > 0:
            outlier_ratio = outlier_count / n_valid if n_valid > 0 else 0
            # More outliers might indicate interesting phenomena
            score += min(0.1, outlier_ratio * 0.2)  # Cap the outlier boost

    # Ensure score is between 0 and 1
    score = max(0.0, min(1.0, score))
    return score


def generate_kpis(df: pd.DataFrame, dataset_profile: Dict[str, Any],
                 min_variability_threshold: float = 0.01,
                 min_unique_ratio: float = 0.01,
                 max_unique_ratio: float = 0.9,
                 top_k: int = 10,
                 eda_summary: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Enhanced KPIs with robust statistical metrics and semantic understanding.
    Addresses issues with misidentified IDs, inappropriate correlations, and 
    meaningless KPI scoring.

    Args:
        df: Input DataFrame
        dataset_profile: Dataset profile from analyser
        min_variability_threshold: Minimum standard deviation to consider a column variable
        min_unique_ratio: Minimum unique ratio for categorical columns
        max_unique_ratio: Maximum unique ratio (to avoid IDs)
        top_k: Number of top KPIs to return

    Returns:
        List of KPI dictionaries with significance scores and semantic meaning
    """
    if df.empty:
        logger.warning("Empty dataframe provided to KPI generator")
        return []

    logger.info(f"Generating enhanced KPIs with parameters: min_variability_threshold={min_variability_threshold}, "
                f"min_unique_ratio={min_unique_ratio}, max_unique_ratio={max_unique_ratio}, top_k={top_k}")

    kpis = []
    columns = dataset_profile["columns"]
    n_rows = dataset_profile.get("n_rows", len(df))

    if n_rows == 0:
        logger.warning("No data rows provided to KPI generator")
        return []

    # Prepare a comprehensive analysis of each column
    column_analyses = []
    
    for col in columns:
        col_name = col["name"]
        series = df[col_name]
        
        # Skip if series is all NaN
        if series.isna().all():
            continue
            
        # Determine if this is likely an identifier
        is_identifier = _is_likely_identifier(series)
        
        # Perform semantic analysis
        semantic_categories = _semantic_column_analysis(df, col_name)
        
        # Calculate significance score
        significance_score = _calculate_significance_score(series, semantic_categories)
        
        # Additional metrics for KPI characterization
        n_unique = series.nunique(dropna=True)
        n_valid = len(series.dropna())
        unique_ratio = n_unique / n_valid if n_valid > 0 else 0.0
        
        # Calculate basic statistics for numeric fields
        numeric_stats = {}
        if pd.api.types.is_numeric_dtype(series):
            numeric_series = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_series) > 0:
                numeric_stats = {
                    'mean': float(numeric_series.mean()) if len(numeric_series) > 0 else None,
                    'std': float(numeric_series.std()) if len(numeric_series) > 1 else 0.0,
                    'min': float(numeric_series.min()) if len(numeric_series) > 0 else None,
                    'max': float(numeric_series.max()) if len(numeric_series) > 0 else None,
                    'median': float(numeric_series.median()) if len(numeric_series) > 0 else None,
                }
        
        # Count outliers if numeric
        outlier_count = _calculate_outliers(series) if pd.api.types.is_numeric_dtype(series) else 0
        
        column_analyses.append({
            'name': col_name,
            'role': col.get('role', 'unknown'),
            'semantic_categories': semantic_categories,
            'is_identifier': is_identifier,
            'significance_score': significance_score,
            'n_unique': n_unique,
            'n_valid': n_valid,
            'unique_ratio': unique_ratio,
            'numeric_stats': numeric_stats,
            'outlier_count': outlier_count
        })

    # Sort by significance score to identify the most important columns
    column_analyses.sort(key=lambda x: x['significance_score'], reverse=True)
    
    # Generate KPIs based on the analyses
    for analysis in column_analyses[:top_k]:  # Take top K by significance
        col_name = analysis['name']
        col_role = analysis['role']
        significance_score = analysis['significance_score']
        
        # Skip identifiers as they are not meaningful KPIs
        if analysis['is_identifier']:
            continue
            
        # Prepare KPI information
        kpi_info = {
            "label": col_name,
            "type": col_role,
            "significance_score": significance_score,
            "semantic_categories": analysis['semantic_categories'],
            "provenance": "enhanced_analysis"
        }
        
        # Add specific metrics based on the column's role and semantic categories
        series = df[col_name]
        
        if col_role == 'numeric':
            numeric_stats = analysis['numeric_stats']
            if numeric_stats and numeric_stats.get('mean') is not None:
                avg_val = numeric_stats['mean']
                kpi_info["value"] = f"{avg_val:.2f} (±{numeric_stats['std']:.2f})"
            else:
                # Compute from series directly if stats weren't cached
                numeric_series = pd.to_numeric(series, errors='coerce').dropna()
                if len(numeric_series) > 0:
                    mean_val = float(numeric_series.mean())
                    std_val = float(numeric_series.std()) if len(numeric_series) > 1 else 0.0
                    kpi_info["value"] = f"{mean_val:.2f} (±{std_val:.2f})"
                else:
                    kpi_info["value"] = "No valid numeric values"

            # Add outlier information
            if analysis['outlier_count'] > 0:
                kpi_info["value"] += f" [{analysis['outlier_count']} outliers]"

        elif col_role in ['categorical', 'text']:
            # For categorical/text, show top value and distribution
            top_value_counts = series.value_counts(dropna=True).head(3)
            if len(top_value_counts) > 0:
                top_val = top_value_counts.index[0]
                top_count = top_value_counts.iloc[0]
                total_valid = analysis['n_valid']
                pct_top = (top_count / total_valid) * 100 if total_valid > 0 else 0
                kpi_info["value"] = f"Top: '{top_val}' ({top_count}, {pct_top:.1f}%)"
            else:
                kpi_info["value"] = "No valid values"
                
        elif col_role == 'datetime':
            # For datetime, show time range
            try:
                datetime_series = pd.to_datetime(series, errors='coerce').dropna()
                if len(datetime_series) > 0:
                    min_date = datetime_series.min().strftime('%Y-%m-%d')
                    max_date = datetime_series.max().strftime('%Y-%m-%d')
                    kpi_info["value"] = f"Range: {min_date} to {max_date}"
                else:
                    kpi_info["value"] = "No valid dates"
            except Exception as e:
                logger.warning(f"Error processing datetime column {col_name}: {e}")
                kpi_info["value"] = "Invalid datetime format"
                
        else:
            # For other roles, show basic statistics
            kpi_info["value"] = f"Unique: {analysis['n_unique']}, Missing: {series.isna().sum()}"
        
        # Add semantic context to the value description if relevant
        if analysis['semantic_categories']:
            semantic_info = ", ".join(analysis['semantic_categories'])
            kpi_info["value"] += f" [{semantic_info}]"
            
        kpis.append(kpi_info)

    # Additionally, add some correlation-based KPIs
    try:
        correlations = _calculate_correlations(df)
        for abs_corr, corr_val, col1, col2 in correlations[:3]:  # Top 3 correlations
            kpis.append({
                "label": f"Correlation: {col1} ↔ {col2}",
                "value": f"{corr_val:.3f}",
                "type": "correlation",
                "correlation_value": corr_val,
                "columns": [col1, col2],
                "significance_score": abs(corr_val),  # Correlation strength as significance
                "provenance": "correlation_insight"
            })
    except Exception as e:
        logger.warning(f"Error calculating correlations: {e}")

    logger.info(f"Generated {len(kpis)} enhanced KPIs")
    return kpis


def generate_basic_kpis(df: pd.DataFrame, dataset_profile: Dict[str, Any],
                       min_variability_threshold: float = 0.01,
                       min_unique_ratio: float = 0.01,
                       max_unique_ratio: float = 0.9,
                       top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Wrapper function to maintain backward compatibility while using the enhanced generator.
    """
    return generate_kpis(
        df=df,
        dataset_profile=dataset_profile,
        min_variability_threshold=min_variability_threshold,
        min_unique_ratio=min_unique_ratio,
        max_unique_ratio=max_unique_ratio,
        top_k=top_k
    )