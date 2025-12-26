import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from src.utils.identifier_detector import is_likely_identifier
from src import config

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


# Note: The _is_likely_identifier function is kept for backward compatibility
# but will not be used in the main KPI generation process to avoid re-analyzing data
def _is_likely_identifier(series: pd.Series, uniqueness_threshold: float = 0.95) -> bool:
    """
    Robustly detect if a series is likely an identifier based on multiple heuristics.
    This function is kept for compatibility but not used in the main KPI generation.
    """
    n_total = len(series)
    if n_total == 0:
        return False

    n_unique = series.nunique(dropna=True)
    unique_ratio = n_unique / n_total if n_total > 0 else 0.0

    # Use the centralized identifier detector for basic checks
    if is_likely_identifier(series, name=series.name or ""):
        return True

    # Additional check for sequential numeric patterns (specific to KPI context)
    if pd.api.types.is_numeric_dtype(series):
        numeric_values = pd.to_numeric(series, errors='coerce').dropna()
        if len(numeric_values) > 2:
            sorted_values = numeric_values.sort_values()
            diffs = sorted_values.diff().dropna()
            # If mostly step of 1, this is likely an internal sequential ID
            sequential_ratio = (diffs == 1).mean() if len(diffs) > 0 else 0.0
            if sequential_ratio > 0.8:
                return True

    # Use combination of uniqueness and name/context clues
    if unique_ratio > uniqueness_threshold:
        # Check if column name suggests it's an ID
        name_lower = (series.name or "").lower()
        id_name_tokens = [
            "id", "identifier", "uuid", "guid", "key", "account", "user", "customer",
            "client", "booking", "transaction", "order", "invoice", "code", "number"
        ]
        looks_like_id_name = any(token in name_lower for token in id_name_tokens)

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
        # Handle case where IQR is 0 (all values in middle 50% are the same)
        if pd.isna(IQR) or IQR == 0:
             lower_bound = Q1 - 1.5
             upper_bound = Q3 + 1.5
        else:
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return len(outliers)

    return 0


def _calculate_distribution_metrics(series: pd.Series) -> Tuple[float, float]:
    """
    Calculate distribution metrics: skewness and kurtosis using pandas.
    Returns (skewness, kurtosis) or (0, 0) if not enough data or calculation fails.
    """
    series = pd.to_numeric(series, errors='coerce').dropna()
    if len(series) < 4:  # Need at least 4 points for meaningful distribution metrics
        return 0.0, 0.0

    try:
        skewness = series.skew()
        kurtosis = series.kurtosis()
        # Ensure valid finite values
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
            if not numeric_values.isna().all():
                 integer_ratio = (numeric_values == numeric_values.round()).sum() / len(numeric_values) if len(numeric_values) > 0 else 0
                 if integer_ratio > 0.9:
                    continue  # Likely ID field, skip
                
        meaningful_numeric_cols.append(col)

    if len(meaningful_numeric_cols) < 2:
        logger.info(f"Not enough meaningful numeric columns to calculate correlations (found {len(meaningful_numeric_cols)})")
        return []

    # Select only the meaningful columns for correlation calculation
    subset_df = df[meaningful_numeric_cols]
    # Ensure all selected columns are numeric for corr()
    subset_df = subset_df.select_dtypes(include=[np.number])
    meaningful_numeric_cols = list(subset_df.columns) # Update list after selection

    if len(meaningful_numeric_cols) < 2:
        logger.info(f"After cleaning, not enough meaningful numeric columns to calculate correlations (found {len(meaningful_numeric_cols)})")
        return []

    corr_matrix = subset_df.corr()

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
    if not isinstance(series, pd.Series):
        logger.warning(f"Input to _calculate_significance_score is not a pandas Series. Type: {type(series)}")
        return 0.0 # Return default score

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
            std_dev = 0.0
            mean_val = 0.0
            try:
                std_dev = float(numeric_series.std())
            except (AttributeError, TypeError, ValueError) as e:
                logger.warning(f"Error calculating std_dev for series '{series.name}': {e}. Defaulting to 0.0.")
            try:
                mean_val = float(numeric_series.mean())
            except (AttributeError, TypeError, ValueError) as e:
                logger.warning(f"Error calculating mean_val for series '{series.name}': {e}. Defaulting to 0.0.")

            if pd.notna(std_dev) and pd.notna(mean_val) and mean_val != 0 and np.isfinite(std_dev) and np.isfinite(mean_val):
                cv = abs(std_dev / mean_val)  # Coefficient of variation
                score += min(0.5, cv)  # Cap at 0.5 to prevent extreme scores
            elif pd.notna(std_dev) and np.isfinite(std_dev): # If mean_val is 0 or NaN, but std_dev is valid
                score += min(0.5, std_dev / 10) # Use raw std dev capped

    # Uniqueness score: avoid both too unique (IDs) and too uniform (constants)
    n_unique = 0
    try:
        n_unique = series.nunique(dropna=True)
    except AttributeError:
        logger.warning(f"AttributeError calculating nunique for series '{series.name}'. Defaulting to 0.")
    except Exception as e:
        logger.warning(f"Error calculating nunique for series '{series.name}': {e}. Defaulting to 0.")

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
    numeric_series_for_dist = pd.to_numeric(series, errors='coerce').dropna()
    if len(numeric_series_for_dist) >= 4:  # Need enough points for distribution metrics
        skewness, kurtosis = _calculate_distribution_metrics(numeric_series_for_dist)
        # Only add score if the metrics are valid numbers
        if abs(skewness) > 1.0 and np.isfinite(skewness):  # Highly skewed - might be important for analysis
            score += 0.1
        if abs(kurtosis) > 1.0 and np.isfinite(kurtosis): # Heavy or light-tailed - might be important
            score += 0.1

    # Outlier significance
    numeric_series_for_outliers = pd.to_numeric(series, errors='coerce').dropna()
    if len(numeric_series_for_outliers) > 0: # Only run if there's data to analyze
        outlier_count = _calculate_outliers(numeric_series_for_outliers)
        if outlier_count > 0:
            outlier_ratio = outlier_count / n_valid if n_valid > 0 else 0
            # More outliers might indicate interesting phenomena
            score += min(0.1, outlier_ratio * 0.2)  # Cap the outlier boost

    # Ensure score is between 0 and 1
    score = max(0.0, min(1.0, score))
    return score


def _calculate_significance_score_from_profile(col_profile: Dict[str, Any], semantic_categories: List[str]) -> float:
    """
    Calculate a significance score for a column based on dataset profile data and semantic meaning.
    This avoids re-analyzing the raw series data.
    """
    score = 0.0

    # Get column information from profile
    col_role = col_profile.get("role", "unknown")
    n_unique = col_profile.get("unique_count", 0)
    n_total = col_profile.get("stats", {}).get("count", 0)
    missing_count = col_profile.get("missing_count", 0)

    # Check if this is an identifier (should have low significance)
    if col_role == "identifier":
        return 0.05  # Very low score for identifiers

    # Check if this is near-constant (only one unique non-null value)
    n_valid = n_total - missing_count
    if n_valid > 0:
        unique_ratio = n_unique / n_valid if n_valid > 0 else 0.0

        # Uniqueness score: avoid both too unique (IDs) and too uniform (constants)
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

    # Add statistical significance for numeric columns based on profile stats
    if col_role == "numeric" and col_profile.get("stats"):
        stats = col_profile["stats"]
        std_val = stats.get("std", 0.0)
        mean_val = stats.get("mean")

        # Variability score: meaningful metrics tend to have variability
        if mean_val is not None and mean_val != 0 and np.isfinite(std_val) and np.isfinite(mean_val):
            cv = abs(std_val / mean_val)  # Coefficient of variation
            score += min(0.5, cv)  # Cap at 0.5 to prevent extreme scores
        elif np.isfinite(std_val):
            score += min(0.5, std_val / 10)  # If mean is 0, use raw std deviation capped

    # Ensure score is between 0 and 1
    score = max(0.0, min(1.0, score))
    return score


def generate_kpis(df: pd.DataFrame, dataset_profile: Dict[str, Any],
                 min_variability_threshold: float = config.MIN_VARIABILITY_THRESHOLD,
                 min_unique_ratio: float = config.MIN_UNIQUE_RATIO,
                 max_unique_ratio: float = config.MAX_UNIQUE_RATIO,
                 top_k: int = config.KPI_TOP_K,
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
        eda_summary: Optional EDA summary to incorporate insights

    Returns:
        List of KPI dictionaries with significance scores and semantic meaning
    """
    if df.empty:
        logger.warning("Empty dataframe provided to KPI generator")
        return []

    logger.info(f"Generating enhanced KPIs with parameters: min_variability_threshold={min_variability_threshold}, "
                f"min_unique_ratio={min_unique_ratio}, max_unique_ratio={max_unique_ratio}, top_k={top_k}")

    kpis = []
    columns = dataset_profile.get("columns", [])
    n_rows = dataset_profile.get("n_rows", len(df))

    if n_rows == 0:
        logger.warning("No data rows provided to KPI generator")
        return []

    # Prepare a comprehensive analysis of each column using dataset_profile
    column_analyses = []

    for col in columns:
        if not isinstance(col, dict): # Validate col is a dict
            logger.warning(f"Invalid column profile entry (not a dictionary): {col}. Skipping.")
            continue

        col_name = col.get("name")
        if col_name is None: # Validate col_name exists
            logger.warning(f"Column profile missing 'name' key: {col}. Skipping.")
            continue

        if col_name not in df.columns: # Validate col_name exists in actual df
            logger.warning(f"Column '{col_name}' from profile not found in DataFrame. Skipping.")
            continue

        # Get semantic tags and role from dataset profile
        semantic_tags = col.get("semantic_tags", [])
        col_role = col.get("role", "unknown")

        # Check if this is an identifier based on dataset profile
        is_identifier = (col_role == "identifier")

        # Check if this is near-constant based on dataset profile
        n_unique = col.get("unique_count", 0)
        n_total = col.get("stats", {}).get("count", n_rows if n_rows > 0 else len(df))
        unique_ratio = n_unique / n_total if n_total > 0 else 0.0

        # Check if this is near-constant (only one unique non-null value)
        is_near_constant = (n_unique <= 1)

        # Check if column has only NaN values
        missing_count = col.get("missing_count", 0)
        is_all_nan = (missing_count == n_total)

        # Skip if all values are NaN
        if is_all_nan:
            logger.debug(f"Skipping column {col_name} as it contains only NaN values.")
            continue

        # Skip if it's an identifier or near-constant
        if is_identifier or is_near_constant:
            logger.debug(f"Skipping {'identifier' if is_identifier else 'near-constant'} column {col_name} from KPI generation.")
            continue

        # Determine semantic categories from profile semantic tags
        semantic_categories = semantic_tags.copy()  # Use semantic tags from profile

        # Calculate significance score based on profile data
        significance_score = _calculate_significance_score_from_profile(col, semantic_categories)

        # Get statistics from profile
        numeric_stats = col.get("stats", {})

        column_analyses.append({
            'name': col_name,
            'role': col_role,
            'semantic_categories': semantic_categories,
            'is_identifier': is_identifier,
            'is_near_constant': is_near_constant,
            'significance_score': significance_score,
            'n_unique': n_unique,
            'n_valid': n_total - missing_count,
            'unique_ratio': unique_ratio,
            'numeric_stats': numeric_stats,
            'missing_count': missing_count
        })

    # Sort by significance score to identify the most important columns
    column_analyses.sort(key=lambda x: x['significance_score'], reverse=True)
    
    # Generate KPIs based on the analyses
    for analysis in column_analyses[:top_k]:  # Take top K by significance
        col_name = analysis['name']
        col_role = analysis['role']
        significance_score = analysis['significance_score']

        # Skip identifiers or near-constant fields as they are not meaningful KPIs
        if analysis['is_identifier'] or analysis['is_near_constant']:
            logger.debug(f"Skipping {'identifier' if analysis['is_identifier'] else 'near-constant'} column {col_name} from KPI generation.")
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
                mean_val = numeric_stats.get('mean')
                std_val = numeric_stats.get('std', 0.0)
                if mean_val is not None and np.isfinite(mean_val):
                    kpi_info["value"] = f"{mean_val:.2f} (±{std_val:.2f})"
                else:
                    kpi_info["value"] = "Invalid numeric value"
            else:
                # Fallback: compute from series directly if stats weren't available in profile
                numeric_series = pd.to_numeric(series, errors='coerce').dropna()
                if len(numeric_series) > 0:
                    mean_val = float(numeric_series.mean())
                    std_val = float(numeric_series.std()) if len(numeric_series) > 1 else 0.0
                    if np.isfinite(mean_val):
                         kpi_info["value"] = f"{mean_val:.2f} (±{std_val:.2f})"
                    else:
                         kpi_info["value"] = "Invalid numeric value"
                else:
                    kpi_info["value"] = "No valid numeric values"

            # Add outlier information if available in profile
            outlier_count = numeric_stats.get('outlier_count', 0) if numeric_stats else 0
            if outlier_count > 0 and 'Invalid' not in kpi_info["value"]:
                kpi_info["value"] += f" [{outlier_count} outliers]"

        elif col_role in ['categorical', 'text']:
            # For categorical/text, show top value and distribution
            top_categories = col.get("top_categories", [])
            if top_categories:
                top_val = top_categories[0]["value"]
                top_count = top_categories[0]["count"]
                total_valid = analysis['n_valid']
                pct_top = (top_count / total_valid) * 100 if total_valid > 0 else 0
                kpi_info["value"] = f"Top: '{top_val}' ({top_count}, {pct_top:.1f}%)"
            else:
                kpi_info["value"] = "No valid values"

        elif col_role == 'datetime':
            # For datetime, show time range from profile
            stats = analysis.get('numeric_stats', {})
            if stats and stats.get('min') and stats.get('max'):
                kpi_info["value"] = f"Range: {stats['min']} to {stats['max']}"
            else:
                # Fallback to series if not in profile
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
            kpi_info["value"] = f"Unique: {analysis['n_unique']}, Missing: {analysis['missing_count']}"

        # Add semantic context to the value description if relevant
        if analysis['semantic_categories']:
            semantic_info = ", ".join(analysis['semantic_categories'])
            kpi_info["value"] += f" [{semantic_info}]"

        kpis.append(kpi_info)

    # Validation step: check for invalid KPIs and filter them out
    validated_kpis = []
    for kpi in kpis:
        # Check for invalid values
        if kpi.get('value') and kpi['value'] not in ['Invalid numeric value', 'No valid values', 'No valid numeric values', 'Invalid datetime format']:
            # Check significance score is valid
            sig_score = kpi.get('significance_score')
            if sig_score is not None and not (isinstance(sig_score, float) and np.isnan(sig_score)):
                validated_kpis.append(kpi)
            else:
                logger.warning(f"Filtered out KPI '{kpi.get('label')}' due to invalid significance score: {sig_score}")
        else:
            logger.warning(f"Filtered out KPI '{kpi.get('label')}' due to invalid value: {kpi.get('value')}")

    kpis = validated_kpis

    # Additionally, add some correlation-based KPIs if EDA summary is available
    if eda_summary:
        try:
            # Try to get correlations from EDA summary first, then fall back to calculating
            correlations = eda_summary.get('patterns_and_relationships', {}).get('correlations', [])
            if not correlations:
                 correlations = _calculate_correlations(df)

            for abs_corr, corr_val, col1, col2 in correlations[:3]:  # Top 3 correlations
                correlation_kpi = {
                    "label": f"Correlation: {col1} ↔ {col2}",
                    "value": f"{corr_val:.3f}",
                    "type": "correlation",
                    "correlation_value": corr_val,
                    "columns": [col1, col2],
                    "significance_score": abs(corr_val),  # Correlation strength as significance
                    "provenance": "correlation_insight"
                }

                # Add correlation KPI if it's valid
                if correlation_kpi['value'] and not (isinstance(correlation_kpi.get('significance_score'), float) and np.isnan(correlation_kpi.get('significance_score'))):
                    kpis.append(correlation_kpi)
        except Exception as e:
            logger.warning(f"Error calculating or adding correlation-based KPIs: {e}")

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