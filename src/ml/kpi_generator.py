# src/ml/kpi_generator.py
import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

def _semantic_column_analysis(df, column_name):
    """
    Analyze column name semantically to identify potential meanings
    """
    col_lower = column_name.lower()

    # Define semantic categories and their patterns
    semantic_patterns = {
        'monetary': [
            r'price', r'cost', r'revenue', r'sales', r'amount', r'value',
            r'income', r'expense', r'salary', r'wage', r'fee', r'charge',
            r'payment', r'profit', r'budget', r'funding', r'investment',
            r'capital', r'dividend', r'tip', r'rate', r'premium'
        ],
        'time': [
            r'time', r'date', r'hour', r'minute', r'second', r'day',
            r'week', r'month', r'year', r'season', r'period', r'interval',
            r'morning', r'evening', r'night', r'afternoon', r'duration'
        ],
        'identifier': [
            r'id', r'identifier', r'code', r'number', r'num', r'index',
            r'key', r'uid', r'uuid', r'pk', r'ssn', r'pin', r'isbn',
            r'product', r'item', r'account', r'customer', r'user', r'client'
        ],
        'geographic': [
            r'country', r'city', r'state', r'province', r'county', r'address',
            r'location', r'coord', r'longitude', r'latitude', r'lat', r'lon',
            r'zip', r'postal', r'area', r'region', r'zone', r'neighborhood'
        ],
        'demographic': [
            r'age', r'gender', r'sex', r'race', r'ethnicity', r'nationality',
            r'education', r'occupation', r'income', r'marital', r'family',
            r'children', r'population', r'birth', r'death', r'life'
        ],
        'rating': [
            r'rating', r'score', r'grade', r'rank', r'level', r'point',
            r'quality', r'satisfaction', r'rating', r'review', r'feedback'
        ],
        'quantity': [
            r'count', r'quantity', r'qty', r'volume', r'weight', r'height',
            r'width', r'length', r'depth', r'size', r'area', r'population',
            r'frequency', r'number', r'amount'
        ],
        'percentage': [
            r'percent', r'percentage', r'pct', r'ratio', r'proportion',
            r'rate', r'fraction', r'part', r'discount', r'tax', r'interest'
        ]
    }

    # Identify semantic category
    identified_categories = []
    for category, patterns in semantic_patterns.items():
        for pattern in patterns:
            if re.search(pattern, col_lower):
                identified_categories.append(category)
                break  # Break to avoid multiple matches for same category

    return identified_categories


def _calculate_outliers(series, method='iqr'):
    """
    Calculate outliers in a numeric series using IQR method
    """
    series = series.dropna()
    if len(series) == 0:
        return 0

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return len(outliers)


def _calculate_distribution_metrics(series):
    """
    Calculate distribution metrics: skewness and kurtosis using pandas
    """
    series = series.dropna()
    if len(series) < 3:
        return 0, 0  # Not enough data points

    series = pd.to_numeric(series, errors='coerce').dropna()
    if len(series) < 3:
        return 0, 0

    try:
        skewness = series.skew()
        kurtosis = series.kurtosis()
        return skewness, kurtosis
    except Exception as e:
        logger.warning(f"Error calculating distribution metrics: {e}")
        return 0, 0


def _calculate_correlations(df):
    """
    Calculate correlation matrix for numeric columns
    Returns list of top correlations between different columns
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        logger.info("Not enough numeric columns to calculate correlations")
        return []

    corr_matrix = df[numeric_cols].corr()

    # Get pairs with highest absolute correlation (excluding self-correlations)
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            correlations.append((abs(corr_value), corr_value, col1, col2))

    # Sort by absolute correlation value (highest first)
    correlations.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"Found {len(correlations)} correlation pairs")
    return correlations


def generate_basic_kpis(df, dataset_profile, min_variability_threshold=0.01, min_unique_ratio=0.01, max_unique_ratio=0.9, top_k=3):
    """
    Enhanced KPIs with advanced statistical metrics and semantic understanding.
    KPIs = columns that look important based on their data behaviour and semantic meaning,
    not on hard-coded name patterns.

    Args:
        df: Input DataFrame
        dataset_profile: Dataset profile from analyser
        min_variability_threshold: Minimum standard deviation to consider a column variable
        min_unique_ratio: Minimum unique ratio for categorical columns
        max_unique_ratio: Maximum unique ratio for categorical columns (to avoid IDs)
        top_k: Number of top KPIs to return for each category

    We highlight:
    - Top numeric columns by variability, correlation, and distribution characteristics
    - Top categorical/text columns by richness
    - Datetime columns
    - Statistical insights about outlier presence and distribution shape
    - Semantic insights about column meaning and category
    """

    # Log the input parameters for transparency
    logger.info(f"Generating KPIs with parameters: min_variability_threshold={min_variability_threshold}, "
                f"min_unique_ratio={min_unique_ratio}, max_unique_ratio={max_unique_ratio}, top_k={top_k}")

    kpis = []
    warnings = []  # Track skipped columns for debugging
    columns = dataset_profile["columns"]
    n_rows = dataset_profile["n_rows"] if dataset_profile.get("n_rows") else len(df)

    if n_rows == 0:
        logger.warning("Empty dataframe provided to KPI generator")
        return kpis

    # ----------------------------
    # 1) Enhanced Numeric "metric" columns with semantic understanding
    # ----------------------------
    numeric_candidates = []
    skipped_numeric = []

    for col in columns:
        if col["role"] != "numeric":
            continue

        series = df[col["name"]].dropna()
        if len(series) == 0:
            skipped_numeric.append((col["name"], "empty series"))
            continue

        stats = col.get("stats") or {}
        std_val = stats.get("std")

        # Check if column has sufficient variability
        if std_val is None or std_val < min_variability_threshold:
            skipped_numeric.append((col["name"], f"low variability (std={std_val})"))
            continue

        # Calculate additional metrics
        outliers_count = _calculate_outliers(series, method='iqr')
        skewness, kurtosis = _calculate_distribution_metrics(series)

        # Semantic analysis
        semantic_categories = _semantic_column_analysis(df, col["name"])

        # Enhanced scoring considering multiple factors
        # Weight by dataset size and magnitude of values
        variability_score = abs(std_val) if std_val else 0
        magnitude_score = abs(series.mean()) if series.notna().any() and len(series) > 0 else 0
        outlier_importance = min(1, outliers_count / max(1, len(series) * 0.1))  # Importance based on % of outliers
        # Distribution-based score (higher for non-normal distributions)
        distribution_score = min(2, abs(skewness) + max(0, kurtosis) * 0.1)

        # Semantic importance bonus
        semantic_bonus = 0
        if 'monetary' in semantic_categories:
            semantic_bonus = 5  # Monetary fields are often important
        elif 'rating' in semantic_categories:
            semantic_bonus = 3  # Rating fields are often important
        elif 'quantity' in semantic_categories:
            semantic_bonus = 2  # Quantity fields are important

        # Combine scores with tie-breaking by magnitude
        combined_score = variability_score + outlier_importance * 10 + distribution_score + semantic_bonus + magnitude_score * 0.01

        numeric_candidates.append((combined_score, col["name"], {
            "std": std_val,
            "outliers_count": outliers_count,
            "outliers_ratio": outliers_count / len(series) if len(series) > 0 else 0,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "semantic_categories": semantic_categories,
            "magnitude_score": magnitude_score
        }))

    # Report skipped numeric columns
    if skipped_numeric:
        logger.info(f"Skipped {len(skipped_numeric)} numeric columns: {skipped_numeric}")
        warnings.extend([f"Numeric column '{name}' skipped: {reason}" for name, reason in skipped_numeric])

    # Sort by combined score descending and take top K
    numeric_candidates.sort(reverse=True, key=lambda x: (x[0], x[2]["magnitude_score"]))  # Primary sort by score, secondary by magnitude
    for i, (_, name, metrics) in enumerate(numeric_candidates[:top_k]):
        # Include semantic info in the value text if available
        semantic_info = f" ({', '.join(metrics['semantic_categories'])})" if metrics['semantic_categories'] else ""
        value_text = f"metric{semantic_info} (outliers: {metrics['outliers_count']}, skew: {metrics['skewness']:.2f})"
        kpis.append({
            "label": name,
            "value": value_text,
            "type": "numeric",
            "outliers_count": metrics["outliers_count"],
            "outliers_ratio": metrics["outliers_ratio"],
            "skewness": metrics["skewness"],
            "kurtosis": metrics["kurtosis"],
            "semantic_categories": metrics["semantic_categories"],
            "score": numeric_candidates[i][0],  # Include score for transparency
            "provenance": "numeric_metric"
        })

    # -----------------------------------------
    # 2) Categorical/text "category" columns with semantic understanding
    # -----------------------------------------
    cat_candidates = []
    skipped_categorical = []

    for col in columns:
        if col["role"] not in ("categorical", "text"):
            continue

        uniq = col.get("unique_count", 0)
        if n_rows <= 0:
            skipped_categorical.append((col["name"], "no data rows"))
            continue
        unique_ratio = uniq / n_rows

        # Skip degenerate ones
        if uniq < 2:
            skipped_categorical.append((col["name"], "only 1 unique value"))
            continue

        # Skip almost-all-unique (look more like IDs)
        if unique_ratio > max_unique_ratio:
            skipped_categorical.append((col["name"], f"too high unique ratio ({unique_ratio:.3f})"))
            continue

        # Skip too few unique (not enough categories to be interesting)
        if unique_ratio < min_unique_ratio:
            skipped_categorical.append((col["name"], f"too low unique ratio ({unique_ratio:.3f})"))
            continue

        # Semantic analysis
        semantic_categories = _semantic_column_analysis(df, col["name"])

        # Score: we like mid/medium richness
        # e.g. 3–100 categories is often interesting
        score = 0.0
        if 0.01 <= unique_ratio <= 0.5:
            score += 10
        elif 0.5 < unique_ratio <= 0.9:
            score += 3
        else:
            score += 1

        # Semantic importance bonus
        if 'identifier' in semantic_categories:
            score -= 5  # Identifiers less important as categories
        elif 'geographic' in semantic_categories:
            score += 2  # Geographic fields often important
        elif 'demographic' in semantic_categories:
            score += 2  # Demographic fields often important

        cat_candidates.append((score, uniq, col["name"], semantic_categories))

    # Report skipped categorical columns
    if skipped_categorical:
        logger.info(f"Skipped {len(skipped_categorical)} categorical columns: {skipped_categorical}")
        warnings.extend([f"Categorical column '{name}' skipped: {reason}" for name, reason in skipped_categorical])

    # Sort: higher score, then more categories
    cat_candidates.sort(key=lambda x: (-x[0], -x[1]))
    for _, _, name, semantic_categories in cat_candidates[:top_k]:
        semantic_info = f" ({', '.join(semantic_categories)})" if semantic_categories else ""
        kpis.append({
            "label": name,
            "value": f"category{semantic_info}",
            "type": "categorical",
            "semantic_categories": semantic_categories,
            "provenance": "categorical_richness"
        })

    # ----------------------------
    # 3) Datetime "time feature" with semantic understanding
    # ----------------------------
    time_cols = [col for col in columns if col["role"] == "datetime"]
    for col in time_cols[:top_k]:
        semantic_categories = _semantic_column_analysis(df, col["name"])
        semantic_info = f" ({', '.join(semantic_categories)})" if semantic_categories else ""
        kpis.append({
            "label": col["name"],
            "value": f"time feature{semantic_info}",
            "type": "datetime",
            "semantic_categories": semantic_categories,
            "provenance": "datetime_feature"
        })

    # ----------------------------
    # 4) Semantic Insights KPIs
    # ----------------------------
    # Add KPIs based on semantic categories
    for col in columns:
        semantic_categories = _semantic_column_analysis(df, col["name"])
        for category in semantic_categories:
            if category in ['monetary', 'rating', 'quantity']:
                kpis.append({
                    "label": f"{category.title()} field: {col['name']}",
                    "value": "High semantic importance",
                    "type": "semantic",
                    "semantic_category": category,
                    "column": col["name"],
                    "provenance": f"semantic_{category}"
                })

    # ----------------------------
    # 5) Statistical Insights KPIs
    # ----------------------------
    # Add correlation insights if we have multiple numeric columns
    try:
        correlations = _calculate_correlations(df)
        for i, (abs_corr, corr_val, col1, col2) in enumerate(correlations[:top_k]):  # Top K correlations
            semantic_categories_1 = _semantic_column_analysis(df, col1)
            semantic_categories_2 = _semantic_column_analysis(df, col2)

            kpis.append({
                "label": f"Correlation: {col1} ↔ {col2}",
                "value": f"{corr_val:.3f}",
                "type": "correlation",
                "correlation_value": corr_val,
                "columns": [col1, col2],
                "semantic_categories_1": semantic_categories_1,
                "semantic_categories_2": semantic_categories_2,
                "provenance": "correlation_insight"
            })
    except Exception as e:
        logger.warning(f"Error calculating correlations: {e}")
        warnings.append(f"Correlation analysis failed: {str(e)}")

    # Add distribution insights for highly skewed variables
    for col in columns:
        if col["role"] == "numeric" and col.get("stats"):
            series = df[col["name"]].dropna()
            if len(series) == 0:
                continue

            skewness, kurtosis = _calculate_distribution_metrics(series)
            if abs(skewness) > 1:  # Highly skewed
                direction = "right" if skewness > 0 else "left"
                semantic_categories = _semantic_column_analysis(df, col["name"])
                kpis.append({
                    "label": f"Skewed Distribution: {col['name']}",
                    "value": f"{direction}-skewed ({skewness:.2f})",
                    "type": "distribution",
                    "skewness": skewness,
                    "semantic_categories": semantic_categories,
                    "provenance": f"distribution_skew_{direction}"
                })

    logger.info(f"Generated {len(kpis)} KPIs")

    # Log any warnings for debugging
    if warnings:
        logger.info(f"KPI generation warnings: {warnings}")

    return kpis
