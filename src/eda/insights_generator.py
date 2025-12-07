"""
Advanced EDA and Insights Generator for the ML Dashboard
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from scipy.stats import pearsonr
from collections import Counter
import re

logger = logging.getLogger(__name__)

def detect_pattern_relationships(df: pd.DataFrame, dataset_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the dataset to detect patterns, relationships, and correlations
    """
    n_rows = dataset_profile.get("n_rows", len(df))
    if n_rows == 0:
        logger.warning("Empty dataframe provided to pattern detection")
        return {
            "correlations": [],
            "trends": [],
            "patterns": [],
            "outliers": [],
            "anomalies": []
        }

    results = {
        "correlations": [],
        "trends": [],
        "patterns": [],
        "outliers": [],
        "anomalies": [],
        "distribution_insights": []
    }

    # 1. Find correlations between numeric columns
    numeric_cols = [col['name'] for col in dataset_profile['columns'] if col['role'] == 'numeric']
    
    if len(numeric_cols) >= 2:
        # Calculate pair-wise correlations
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                
                # Convert to numeric and drop NaN
                series1 = pd.to_numeric(df[col1], errors='coerce')
                series2 = pd.to_numeric(df[col2], errors='coerce')
                
                # Remove NaN values
                mask = ~(series1.isna() | series2.isna())
                s1_clean = series1[mask]
                s2_clean = series2[mask]
                
                if len(s1_clean) > 2:  # Need at least 3 points for correlation
                    try:
                        corr, p_value = pearsonr(s1_clean, s2_clean)
                        
                        if not np.isnan(corr):
                            results["correlations"].append({
                                "variable1": col1,
                                "variable2": col2,
                                "correlation": corr,
                                "p_value": p_value,
                                "strength": "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak",
                                "type": "positive" if corr > 0 else "negative"
                            })
                    except Exception as e:
                        logger.warning(f"Error calculating correlation between {col1} and {col2}: {e}")
    
    # 2. Detect potential trends in time-based data
    datetime_cols = [col['name'] for col in dataset_profile['columns'] if col['role'] == 'datetime']
    
    for dt_col in datetime_cols:
        for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            try:
                dt_series = pd.to_datetime(df[dt_col], errors='coerce')
                num_series = pd.to_numeric(df[num_col], errors='coerce')
                
                # Create a valid data frame
                temp_df = pd.DataFrame({dt_col: dt_series, num_col: num_series}).dropna()
                
                if len(temp_df) > 2:
                    # Use the index as a proxy for time and calculate correlation
                    temp_df = temp_df.sort_values(dt_col)
                    temp_df['time_index'] = range(len(temp_df))
                    
                    trend_corr, p_value = pearsonr(temp_df['time_index'], temp_df[num_col])
                    
                    if not np.isnan(trend_corr):
                        trend_type = "increasing" if trend_corr > 0.1 else "decreasing" if trend_corr < -0.1 else "stable"
                        results["trends"].append({
                            "datetime_column": dt_col,
                            "numeric_column": num_col,
                            "trend_correlation": trend_corr,
                            "trend_type": trend_type,
                            "p_value": p_value
                        })
            except Exception as e:
                logger.warning(f"Error detecting trend for {dt_col} and {num_col}: {e}")
    
    # 3. Identify patterns in categorical data
    categorical_cols = [col['name'] for col in dataset_profile['columns'] if col['role'] in ['categorical', 'boolean']]
    
    for col in categorical_cols:
        try:
            # Most common categories
            value_counts = df[col].value_counts()
            total_count = len(df[col])
            
            # Identify dominant categories (>20% of the data)
            dominant_categories = []
            for cat, count in value_counts.items():
                ratio = count / total_count
                if ratio > 0.2:  # More than 20% of the data
                    dominant_categories.append({
                        "category": cat,
                        "count": count,
                        "percentage": ratio * 100
                    })
            
            if dominant_categories:
                results["patterns"].append({
                    "column": col,
                    "pattern_type": "dominant_categories",
                    "categories": dominant_categories
                })
                
            # Identify low-variety categories (high concentration)
            unique_count = len(value_counts)
            if unique_count > 1 and total_count > 0:
                entropy = -sum((count/total_count) * np.log2(count/total_count) for count in value_counts if count > 0)
                max_entropy = np.log2(unique_count)
                
                if max_entropy > 0:
                    normalized_entropy = entropy / max_entropy
                    if normalized_entropy < 0.5:  # Low entropy = high concentration
                        results["patterns"].append({
                            "column": col,
                            "pattern_type": "low_entropy",
                            "entropy": normalized_entropy,
                            "unique_values": unique_count
                        })
        except Exception as e:
            logger.warning(f"Error detecting patterns for column {col}: {e}")
    
    # 4. Detect outliers in numeric data
    for col in numeric_cols:
        try:
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            
            if len(series) > 4:  # Need at least 5 values to detect outliers meaningfully
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                
                if len(outliers) > 0:
                    results["outliers"].append({
                        "column": col,
                        "outlier_count": len(outliers),
                        "outlier_percentage": (len(outliers) / len(series)) * 100,
                        "outlier_values": outliers.head(10).tolist()  # Limit to first 10 outliers
                    })
        except Exception as e:
            logger.warning(f"Error detecting outliers for column {col}: {e}")
    
    # 5. Detect anomalies based on data distribution
    for col in dataset_profile['columns']:
        try:
            if col['role'] in ['categorical', 'text'] and col['unique_count'] > 1:
                # Detect potential data quality issues
                value_counts = df[col['name']].value_counts()
                
                # Check for very low frequency values that might be typos
                rare_values = value_counts[value_counts < 3]  # Values that appear less than 3 times
                if len(rare_values) > 0:
                    results["anomalies"].append({
                        "column": col['name'],
                        "anomaly_type": "rare_values",
                        "rare_values_count": len(rare_values),
                        "example_values": rare_values.head(5).index.tolist()
                    })
                    
                # Check for highly imbalanced distributions
                if len(value_counts) > 2:
                    total = sum(value_counts)
                    largest_category = value_counts.iloc[0]
                    if largest_category / total > 0.95:  # 95% of values are the same
                        results["anomalies"].append({
                            "column": col['name'],
                            "anomaly_type": "highly_imbalanced",
                            "largest_category_ratio": largest_category / total,
                            "largest_category": value_counts.index[0]
                        })
        except Exception as e:
            logger.warning(f"Error detecting anomalies for column {col['name']}: {e}")
    
    # 6. Generate distribution insights
    for col in numeric_cols:
        try:
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            
            if len(series) > 2:
                # Calculate skewness and kurtosis
                skewness = series.skew()
                kurtosis = series.kurtosis()
                
                results["distribution_insights"].append({
                    "column": col,
                    "skewness": skewness,
                    "kurtosis": kurtosis,
                    "distribution_type": "right_skewed" if skewness > 1 else "left_skewed" if skewness < -1 else "symmetric",
                    "tail_type": "heavy_tailed" if kurtosis > 0 else "light_tailed"
                })
        except Exception as e:
            logger.warning(f"Error calculating distribution insights for column {col}: {e}")
    
    return results


def extract_use_cases(df: pd.DataFrame, dataset_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract potential use cases from the dataset based on content analysis
    """
    use_cases = []

    n_rows = dataset_profile.get("n_rows", len(df))
    if n_rows == 0:
        return []

    # Extract semantic clues from column names and types
    column_names = [col['name'] for col in dataset_profile['columns']]

    # Identify potential use cases based on column patterns
    semantic_categories = {
        'sales': {
            'keywords': ['price', 'cost', 'revenue', 'sales', 'amount', 'profit', 'fee', 'charge', 'payment', 'discount', 'tax', 'margin'],
            'key_inputs': ['product_id', 'customer_id', 'order_date', 'quantity'],
            'indicators': ['total_revenue', 'profit_margin', 'sales_volume']
        },
        'demographics': {
            'keywords': ['age', 'gender', 'race', 'ethnicity', 'income', 'education', 'occupation', 'family', 'children', 'marital', 'birth'],
            'key_inputs': ['id', 'date_of_birth', 'location', 'survey_date'],
            'indicators': ['avg_income', 'education_level_distribution', 'age_median']
        },
        'location': {
            'keywords': ['city', 'state', 'country', 'address', 'location', 'region', 'zip', 'postal', 'latitude', 'longitude', 'area'],
            'key_inputs': ['coordinates', 'administrative_divisions', 'time_zone'],
            'indicators': ['population_density', 'geographic_distribution', 'distance_metrics']
        },
        'time': {
            'keywords': ['date', 'time', 'day', 'week', 'month', 'year', 'season', 'period', 'duration', 'timestamp', 'hour', 'minute'],
            'key_inputs': ['id', 'event_type', 'start_time', 'end_time'],
            'indicators': ['trend_over_time', 'seasonal_patterns', 'frequency']
        },
        'rating': {
            'keywords': ['rating', 'score', 'grade', 'review', 'feedback', 'satisfaction', 'rating_count', 'stars', 'vote'],
            'key_inputs': ['reviewer_id', 'item_id', 'review_text', 'review_date'],
            'indicators': ['avg_rating', 'rating_distribution', 'review_sentiment']
        },
        'quantity': {
            'keywords': ['count', 'quantity', 'number', 'volume', 'size', 'frequency', 'frequency', 'instances', 'cases', 'instances'],
            'key_inputs': ['item_id', 'category', 'measurement_unit'],
            'indicators': ['total_count', 'avg_quantity', 'distribution']
        },
        'health': {
            'keywords': ['patient', 'diagnosis', 'treatment', 'symptom', 'medication', 'disease', 'condition', 'blood_pressure', 'pulse', 'temperature'],
            'key_inputs': ['patient_id', 'doctor_id', 'diagnosis_date', 'symptom onset'],
            'indicators': ['recovery_rate', 'diagnosis_distribution', 'treatment_success']
        },
        'education': {
            'keywords': ['student', 'grade', 'score', 'subject', 'school', 'enrollment', 'test', 'exam', 'course', 'gpa', 'attendance'],
            'key_inputs': ['student_id', 'course_id', 'exam_date', 'instructor'],
            'indicators': ['avg_score', 'pass_rate', 'attendance_rate']
        },
        'finance': {
            'keywords': ['account', 'balance', 'transaction', 'credit', 'loan', 'interest', 'investment', 'portfolio', 'return', 'equity'],
            'key_inputs': ['account_id', 'transaction_date', 'counterparty', 'reference'],
            'indicators': ['cash_flow', 'return_on_investment', 'risk_metrics']
        },
        'technology': {
            'keywords': ['device', 'os', 'platform', 'software', 'version', 'model', 'type', 'cpu', 'memory', 'resolution', 'bandwidth'],
            'key_inputs': ['device_id', 'manufacturer', 'release_date', 'specifications'],
            'indicators': ['usage_patterns', 'performance_metrics', 'adoption_rate']
        },
        'transportation': {
            'keywords': ['vehicle', 'model', 'year', 'mileage', 'route', 'trip', 'distance', 'speed', 'fuel', 'departure', 'arrival'],
            'key_inputs': ['vehicle_id', 'driver_id', 'route_id', 'timestamp'],
            'indicators': ['avg_speed', 'fuel_efficiency', 'on_time_rate']
        },
        'marketing': {
            'keywords': ['campaign', 'click', 'impression', 'conversion', 'revenue', 'cost', 'cpc', 'cpa', 'roi', 'engagement'],
            'key_inputs': ['campaign_id', 'ad_group', 'keyword', 'audience'],
            'indicators': ['conversion_rate', 'roi', 'cost_per_conversion']
        },
        'retail': {
            'keywords': ['product', 'inventory', 'stock', 'sku', 'brand', 'category', 'supplier', 'shelf', 'vendor', 'order'],
            'key_inputs': ['product_id', 'store_id', 'supplier_id', 'reorder_date'],
            'indicators': ['inventory_turnover', 'stockout_rate', 'profit_margin']
        },
        'social_media': {
            'keywords': ['user', 'post', 'like', 'comment', 'share', 'follower', 'engagement', 'reach', 'impression', 'hashtag'],
            'key_inputs': ['user_id', 'post_id', 'timestamp', 'content_type'],
            'indicators': ['engagement_rate', 'follower_growth', 'content_popularity']
        }
    }

    # Find which semantic categories are present in the dataset
    present_categories = []
    for cat, info in semantic_categories.items():
        category_match = False

        # Check keywords in column names
        for col_name in column_names:
            if any(keyword in col_name.lower() for keyword in info['keywords']):
                category_match = True
                break

        # Check if this category is present
        if category_match:
            present_categories.append((cat, info))

    # Generate use case suggestions based on detected semantic categories
    for cat, info in present_categories:
        # Identify key columns relevant to this category
        key_columns = [name for name in column_names if any(kw in name.lower() for kw in info['keywords'])]

        # Add key inputs if they exist in the dataset
        for input_col in info['key_inputs']:
            if input_col in column_names and input_col not in key_columns:
                key_columns.append(input_col)

        # Use case specific to this category
        if cat == 'sales':
            use_cases.append({
                "use_case": "Sales Performance Analysis",
                "description": "Analyze product performance, revenue trends, and sales metrics",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Revenue by time", "Top selling products", "Sales by category", "Profit margins"]
            })
        elif cat == 'demographics':
            use_cases.append({
                "use_case": "Demographic Analysis",
                "description": "Understand customer or subject demographics and patterns",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Age distribution", "Gender breakdown", "Income vs other factors", "Education level"]
            })
        elif cat == 'location':
            use_cases.append({
                "use_case": "Geographic Analysis",
                "description": "Analyze geographic patterns and location-based trends",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Sales by region", "Geographic distribution", "Location vs other metrics", "Heatmaps"]
            })
        elif cat == 'time':
            use_cases.append({
                "use_case": "Time Series Analysis",
                "description": "Analyze trends, seasonality, and time-based patterns",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Trends over time", "Seasonal patterns", "Period comparisons", "Moving averages"]
            })
        elif cat == 'rating':
            use_cases.append({
                "use_case": "Performance Rating Analysis",
                "description": "Analyze ratings, scores, and performance metrics",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Rating distributions", "Average scores by category", "Rating trends", "Review sentiment"]
            })
        elif cat == 'health':
            use_cases.append({
                "use_case": "Healthcare Analysis",
                "description": "Analyze patient data, treatments, and health outcomes",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Condition distribution", "Treatment effectiveness", "Patient demographics", "Health metrics over time"]
            })
        elif cat == 'education':
            use_cases.append({
                "use_case": "Educational Performance Analysis",
                "description": "Analyze student performance, course outcomes, and educational metrics",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Average scores by subject", "Pass rates", "Attendance patterns", "Performance trends"]
            })
        elif cat == 'finance':
            use_cases.append({
                "use_case": "Financial Analysis",
                "description": "Analyze financial performance, risk, and investment outcomes",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Revenue trends", "Risk metrics", "Investment returns", "Cash flow analysis"]
            })
        elif cat == 'technology':
            use_cases.append({
                "use_case": "Technology Usage Analysis",
                "description": "Analyze device usage, software adoption, and technology performance",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Device usage patterns", "Software adoption", "Performance metrics", "Technology trends"]
            })
        elif cat == 'transportation':
            use_cases.append({
                "use_case": "Transportation Analysis",
                "description": "Analyze route efficiency, vehicle performance, and transportation metrics",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Average speed patterns", "Fuel efficiency", "On-time performance", "Route optimization"]
            })
        elif cat == 'marketing':
            use_cases.append({
                "use_case": "Marketing Campaign Analysis",
                "description": "Analyze campaign performance, conversion rates, and marketing ROI",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Conversion rates", "ROI by channel", "Cost per acquisition", "Engagement metrics"]
            })
        elif cat == 'retail':
            use_cases.append({
                "use_case": "Retail Operations Analysis",
                "description": "Analyze inventory, sales, and retail operational metrics",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Inventory turnover", "Product performance", "Stock levels", "Sales patterns"]
            })
        elif cat == 'social_media':
            use_cases.append({
                "use_case": "Social Media Engagement Analysis",
                "description": "Analyze user engagement, content performance, and social metrics",
                "key_inputs": key_columns,
                "key_indicators": info['indicators'],
                "suggested_visualizations": ["Engagement rates", "Follower growth", "Content popularity", "User activity patterns"]
            })

    # Generate a general use case if no specific ones were detected
    if not use_cases:
        use_cases.append({
            "use_case": "General Data Exploration",
            "description": "Explore and understand the structure and content of the dataset",
            "key_inputs": column_names[:5],  # First 5 columns
            "key_indicators": ["data_completeness", "uniqueness", "data_types"],
            "suggested_visualizations": ["Column distributions", "Missing value patterns", "Data types overview"]
        })

    # Add cross-domain use cases if multiple categories are detected
    if len(present_categories) > 1:
        # Identify common columns that might connect different categories
        common_columns = []
        for col in dataset_profile['columns']:
            col_name = col['name']
            # Look for ID columns that might connect different domains
            if 'id' in col_name.lower() or 'key' in col_name.lower():
                common_columns.append(col_name)

        if common_columns:
            use_cases.append({
                "use_case": "Cross-Domain Analysis",
                "description": "Analyze relationships between different data domains using common identifiers",
                "key_inputs": common_columns,
                "key_indicators": ["connection_strength", "data_integration_points"],
                "suggested_visualizations": ["Domain correlations", "Common identifier distributions", "Cross-domain patterns"]
            })

    return use_cases


def identify_key_indicators(df: pd.DataFrame, dataset_profile: Dict[str, Any], correlations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Identify key indicators based on data patterns, correlations, and statistical significance
    """
    key_indicators = []

    n_rows = dataset_profile.get("n_rows", len(df))
    if n_rows == 0:
        return []

    # 1. High-impact numeric columns (high variance or frequently correlated)
    numeric_cols = [col['name'] for col in dataset_profile['columns'] if col['role'] == 'numeric']

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors='coerce').dropna()

        if len(series) > 0:
            # Calculate the coefficient of variation (std/mean) - high values indicate high variability
            mean_val = series.mean()
            std_val = series.std()

            if mean_val != 0:  # Avoid division by zero
                cv = abs(std_val / mean_val) if std_val is not None else 0
            else:
                cv = std_val if std_val is not None else 0

            # Count how many times this column appears in strong correlations
            correlation_count = sum(1 for corr in correlations
                                  if corr['strength'] == 'strong' and col in [corr['variable1'], corr['variable2']])

            # Calculate outlier impact
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = len(series[(series < lower_bound) | (series > upper_bound)])
            outlier_ratio = outliers / len(series) if len(series) > 0 else 0

            # Calculate skewness
            skewness = series.skew()

            key_indicators.append({
                "indicator": col,
                "indicator_type": "numeric",
                "significance_score": cv + correlation_count * 0.5 + outlier_ratio * 2,  # Combine variability, correlation, and outlier impact
                "metric_type": "continuous",
                "description": f"Numeric column with {correlation_count} strong correlation(s), CV of {cv:.2f}, and {outlier_ratio:.2%} outliers",
                "statistical_properties": {
                    "mean": float(mean_val),
                    "std": float(std_val) if std_val is not None else 0,
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "coefficient_of_variation": cv,
                    "skewness": skewness,
                    "outlier_ratio": outlier_ratio
                }
            })

    # 2. High-impact categorical columns (high cardinality or low entropy)
    categorical_cols = [col['name'] for col in dataset_profile['columns'] if col['role'] in ['categorical', 'boolean']]

    for col in categorical_cols:
        series = df[col].dropna()
        unique_count = series.nunique()
        total_count = len(series)

        if total_count > 0:
            # Calculate entropy to understand distribution diversity
            value_counts = series.value_counts()
            probs = value_counts / total_count
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            max_entropy = np.log2(unique_count) if unique_count > 0 else 0

            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

            # Calculate imbalance ratio (how much the most common category dominates)
            most_common_ratio = value_counts.iloc[0] / total_count if len(value_counts) > 0 else 0

            key_indicators.append({
                "indicator": col,
                "indicator_type": "categorical",
                "significance_score": (1 - normalized_entropy) * unique_count + (most_common_ratio * 10),  # Higher score for low entropy + high unique count + high imbalance
                "metric_type": "categorical",
                "description": f"Categorical column with {unique_count} unique values, normalized entropy of {normalized_entropy:.2f}, and max category ratio of {most_common_ratio:.2f}",
                "statistical_properties": {
                    "unique_count": unique_count,
                    "total_count": total_count,
                    "normalized_entropy": normalized_entropy,
                    "most_common_value": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "most_common_ratio": most_common_ratio,
                    "top_categories": [
                        {"value": str(idx), "count": int(cnt), "percentage": f"{(cnt/total_count)*100:.2f}%"}
                        for idx, cnt in value_counts.head(5).items()
                    ]
                }
            })

    # 3. DateTime columns (time-based indicators)
    datetime_cols = [col['name'] for col in dataset_profile['columns'] if col['role'] == 'datetime']

    for col in datetime_cols:
        try:
            dt_series = pd.to_datetime(df[col], errors='coerce').dropna()

            if len(dt_series) > 0:
                time_span = dt_series.max() - dt_series.min()
                time_span_days = time_span.days if hasattr(time_span, 'days') else time_span.total_seconds() / 86400

                key_indicators.append({
                    "indicator": col,
                    "indicator_type": "datetime",
                    "significance_score": time_span_days,  # Longer time spans may be more significant
                    "metric_type": "datetime",
                    "description": f"Datetime column spanning {time_span_days:.2f} days",
                    "statistical_properties": {
                        "min_date": dt_series.min().isoformat() if not pd.isna(dt_series.min()) else None,
                        "max_date": dt_series.max().isoformat() if not pd.isna(dt_series.max()) else None,
                        "time_span_days": time_span_days,
                        "total_observations": len(dt_series)
                    }
                })
        except Exception as e:
            logger.warning(f"Error processing datetime column {col}: {e}")

    # Sort indicators by significance score in descending order
    key_indicators.sort(key=lambda x: x['significance_score'], reverse=True)

    return key_indicators


def generate_eda_summary(df: pd.DataFrame, dataset_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a comprehensive EDA summary for the dataset
    """
    logger.info("Starting EDA analysis for dataset")
    
    # Perform pattern and relationship analysis
    patterns = detect_pattern_relationships(df, dataset_profile)
    
    # Extract potential use cases
    use_cases = extract_use_cases(df, dataset_profile)
    
    # Identify key indicators
    key_indicators = identify_key_indicators(df, dataset_profile, patterns['correlations'])
    
    # Create the EDA summary object
    eda_summary = {
        "summary_statistics": {
            "total_rows": dataset_profile.get("n_rows", len(df)),
            "total_columns": dataset_profile.get("n_cols", len(df.columns)),
            "numeric_columns": len([c for c in dataset_profile['columns'] if c['role'] == 'numeric']),
            "categorical_columns": len([c for c in dataset_profile['columns'] if c['role'] in ['categorical', 'boolean']]),
            "datetime_columns": len([c for c in dataset_profile['columns'] if c['role'] == 'datetime']),
            "text_columns": len([c for c in dataset_profile['columns'] if c['role'] == 'text'])
        },
        "patterns_and_relationships": patterns,
        "use_cases": use_cases,
        "key_indicators": key_indicators,
        "recommendations": generate_recommendations(df, dataset_profile, patterns, use_cases, key_indicators)
    }
    
    logger.info("EDA analysis completed")
    return eda_summary


def generate_recommendations(df: pd.DataFrame, dataset_profile: Dict[str, Any], 
                           patterns: Dict[str, Any], use_cases: List[Dict[str, Any]], 
                           key_indicators: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate actionable recommendations based on EDA analysis
    """
    recommendations = []
    
    # Recommendation based on correlations
    strong_correlations = [corr for corr in patterns['correlations'] if corr['strength'] == 'strong']
    if strong_correlations:
        recommendations.append({
            "type": "correlation_insight",
            "title": "Strong correlations detected",
            "description": f"Detected {len(strong_correlations)} strong correlations. These variables may have causal relationships or shared underlying factors.",
            "details": strong_correlations[:3]  # Limit to top 3 for brevity
        })
    
    # Recommendation based on outliers
    outlier_columns = [out for out in patterns['outliers'] if out['outlier_percentage'] > 5]
    if outlier_columns:
        recommendations.append({
            "type": "data_quality",
            "title": "Potential data quality issues",
            "description": f"Detected outliers in {len(outlier_columns)} columns (>5% of values). These may be data entry errors or genuine extreme values.",
            "details": outlier_columns[:3]  # Limit to top 3 for brevity
        })
    
    # Recommendation based on anomalies
    if patterns['anomalies']:
        recommendations.append({
            "type": "data_insight",
            "title": "Anomalies detected",
            "description": f"Found {len(patterns['anomalies'])} anomalies in the dataset that may require further investigation.",
            "details": patterns['anomalies'][:3]  # Limit to top 3 for brevity
        })
    
    # Recommendation based on trends
    if patterns['trends']:
        trend_directions = [trend['trend_type'] for trend in patterns['trends']]
        increasing_trends = trend_directions.count('increasing')
        decreasing_trends = trend_directions.count('decreasing')
        
        recommendations.append({
            "type": "trend_analysis",
            "title": "Time-based trends identified",
            "description": f"Found {len(patterns['trends'])} time-based trends ({increasing_trends} increasing, {decreasing_trends} decreasing).",
            "details": patterns['trends'][:3]  # Limit to top 3 for brevity
        })
    
    # Recommendation based on use cases
    if use_cases:
        recommendations.append({
            "type": "use_case",
            "title": "Suggested use cases for this dataset",
            "description": "Based on the content of your dataset, you could focus on these analytical approaches:",
            "details": use_cases[:2]  # Limit to top 2 for brevity
        })
    
    # Recommendation based on key indicators
    if key_indicators:
        top_indicators = key_indicators[:3]  # Top 3 indicators
        recommendations.append({
            "type": "key_indicator",
            "title": "Key indicators to focus on",
            "description": "These variables have high significance scores and should be prioritized in your analysis:",
            "details": top_indicators
        })
    
    return recommendations