# src/viz/plotly_renderer.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ChartPayload:
    """Typed chart payload object for validation"""
    title: str
    column: str
    data: List[Dict[str, Any]]
    type: Optional[str] = None
    schema_version: str = "1.0"

def _build_category_count_data(
    df: pd.DataFrame,
    column: str,
    max_categories: int = 10,
):
    """
    Builds category count data with truncation for high-cardinality cases
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in dataframe")
        return None

    # Get value counts
    counts = df[column].value_counts(dropna=False)

    # Handle high cardinality: truncate and add "Others" category
    if len(counts) > max_categories:
        # Keep top N categories and group the rest as "Others"
        top_counts = counts.head(max_categories - 1)  # Leave room for "Others"
        others_count = counts.iloc[max_categories - 1:].sum()

        if others_count > 0:
            top_counts.loc["Others"] = others_count

        logger.info(f"Truncated categories for '{column}' from {len(counts)} to {len(top_counts)} with 'Others' bucket")
        counts = top_counts

    categories = [str(idx) for idx in counts.index]
    values = [int(v) for v in counts.values]

    if not categories:
        logger.warning(f"No valid categories found for column '{column}'")
        return None

    table_data = [
        {"category": cat, "count": val}
        for cat, val in zip(categories, values)
    ]

    # Validate and normalize the chart payload
    chart_payload = ChartPayload(
        title=f"Count of {column}",
        column=column,
        data=table_data,
        type="category_count"
    )

    # Convert to dictionary format expected by the frontend
    return {
        "title": chart_payload.title,
        "column": chart_payload.column,
        "data": chart_payload.data,
        "type": chart_payload.type
    }


def _build_histogram_data(
    df: pd.DataFrame,
    column: str,
    bins: int = 20,
):
    """
    Builds histogram data with adaptive binning strategy based on data size and skew
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in dataframe")
        return None

    # Ensure the column is numeric
    series = pd.to_numeric(df[column], errors='coerce')
    # Drop NaN values
    series = series.dropna()
    if series.empty:
        logger.warning(f"Column '{column}' has no valid numeric values")
        return None

    # Adaptive binning strategy based on data size and skew
    n_samples = len(series)
    if n_samples < 10:
        # Very small dataset - use fewer bins
        bins = 3
    elif n_samples < 50:
        bins = 5
    elif n_samples > 10000:
        # Very large dataset - limit to reasonable number of bins
        bins = min(bins, 100)
    else:
        bins = min(bins, n_samples // 10)  # At most 10 samples per bin

    # Check skewness and adjust if needed
    if n_samples > 2:  # Need at least 3 values to calculate skewness
        try:
            # Calculate skewness using pandas
            skewness = series.skew()
            if abs(skewness) > 1:  # Highly skewed
                # For highly skewed data, use quantile-based bins instead of equal-width bins
                try:
                    # Divide the data into quantile-based bins for skewed distributions
                    quantiles = pd.qcut(series, bins, duplicates='drop', precision=0)
                    value_counts = quantiles.value_counts(sort=False)

                    categories = [f"{interval.left:.2f} - {interval.right:.2f}" for interval in value_counts.index]
                    values = [int(count) for count in value_counts.values]
                except ValueError:
                    # If quantile binning fails, fall back to regular binning
                    hist, bin_edges = pd.cut(series, bins=bins, retbins=True)
                    value_counts = hist.value_counts().sort_index()
                    categories = [f"{interval.left:.2f} - {interval.right:.2f}" for interval in value_counts.index]
                    values = [int(count) for count in value_counts.values]
            else:
                # Normal distribution - use equal-width bins
                hist, bin_edges = pd.cut(series, bins=bins, retbins=True)
                value_counts = hist.value_counts().sort_index()
                categories = [f"{interval.left:.2f} - {interval.right:.2f}" for interval in value_counts.index]
                values = [int(count) for count in value_counts.values]
        except:
            # If skewness calculation fails, use regular binning
            hist, bin_edges = pd.cut(series, bins=bins, retbins=True)
            value_counts = hist.value_counts().sort_index()
            categories = [f"{interval.left:.2f} - {interval.right:.2f}" for interval in value_counts.index]
            values = [int(count) for count in value_counts.values]
    else:
        # Default case for small datasets
        hist, bin_edges = pd.cut(series, bins=bins, retbins=True)
        value_counts = hist.value_counts().sort_index()
        categories = [f"{interval.left:.2f} - {interval.right:.2f}" for interval in value_counts.index]
        values = [int(count) for count in value_counts.values]

    # If there are too many bins with low counts, consider it sparse and reduce bins
    if len(categories) > 10:
        low_count_bins = sum(1 for count in values if count < 2)
        if low_count_bins > len(values) * 0.6:  # 60% of bins have low counts
            # Reduce number of bins
            bins = max(5, len(categories) // 2)
            hist, bin_edges = pd.cut(series, bins=bins, retbins=True)
            value_counts = hist.value_counts().sort_index()
            categories = [f"{interval.left:.2f} - {interval.right:.2f}" for interval in value_counts.index]
            values = [int(count) for count in value_counts.values]

    # Create histogram if there are enough unique values
    if len(categories) < 2:
        # If only one bin, just return that
        unique_val = series.iloc[0] if len(series) > 0 else 0
        categories = [str(unique_val)]
        values = [len(series)]

    if not categories:
        logger.warning(f"No valid categories found for histogram of column '{column}'")
        return None

    table_data = [
        {"category": cat, "count": val}
        for cat, val in zip(categories, values)
    ]

    # Validate and normalize the chart payload
    chart_payload = ChartPayload(
        title=f"Distribution of {column}",
        column=column,
        data=table_data,
        type="histogram"
    )

    return {
        "title": chart_payload.title,
        "column": chart_payload.column,
        "data": chart_payload.data,
        "type": chart_payload.type
    }


def _normalize_axis_labels(title: str, values: List, axis_type: str = "y") -> Dict[str, str]:
    """
    Centralize axis/label formatting to avoid duplication.

    Args:
        title: Title for the axis
        values: List of values for formatting decisions
        axis_type: Either "x" or "y" to determine formatting approach

    Returns:
        Dictionary with formatted labels
    """
    # Detect potential numeric/currency/percentage data for proper formatting
    if values and isinstance(values[0], (int, float)):
        # Check if values look like percentages (0-1 or 0-100)
        all_positive = all(v >= 0 for v in values if isinstance(v, (int, float)))
        max_val = max(values) if values else 0

        if all_positive and max_val <= 1.0:
            # Likely decimal percentages
            return {
                "format": ".1%",
                "suffix": "%",
                "prefix": "",
                "title": f"{title} (%)"
            }
        elif all_positive and max_val <= 100.0:
            # Likely whole number percentages
            return {
                "format": ".0f",
                "suffix": "%",
                "prefix": "",
                "title": f"{title} (%)"
            }
        elif all_positive and max_val > 1000:
            # Large numbers - use compact notation
            return {
                "format": ".2s",  # Compact notation: 1.2K, 1.2M, etc.
                "suffix": "",
                "prefix": "",
                "title": title
            }
        else:
            # Standard numeric values
            return {
                "format": ".2f",
                "suffix": "",
                "prefix": "$" if "price" in title.lower() or "cost" in title.lower() or "revenue" in title.lower() else "",
                "title": title
            }
    else:
        # Non-numeric values, return standard formatting
        return {
            "format": "",
            "suffix": "",
            "prefix": "",
            "title": title
        }


def _build_category_summary_data(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    agg_func: str = "sum",
):
    """
    Builds category summary data with centralized axis/label formatting.
    """
    if x_column not in df.columns or y_column not in df.columns:
        logger.warning(f"One of columns '{x_column}' or '{y_column}' not found in dataframe")
        return None

    # Drop rows where either column is NaN
    subset = df[[x_column, y_column]].dropna()
    if subset.empty:
        logger.warning(f"No valid data for category summary between '{x_column}' and '{y_column}'")
        return None

    # Check if the x_column is suitable for grouping (categorical/low cardinality)
    unique_count = subset[x_column].nunique()
    if unique_count > 50:  # Too many categories for a bar chart
        logger.info(f"Too many unique values ({unique_count}) for column '{x_column}', skipping category summary")
        return None

    # Apply aggregation
    if agg_func == "sum":
        result = subset.groupby(x_column)[y_column].sum()
    elif agg_func == "mean":
        result = subset.groupby(x_column)[y_column].mean()
    elif agg_func == "count":
        result = subset.groupby(x_column)[y_column].count()
    elif agg_func == "min":
        result = subset.groupby(x_column)[y_column].min()
    elif agg_func == "max":
        result = subset.groupby(x_column)[y_column].max()
    else:
        # Default to sum
        result = subset.groupby(x_column)[y_column].sum()

    # Prepare chart data
    categories = [str(idx) for idx in result.index]
    values = [float(val) for val in result.values]

    if not categories:
        logger.warning(f"No valid data points after aggregation for columns '{x_column}' and '{y_column}'")
        return None

    table_data = [
        {"category": cat, "count": val}  # Changed to use "count" field name consistently
        for cat, val in zip(categories, values)
    ]

    # Use centralized axis/label formatting
    y_formatting = _normalize_axis_labels(f"{agg_func.title()} of {y_column}", values, "y")

    # Validate and normalize the chart payload
    chart_payload = ChartPayload(
        title=f"{agg_func.title()} of {y_column} by {x_column}",
        column=f"{x_column}_{agg_func}",
        data=table_data,
        type="category_summary"
    )

    return {
        "title": chart_payload.title,
        "x_column": x_column,
        "y_column": y_column,
        "data": chart_payload.data,
        "type": chart_payload.type,
        "y_formatting": y_formatting  # Include formatting info
    }


def _build_time_series_data(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    agg_func: str = "sum",
):

    if x_column not in df.columns or y_column not in df.columns:
        return None

    # Convert x_column to datetime if it's not already
    x_series = pd.to_datetime(df[x_column], errors='coerce')
    y_series = df[y_column]

    # Combine and drop rows with NaN datetime values
    combined = pd.DataFrame({'x': x_series, 'y': y_series})
    combined = combined.dropna(subset=['x', 'y'])

    if combined.empty:
        return None

    # Sort by date
    combined = combined.sort_values('x')

    # Prepare chart data
    dates = [date.isoformat() for date in combined['x']]
    values = [float(val) for val in combined['y']]

    if not dates:
        return None

    table_data = [
        {"date": date, "value": val}
        for date, val in zip(dates, values)
    ]

    return {
        "title": f"{agg_func.title()} of {y_column} over {x_column}",
        "x_column": x_column,
        "y_column": y_column,
        "data": table_data,
    }


def _build_scatter_data(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
):

    if x_column not in df.columns or y_column not in df.columns:
        return None

    # Drop rows where either column is NaN
    subset = df[[x_column, y_column]].dropna()
    if subset.empty:
        return None

    # Prepare chart data ensuring both x and y are numeric
    x_series = pd.to_numeric(subset[x_column], errors='coerce')
    y_series = pd.to_numeric(subset[y_column], errors='coerce')

    # Combine and remove rows where conversion failed
    combined = pd.DataFrame({'x': x_series, 'y': y_series}).dropna()

    if combined.empty:
        return None

    # Prepare chart data
    x_values = [float(val) for val in combined['x']]
    y_values = [float(val) for val in combined['y']]

    if not x_values or not y_values:
        return None

    table_data = [
        {"x": x_val, "y": y_val}
        for x_val, y_val in zip(x_values, y_values)
    ]

    return {
        "title": f"Scatter plot: {x_column} vs {y_column}",
        "x_column": x_column,
        "y_column": y_column,
        "data": table_data,
    }


def _build_pie_data(
    df: pd.DataFrame,
    column: str,
):
    """
    Builds pie chart data with unit-friendly formatting (percentages).
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in dataframe")
        return None

    # Get value counts
    counts = df[column].value_counts(dropna=True)

    categories = [str(idx) for idx in counts.index]
    values = [int(val) for val in counts.values]

    if not categories:
        logger.warning(f"No valid categories found for pie chart of column '{column}'")
        return None

    # Ensure we have positive values
    table_data = [
        {"category": cat, "value": val}
        for cat, val in zip(categories, values) if val > 0
    ]

    if not table_data:
        logger.warning(f"No positive values found for pie chart of column '{column}'")
        return None

    # Add unit-friendly formatting info to payload
    chart_payload = ChartPayload(
        title=f"Distribution of {column}",
        column=column,
        data=table_data,
        type="pie"
    )

    return {
        "title": chart_payload.title,
        "column": chart_payload.column,
        "data": chart_payload.data,
        "type": chart_payload.type
    }


def _build_box_plot_data(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
):

    if x_column not in df.columns or y_column not in df.columns:
        return None

    # Drop rows where either column is NaN
    subset = df[[x_column, y_column]].dropna()
    if subset.empty:
        return None

    # Ensure y_column is numeric
    y_series = pd.to_numeric(subset[y_column], errors='coerce')
    subset = subset.assign(y_numeric=y_series).dropna(subset=['y_numeric'])

    if subset.empty:
        return None

    # Group by x_column and collect y values for each group
    grouped_data = subset.groupby(x_column)['y_numeric'].apply(list).to_dict()

    categories = [str(cat) for cat in grouped_data.keys()]
    # Values will be lists of values for each category
    values_lists = list(grouped_data.values())

    if not categories or not any(values_lists):
        return None

    table_data = [
        {"category": cat, "values": vals}
        for cat, vals in zip(categories, values_lists)
    ]

    return {
        "title": f"Box plot: {y_column} by {x_column}",
        "x_column": x_column,
        "y_column": y_column,
        "data": table_data,
    }


def _graceful_fallback(data, default_value=None, error_msg="Data processing failed"):
    """
    Provide graceful fallbacks for empty/NaN-heavy series.

    Args:
        data: Input data that might be empty or have many NaN values
        default_value: Value to return if data is invalid
        error_msg: Message to log when fallback is used

    Returns:
        Validated data or default_value
    """
    if data is None:
        logger.warning(error_msg)
        return default_value

    if hasattr(data, '__len__') and len(data) == 0:
        logger.warning(f"{error_msg}: Data is empty")
        return default_value

    if isinstance(data, (pd.DataFrame, pd.Series)):
        if data.empty:
            logger.warning(f"{error_msg}: DataFrame/Series is empty")
            return default_value

        if hasattr(data, 'isna'):
            # Check for high percentage of NaN values
            nan_ratio = data.isna().sum().sum() / data.size if data.size > 0 else 0
            if nan_ratio > 0.9:  # More than 90% NaN values
                logger.warning(f"{error_msg}: Data has high NaN content ({nan_ratio:.2%})")
                return default_value

    return data


def _build_correlation_data(
    df: pd.DataFrame,
    numeric_columns: list,
):
    """
    Builds correlation matrix data with graceful fallbacks for empty/NaN-heavy series.
    """
    if len(numeric_columns) < 2:
        logger.warning(f"Not enough numeric columns ({len(numeric_columns)}) for correlation matrix")
        return _graceful_fallback(None, default_value=None, error_msg="Insufficient numeric columns for correlation")

    # Filter dataframe to only include the specified numeric columns
    subset_df = df[numeric_columns]

    # Convert all columns to numeric, handling errors
    for col in subset_df.columns:
        subset_df[col] = pd.to_numeric(subset_df[col], errors='coerce')

    # Gracefully handle NaN-heavy data
    subset_df = _graceful_fallback(subset_df, default_value=None, error_msg="NaN-heavy dataset after numeric conversion")
    if subset_df is None:
        return None

    # Drop rows with NaN values for correlation calculation
    subset_df = subset_df.dropna()

    # Additional fallback check after dropping NaNs
    if subset_df.empty or len(subset_df.columns) < 2:
        logger.warning("Dataset is empty after dropping NaNs for correlation calculation")
        return _graceful_fallback(None, default_value=None, error_msg="Dataset empty after NaN removal")

    # Calculate correlation matrix
    try:
        corr_matrix = subset_df.corr()

        # Handle case where correlation calculation returns all NaN
        if corr_matrix.isna().all().all():
            logger.warning("All correlations are NaN, likely due to constant values")
            return _graceful_fallback(None, default_value=None, error_msg="All correlations are NaN")

    except Exception as e:
        logger.error(f"Error calculating correlation matrix: {e}")
        return _graceful_fallback(None, default_value=None, error_msg=f"Error in correlation calculation: {e}")

    # Prepare chart data
    categories = [str(col) for col in corr_matrix.columns]
    values_matrix = corr_matrix.values.tolist()

    if not categories:
        logger.warning("No valid categories in correlation matrix")
        return _graceful_fallback(None, default_value=None, error_msg="No valid categories in correlation matrix")

    # Validate correlation values to ensure they're in -1 to 1 range
    for row_idx, row in enumerate(values_matrix):
        for col_idx, val in enumerate(row):
            if pd.isna(val):
                values_matrix[row_idx][col_idx] = 0.0  # Default to 0 for NaN correlations
            elif abs(val) > 1.0:
                # Clamp correlation values to acceptable range
                values_matrix[row_idx][col_idx] = max(-1.0, min(1.0, val))

    table_data = {
        "categories": categories,
        "values": values_matrix
    }

    # Validate and normalize the chart payload
    chart_payload = ChartPayload(
        title="Correlation Matrix",
        column="correlation_matrix",
        data=table_data,
        type="correlation"
    )

    return {
        "title": chart_payload.title,
        "data": chart_payload.data,
        "type": chart_payload.type
    }


def build_charts_from_specs(
    df: pd.DataFrame,
    chart_specs,
    max_categories: int = 10,
    max_charts: int = 20,
):

    charts = {}

    if not chart_specs:
        return charts

    for spec in chart_specs:
        intent = spec.get("intent")
        chart_id = spec.get("id", f"chart_{len(charts)}")

        # Handle different chart intents
        if intent == "category_count":
            col = spec.get("x_field")
            if not col or col in charts:
                continue

            chart_obj = _build_category_count_data(
                df,
                column=col,
                max_categories=max_categories,
            )

        elif intent == "histogram":
            col = spec.get("x_field")
            if not col or chart_id in charts:
                continue

            chart_obj = _build_histogram_data(
                df,
                column=col,
            )

        elif intent == "category_summary":
            x_col = spec.get("x_field")
            y_col = spec.get("y_field")
            agg_func = spec.get("agg_func", "sum")

            if not x_col or not y_col or chart_id in charts:
                continue

            chart_obj = _build_category_summary_data(
                df,
                x_column=x_col,
                y_column=y_col,
                agg_func=agg_func,
            )

        elif intent == "time_series":
            x_col = spec.get("x_field")
            y_col = spec.get("y_field")
            agg_func = spec.get("agg_func", "sum")

            if not x_col or not y_col or chart_id in charts:
                continue

            chart_obj = _build_time_series_data(
                df,
                x_column=x_col,
                y_column=y_col,
                agg_func=agg_func,
            )

        elif intent == "scatter":
            x_col = spec.get("x_field")
            y_col = spec.get("y_field")

            if not x_col or not y_col or chart_id in charts:
                continue

            chart_obj = _build_scatter_data(
                df,
                x_column=x_col,
                y_column=y_col,
            )

        elif intent == "category_pie":
            col = spec.get("x_field")
            if not col or chart_id in charts:
                continue

            chart_obj = _build_pie_data(
                df,
                column=col,
            )

        elif intent == "box_plot":
            x_col = spec.get("x_field")
            y_col = spec.get("y_field")

            if not x_col or not y_col or chart_id in charts:
                continue

            chart_obj = _build_box_plot_data(
                df,
                x_column=x_col,
                y_column=y_col,
            )

        elif intent == "correlation":
            # Get all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2 or chart_id in charts:
                continue

            chart_obj = _build_correlation_data(
                df,
                numeric_columns=numeric_cols,
            )

        else:
            # Unknown intent, skip
            continue

        if chart_obj is None:
            continue

        charts[chart_id] = chart_obj

        if len(charts) >= max_charts:
            break

    return charts


def build_category_count_charts(
    df: pd.DataFrame,
    chart_specs,
    max_categories: int = 10,
    max_charts: int = 20,
):

    charts = {}

    if not chart_specs:
        return charts

    for spec in chart_specs:
        if spec.get("intent") != "category_count":
            continue

        col = spec.get("x_field")
        if not col or col in charts:
            continue

        chart_obj = _build_category_count_data(
            df,
            column=col,
            max_categories=max_categories,
        )

        if chart_obj is None:
            continue

        charts[col] = chart_obj

        if len(charts) >= max_charts:
            break

    return charts