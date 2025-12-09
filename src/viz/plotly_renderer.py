import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from plotly.subplots import make_subplots
import re
import math

logger = logging.getLogger(__name__)

@dataclass
class ChartPayload:
    """Typed chart payload object with validation"""
    title: str
    column: str
    data: List[Dict[str, Any]]
    type: Optional[str] = None
    schema_version: str = "1.0"


def _is_likely_identifier(series: pd.Series, name: str = "") -> bool:
    """
    Robust identifier detection that matches the new correlation engine logic.
    """
    n_total = len(series)
    if n_total == 0:
        return False

    n_unique = series.nunique()
    unique_ratio = n_unique / n_total if n_total > 0 else 0.0

    # High cardinality check
    if unique_ratio > 0.98:
        # Numeric sequential pattern check
        if pd.api.types.is_numeric_dtype(series):
            numeric_vals = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_vals) > 5:
                sorted_vals = numeric_vals.sort_values()
                diffs = sorted_vals.diff().dropna()
                if len(diffs) > 0:
                    # If diffs are mostly 1, likely sequential ID
                    sequential_ratio = (diffs == 1).mean()
                    if sequential_ratio > 0.8:
                        return True
        # UUID pattern check
        if series.dtype == 'object':
            sample = series.dropna().head(20).astype(str)
            uuid_matches = 0
            for val in sample:
                if re.match(r'^[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}$', val, re.IGNORECASE):
                    uuid_matches += 1
            if uuid_matches / len(sample) > 0.5:  # More than 50% are UUIDs
                return True

    # Name-based check
    name_lower = name.lower()
    id_keywords = [
        "id", "uuid", "guid", "key", "code", "no", "number", "index",
        "account", "user", "customer", "product", "item", "order",
        "transaction", "invoice", "booking", "session", "token", "hash"
    ]

    if any(keyword in name_lower for keyword in id_keywords):
        return True

    return False


def _build_category_count_data(
    df: pd.DataFrame,
    column: str,
    max_categories: int = 10,
    dataset_profile: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Builds category count data with intelligent truncation and ID exclusion.
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in dataframe")
        return None

    # Check if this is likely an identifier column
    series = df[column]
    if dataset_profile:
        # Check if column is marked as identifier in dataset profile
        col_profile = next((col for col in dataset_profile.get('columns', []) if col['name'] == column), None)
        if col_profile and col_profile.get('role') == 'identifier':
            logger.info(f"Skipping identifier column '{column}' from category count chart")
            return None
    
    # Also check using our identifier detection function
    if _is_likely_identifier(series, column):
        logger.info(f"Skipping likely identifier column '{column}' from category count chart")
        return None

    # Get value counts
    counts = df[column].value_counts(dropna=True)

    # Handle high cardinality: truncate and add "Others" category
    if len(counts) > max_categories:
        # Keep top N categories and group the rest as "Others"
        top_counts = counts.head(max_categories - 1)  # Leave room for "Others"
        others_count = counts.iloc[max_categories - 1:].sum()

        if others_count > 0:
            top_counts.loc["Others"] = others_count

        logger.info(f"Truncated categories for '{column}' from {len(counts)} to {len(top_counts)} with 'Others' bucket")
        counts = top_counts

    categories = [str(idx) for idx in counts.index if idx is not None]
    values = [int(v) for v in counts.values if not pd.isna(v)]

    if not categories:
        logger.warning(f"No valid categories found for column '{column}' after filtering")
        return None

    table_data = [
        {"category": cat, "count": val}
        for cat, val in zip(categories, values)
    ]

    # Validate and normalize the chart payload
    chart_payload = ChartPayload(
        title=f"Count of {column.replace('_', ' ').title()}",
        column=column,
        data=table_data,
        type="category_count"
    )

    # Convert to dictionary format expected by frontend
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
    dataset_profile: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Builds histogram data with adaptive binning and identifier exclusion.
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in dataframe")
        return None

    # Check if this is likely an identifier column
    series = df[column]
    if dataset_profile:
        # Check if column is marked as identifier in dataset profile
        col_profile = next((col for col in dataset_profile.get('columns', []) if col['name'] == column), None)
        if col_profile and col_profile.get('role') == 'identifier':
            logger.info(f"Skipping identifier column '{column}' from histogram")
            return None
    
    # Also check using our identifier detection function
    if _is_likely_identifier(series, column):
        logger.info(f"Skipping likely identifier column '{column}' from histogram")
        return None

    # Ensure the column is numeric
    series = pd.to_numeric(df[column], errors='coerce')
    # Drop NaN values
    series = series.dropna()
    if series.empty:
        logger.warning(f"Column '{column}' has no valid numeric values after cleaning")
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
    if n_samples > 2 and len(series) > 2:
        try:
            # Calculate skewness
            skewness = series.skew()
            if abs(skewness) > 1:  # Highly skewed
                # For highly skewed data, use quantile-based bins instead of equal-width bins
                try:
                    # Divide the data into quantile-based bins for skewed distributions
                    quantiles = pd.qcut(series, bins, duplicates='drop', precision=1)
                    value_counts = quantiles.value_counts(sort=False)
                    
                    categories = [f"{interval.left:.2f} - {interval.right:.2f}" for interval in value_counts.index]
                    values = [int(count) for count in value_counts.values]
                except Exception:
                    # If quantile binning fails, fall back to regular binning
                    hist, bin_edges = pd.cut(series, bins=bins, retbins=True)
                    value_counts = hist.value_counts(sort=False)  # Sort by bin order, not by count
                    valid_indices = [i for i, interval in enumerate(value_counts.index) if pd.notna(interval.left)]
                    categories = [f"{value_counts.index[i].left:.2f} - {value_counts.index[i].right:.2f}" for i in valid_indices]
                    values = [int(value_counts.iloc[i]) for i in valid_indices]
            else:
                # Normal distribution - use equal-width bins
                hist, bin_edges = pd.cut(series, bins=bins, retbins=True)
                value_counts = hist.value_counts(sort=False)  # Sort by bin order, not by count
                valid_indices = [i for i, interval in enumerate(value_counts.index) if pd.notna(interval.left)]
                categories = [f"{value_counts.index[i].left:.2f} - {value_counts.index[i].right:.2f}" for i in valid_indices]
                values = [int(value_counts.iloc[i]) for i in valid_indices]
        except Exception as e:
            logger.warning(f"Error in adaptive binning for {column}: {e}")
            # Fall back to regular binning
            hist, bin_edges = pd.cut(series, bins=bins, retbins=True)
            value_counts = hist.value_counts(sort=False)  # Sort by bin order, not by count
            valid_indices = [i for i, interval in enumerate(value_counts.index) if pd.notna(interval.left)]
            categories = [f"{value_counts.index[i].left:.2f} - {value_counts.index[i].right:.2f}" for i in valid_indices]
            values = [int(value_counts.iloc[i]) for i in valid_indices]
    else:
        # Default case for small datasets
        hist, bin_edges = pd.cut(series, bins=bins, retbins=True)
        value_counts = hist.value_counts(sort=False)  # Sort by bin order, not by count
        valid_indices = [i for i, interval in enumerate(value_counts.index) if pd.notna(interval.left)]
        categories = [f"{value_counts.index[i].left:.2f} - {value_counts.index[i].right:.2f}" for i in valid_indices]
        values = [int(value_counts.iloc[i]) for i in valid_indices]

    # If there are too many bins with low counts, consider it sparse and reduce bins
    if len(categories) > 10:
        low_count_bins = sum(1 for count in values if count < 2)
        if low_count_bins > len(values) * 0.6:  # 60% of bins have low counts
            # Reduce number of bins
            new_bins = max(5, len(categories) // 2)
            try:
                hist, bin_edges = pd.cut(series, bins=new_bins, retbins=True)
                value_counts = hist.value_counts(sort=False)  # Sort by bin order, not by count
                valid_indices = [i for i, interval in enumerate(value_counts.index) if pd.notna(interval.left)]
                categories = [f"{value_counts.index[i].left:.2f} - {value_counts.index[i].right:.2f}" for i in valid_indices]
                values = [int(value_counts.iloc[i]) for i in valid_indices]
            except Exception as e:
                logger.warning(f"Error adjusting bins for {column}: {e}")

    if not categories:
        if not series.empty:
            # If we have data but couldn't create bins, just use the single value
            single_value = series.iloc[0]
            categories = [str(single_value)]
            values = [len(series)]
        else:
            logger.warning(f"No valid categories found for histogram of column '{column}'")
            return None

    table_data = [
        {"bin_range": cat, "count": val}
        for cat, val in zip(categories, values)
    ]

    # Validate and normalize the chart payload
    chart_payload = ChartPayload(
        title=f"Distribution of {column.replace('_', ' ').title()}",
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


def _build_category_summary_data(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    agg_func: str = "mean",
    dataset_profile: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Builds category vs numeric summary data with ID filtering and semantic awareness.
    """
    if x_column not in df.columns or y_column not in df.columns:
        logger.warning(f"One of columns '{x_column}' or '{y_column}' not found in dataframe")
        return None

    # Check if either column is likely an identifier
    x_series = df[x_column]
    y_series = df[y_column]
    
    if _is_likely_identifier(x_series, x_column) or _is_likely_identifier(y_series, y_column):
        logger.info(f"Skipping summary for columns '{x_column}' and '{y_column}' - one is likely identifier")
        return None

    # Check dataset profile for roles
    if dataset_profile:
        x_profile = next((col for col in dataset_profile.get('columns', []) if col['name'] == x_column), None)
        y_profile = next((col for col in dataset_profile.get('columns', []) if col['name'] == y_column), None)
        
        if x_profile and x_profile.get('role') == 'identifier':
            logger.info(f"Skipping summary - X column '{x_column}' is an identifier")
            return None
        if y_profile and y_profile.get('role') == 'identifier':
            logger.info(f"Skipping summary - Y column '{y_column}' is an identifier")
            return None

    # Drop rows where either column is NaN
    combined_series = pd.concat([x_series, y_series], axis=1).dropna()
    if combined_series.empty:
        logger.warning(f"No valid data for category summary between '{x_column}' and '{y_column}' after dropping NaNs")
        return None

    x_clean = combined_series.iloc[:, 0]
    y_clean = pd.to_numeric(combined_series.iloc[:, 1], errors='coerce')

    # Create valid combined data
    valid_data = pd.concat([x_clean, y_clean], axis=1).dropna()
    if valid_data.empty:
        logger.warning(f"No valid combined data for category summary between '{x_column}' and '{y_column}'")
        return None

    x_final = valid_data.iloc[:, 0]
    y_final = valid_data.iloc[:, 1]

    # Apply aggregation by group
    if agg_func == "sum":
        result = y_final.groupby(x_final).sum()
    elif agg_func == "mean":
        result = y_final.groupby(x_final).mean()
    elif agg_func == "count":
        result = y_final.groupby(x_final).count()
    elif agg_func == "min":
        result = y_final.groupby(x_final).min()
    elif agg_func == "max":
        result = y_final.groupby(x_final).max()
    elif agg_func == "std":
        result = y_final.groupby(x_final).std()
    elif agg_func == "median":
        result = y_final.groupby(x_final).median()
    else:
        # Default to mean
        result = y_final.groupby(x_final).mean()

    # Check if we have enough categories to visualize meaningfully (not too many for readability)
    if len(result) > 20:
        logger.info(f"Too many categories ({len(result)}) for '{x_column}' vs '{y_column}' summary, skipping")
        return None

    categories = [str(idx) for idx in result.index if idx is not None]
    values = [float(val) for val in result.values if pd.notna(val)]

    if not categories:
        logger.warning(f"No valid categories for summary between '{x_column}' and '{y_column}'")
        return None

    table_data = [
        {"category": cat, "agg_value": val}
        for cat, val in zip(categories, values)
    ]

    # Validate and normalize the chart payload
    agg_display = agg_func.title()
    chart_payload = ChartPayload(
        title=f"{agg_display} of {y_column.replace('_', ' ').title()} by {x_column.replace('_', ' ').title()}",
        column=f"agg_{agg_func}_{x_column}",
        data=table_data,
        type="category_summary"
    )

    return {
        "title": chart_payload.title,
        "x_column": x_column,
        "y_column": y_column,
        "data": chart_payload.data,
        "type": chart_payload.type,
        "agg_func": agg_func
    }


def _build_time_series_data(
    df: pd.DataFrame,
    x_column: str,  # datetime column
    y_column: str,  # numeric column
    agg_func: str = "mean",
    dataset_profile: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Builds time series data with datetime validation and identifier exclusion.
    """
    if x_column not in df.columns or y_column not in df.columns:
        logger.warning(f"One of columns '{x_column}' or '{y_column}' not found in dataframe")
        return None

    # Check if X column is datetime-capable
    x_series = df[x_column]
    try:
        x_dt = pd.to_datetime(x_series, errors='coerce')
    except Exception as e:
        logger.warning(f"Cannot convert column '{x_column}' to datetime: {e}")
        return None

    # If X column is likely an identifier, skip
    if _is_likely_identifier(x_series, x_column):
        logger.info(f"Skipping time series - X column '{x_column}' is likely an identifier")
        return None

    # Check dataset profile for roles
    if dataset_profile:
        x_profile = next((col for col in dataset_profile.get('columns', []) if col['name'] == x_column), None)
        y_profile = next((col for col in dataset_profile.get('columns', []) if col['name'] == y_column), None)
        
        if x_profile and x_profile.get('role') == 'identifier':
            logger.info(f"Skipping time series - X column '{x_column}' is an identifier")
            return None
        if y_profile and y_profile.get('role') == 'identifier':
            logger.info(f"Skipping time series - Y column '{y_column}' is an identifier")
            return None

    # Convert Y to numeric
    y_series = df[y_column]
    y_numeric = pd.to_numeric(y_series, errors='coerce')

    # Combine and drop NaN values
    combined = pd.concat([x_dt, y_numeric], axis=1).dropna()
    if combined.empty or len(combined) < 2:
        logger.warning(f"Not enough valid data points for time series between '{x_column}' and '{y_column}'")
        return None

    x_final = combined.iloc[:, 0]
    y_final = combined.iloc[:, 1]

    # Check if time series is meaningful (not just one repeated value)
    if y_final.nunique() < 2:
        logger.info(f"Y column '{y_column}' has less than 2 unique values over time, skipping time series")
        return None

    # Group by date if there are duplicate dates
    if x_final.duplicated().any():
        grouped = y_final.groupby(x_final).agg(agg_func)
        dates = [dt.isoformat() for dt in grouped.index]
        values = [float(val) for val in grouped.values]
    else:
        # Sort by date
        sorted_combined = combined.sort_values(x_column)
        dates = [dt.isoformat() for dt in sorted_combined.iloc[:, 0]]
        values = [float(val) for val in sorted_combined.iloc[:, 1]]

    if not dates:
        logger.warning(f"No valid dates for time series between '{x_column}' and '{y_column}'")
        return None

    table_data = [
        {"date": date, "value": val}
        for date, val in zip(dates, values)
    ]

    # Validate and normalize the chart payload
    chart_payload = ChartPayload(
        title=f"Trend of {y_column.replace('_', ' ').title()} over {x_column.replace('_', ' ').title()}",
        column=f"ts_{x_column}_{y_column}",
        data=table_data,
        type="time_series"
    )

    return {
        "title": chart_payload.title,
        "x_column": x_column,
        "y_column": y_column,
        "data": chart_payload.data,
        "type": chart_payload.type,
        "agg_func": agg_func
    }


def _build_scatter_data(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    dataset_profile: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Builds scatter plot data with identifier exclusion and correlation validation.
    """
    if x_column not in df.columns or y_column not in df.columns:
        logger.warning(f"One of columns '{x_column}' or '{y_column}' not found in dataframe")
        return None

    # Check if either column is likely an identifier
    x_series = df[x_column]
    y_series = df[y_column]
    
    if _is_likely_identifier(x_series, x_column) or _is_likely_identifier(y_series, y_column):
        logger.info(f"Skipping scatter plot for columns '{x_column}' and '{y_column}' - identifier detected")
        return None

    # Check dataset profile for roles
    if dataset_profile:
        x_profile = next((col for col in dataset_profile.get('columns', []) if col['name'] == x_column), None)
        y_profile = next((col for col in dataset_profile.get('columns', []) if col['name'] == y_column), None)
        
        if x_profile and x_profile.get('role') == 'identifier':
            logger.info(f"Skipping scatter - X column '{x_column}' is an identifier")
            return None
        if y_profile and y_profile.get('role') == 'identifier':
            logger.info(f"Skipping scatter - Y column '{y_column}' is an identifier")
            return None

    # Convert both to numeric
    x_numeric = pd.to_numeric(x_series, errors='coerce')
    y_numeric = pd.to_numeric(y_series, errors='coerce')

    # Combine and drop NaN values
    combined = pd.concat([x_numeric, y_numeric], axis=1).dropna()
    if combined.empty or len(combined) < 3:  # Need at least 3 points for meaningful scatter
        logger.warning(f"Not enough valid data points for scatter plot between '{x_column}' and '{y_column}'")
        return None

    x_final = combined.iloc[:, 0]
    y_final = combined.iloc[:, 1]

    # Check for minimal variance (constant or near-constant values)
    x_std = x_final.std()
    y_std = y_final.std()
    
    if pd.isna(x_std) or pd.isna(y_std) or x_std < 0.001 or y_std < 0.001:
        logger.info(f"Low variance in columns '{x_column}' or '{y_column}', skipping scatter plot")
        return None

    # Prepare data
    x_values = [float(val) for val in x_final]
    y_values = [float(val) for val in y_final]

    table_data = [
        {"x": x_val, "y": y_val}
        for x_val, y_val in zip(x_values, y_values)
    ]

    if not x_values or not y_values or len(x_values) != len(y_values):
        logger.warning(f"Mismatched array lengths for scatter plot between '{x_column}' and '{y_column}'")
        return None

    # Validate and normalize the chart payload
    chart_payload = ChartPayload(
        title=f"{x_column.replace('_', ' ').title()} vs {y_column.replace('_', ' ').title()}",
        column=f"scatter_{x_column}_{y_column}",
        data=table_data,
        type="scatter"
    )

    return {
        "title": chart_payload.title,
        "x_column": x_column,
        "y_column": y_column,
        "data": chart_payload.data,
        "type": chart_payload.type
    }


def _build_pie_data(
    df: pd.DataFrame,
    column: str,
    max_categories: int = 10,
    dataset_profile: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Builds pie chart data with cardinality validation and identifier exclusion.
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in dataframe")
        return None

    # Check if this is likely an identifier
    series = df[column]
    if _is_likely_identifier(series, column):
        logger.info(f"Skipping pie chart for column '{column}' - likely identifier")
        return None

    # Check dataset profile for roles
    if dataset_profile:
        col_profile = next((col for col in dataset_profile.get('columns', []) if col['name'] == column), None)
        if col_profile and col_profile.get('role') == 'identifier':
            logger.info(f"Skipping pie chart - column '{column}' is an identifier")
            return None

    # Get value counts
    counts = series.value_counts(dropna=True)

    # For pie charts, limit to a reasonable number of categories for readability
    if len(counts) > max_categories:
        logger.info(f"Too many categories ({len(counts)}) for pie chart of '{column}', only taking top {max_categories}")
        counts = counts.head(max_categories)

    categories = [str(idx) for idx in counts.index if idx is not None]
    values = [int(val) for val in counts.values if pd.notna(val)]

    if not categories:
        logger.warning(f"No valid categories for pie chart of column '{column}'")
        return None

    table_data = [
        {"category": cat, "value": val}
        for cat, val in zip(categories, values)
    ]

    # Validate and normalize the chart payload
    chart_payload = ChartPayload(
        title=f"Distribution of {column.replace('_', ' ').title()}",
        column=column,
        data=table_data,
        type="pie"
    )

    return {
        "title": chart_payload.title,
        "column": column,
        "data": chart_payload.data,
        "type": chart_payload.type
    }


def _build_box_plot_data(
    df: pd.DataFrame,
    x_column: str,  # categorical column
    y_column: str,  # numeric column
    dataset_profile: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Builds box plot data with appropriate validation for identifier exclusion.
    """
    if x_column not in df.columns or y_column not in df.columns:
        logger.warning(f"One of columns '{x_column}' or '{y_column}' not found in dataframe")
        return None

    # Check if either column is likely an identifier
    x_series = df[x_column]
    y_series = df[y_column]
    
    if _is_likely_identifier(x_series, x_column) or _is_likely_identifier(y_series, y_column):
        logger.info(f"Skipping box plot for columns '{x_column}' and '{y_column}' - identifier detected")
        return None

    # Check dataset profile for roles
    if dataset_profile:
        x_profile = next((col for col in dataset_profile.get('columns', []) if col['name'] == x_column), None)
        y_profile = next((col for col in dataset_profile.get('columns', []) if col['name'] == y_column), None)
        
        if x_profile and x_profile.get('role') == 'identifier':
            logger.info(f"Skipping box plot - X column '{x_column}' is an identifier")
            return None
        if y_profile and y_profile.get('role') == 'identifier':
            logger.info(f"Skipping box plot - Y column '{y_column}' is an identifier")
            return None

    # Convert Y to numeric
    y_numeric = pd.to_numeric(y_series, errors='coerce')

    # Combine and drop NaN values
    combined = pd.concat([x_series, y_numeric], axis=1).dropna()
    if combined.empty:
        logger.warning(f"No valid data for box plot between '{x_column}' and '{y_column}'")
        return None

    x_final = combined.iloc[:, 0]
    y_final = combined.iloc[:, 1]

    # Check that X column has multiple categories to compare
    n_unique_x = x_final.nunique()
    if n_unique_x < 2:
        logger.info(f"X column '{x_column}' has less than 2 unique values, skipping box plot")
        return None

    # Limit to reasonable number of categories for readability
    if n_unique_x > 20:
        logger.info(f"X column '{x_column}' has too many categories ({n_unique_x}), skipping box plot")
        return None

    # Group the data for box plot
    grouped = y_final.groupby(x_final)
    
    table_data = []
    for category, values in grouped:
        value_list = [float(v) for v in values if pd.notna(v)]
        if value_list:  # Only add if there are values
            table_data.append({
                "category": str(category),
                "values": value_list
            })

    if not table_data:
        logger.warning(f"No valid data for box plot between '{x_column}' and '{y_column}'")
        return None

    # Validate and normalize the chart payload
    chart_payload = ChartPayload(
        title=f"Distribution of {y_column.replace('_', ' ').title()} by {x_column.replace('_', ' ').title()}",
        column=f"box_{x_column}",
        data=table_data,
        type="box_plot"
    )

    return {
        "title": chart_payload.title,
        "x_column": x_column,
        "y_column": y_column,
        "data": chart_payload.data,
        "type": chart_payload.type
    }


def _build_correlation_heatmap_data(
    df: pd.DataFrame,
    dataset_profile: Dict[str, Any],
    correlation_insights: Optional[List[Dict[str, Any]]] = None
) -> Optional[Dict[str, Any]]:
    """
    Builds correlation heatmap data using the new correlation engine insights.
    Filters out identifiers and low-meaningful correlations.
    """
    if df.empty:
        logger.warning("DataFrame is empty, cannot build correlation heatmap")
        return None

    # Get numeric columns that are not identifiers
    numeric_cols = []
    for col in dataset_profile.get('columns', []):
        if col.get('role') == 'numeric':
            # Check if this column is likely an identifier
            series = df[col['name']]
            if not _is_likely_identifier(series, col['name']):
                numeric_cols.append(col['name'])

    if len(numeric_cols) < 2:
        logger.info(f"Not enough meaningful numeric columns for correlation heatmap ({len(numeric_cols)} found)")
        return None

    # Use only the meaningful numeric columns
    numeric_df = df[numeric_cols]
    
    # Convert to numeric values, handling any potential issues
    for col_name in numeric_df.columns:
        numeric_df[col_name] = pd.to_numeric(numeric_df[col_name], errors='coerce')
    
    # Drop columns that didn't convert properly (ended up with too many NaNs)
    numeric_df = numeric_df.select_dtypes(include=[np.number])
    numeric_cols = [col for col in numeric_cols if col in numeric_df.columns]

    if len(numeric_cols) < 2:
        logger.info(f"After cleaning, not enough meaningful numeric columns for correlation heatmap ({len(numeric_cols)} found)")
        return None

    # Calculate correlation matrix
    try:
        corr_matrix = numeric_df.corr()
        
        # Only include correlations that are meaningful (>0.1 absolute value) or if we have correlation insights
        meaningful_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if pd.notna(corr_val) and abs(corr_val) > 0.1:  # Only meaningful correlations
                    meaningful_corrs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        if not meaningful_corrs:
            logger.info("No meaningful correlations found for heatmap")
            return None

    except Exception as e:
        logger.error(f"Error calculating correlation matrix: {e}")
        return None

    # Prepare data for heatmap
    categories = [str(col) for col in corr_matrix.columns]
    values_matrix = corr_matrix.values.tolist()

    # Ensure valid correlation values in the range [-1, 1]
    for row_idx, row in enumerate(values_matrix):
        for col_idx, val in enumerate(row):
            if pd.isna(val):
                values_matrix[row_idx][col_idx] = 0.0
            else:
                # Clamp correlation values to acceptable range
                values_matrix[row_idx][col_idx] = max(-1.0, min(1.0, val))

    table_data = {
        "categories": categories,
        "values": values_matrix
    }

    # Validate and normalize the chart payload
    chart_payload = ChartPayload(
        title="Correlation Heatmap (Meaningful Relationships Only)",
        column="correlation_matrix",
        data=table_data,
        type="correlation_heatmap"
    )

    return {
        "title": chart_payload.title,
        "data": chart_payload.data,
        "type": chart_payload.type
    }


def build_charts_from_specs(
    df: pd.DataFrame,
    chart_specs,
    dataset_profile: Optional[Dict[str, Any]] = None,
    eda_summary: Optional[Dict[str, Any]] = None,
    max_categories: int = 10,
    max_charts: int = 20,
) -> Dict[str, Any]:
    """
    Intelligent chart suggestion system that considers column roles, semantic tags,
    and meaningful relationships instead of naive heuristics.
    
    Args:
        df: Input DataFrame
        chart_specs: Specifications for what charts to build
        dataset_profile: Dataset profile with column information
        eda_summary: EDA summary with additional insights
        max_categories: Maximum categories for categorical charts
        max_charts: Maximum number of charts to build
        
    Returns:
        Dictionary mapping chart IDs to chart specifications
    """
    if df.empty:
        logger.warning("DataFrame is empty, returning empty charts")
        return {}

    n_rows, n_cols = df.shape
    if n_rows == 0 or n_cols == 0:
        logger.warning(f"Invalid dataframe shape: {n_rows}x{n_cols}")
        return {}

    logger.info(f"Building charts for dataset with {n_rows} rows and {n_cols} columns")

    charts = {}

    if not chart_specs:
        logger.warning("No chart specs provided")
        return charts

    # Process each chart specification
    for spec in chart_specs:
        intent = spec.get('intent')
        chart_id = str(spec.get('id', f'chart_{len(charts)}'))
        
        # Skip if we've reached the maximum charts
        if len(charts) >= max_charts:
            break

        chart_data = None

        try:
            if intent == 'category_count':
                col = spec.get('x_field')
                if col:
                    chart_data = _build_category_count_data(
                        df, 
                        column=col, 
                        max_categories=max_categories,
                        dataset_profile=dataset_profile
                    )

            elif intent == 'histogram':
                col = spec.get('x_field')
                if col:
                    chart_data = _build_histogram_data(
                        df, 
                        column=col,
                        dataset_profile=dataset_profile
                    )

            elif intent == 'category_summary':
                x_col = spec.get('x_field')
                y_col = spec.get('y_field')
                agg_func = spec.get('agg_func', 'mean')
                if x_col and y_col:
                    chart_data = _build_category_summary_data(
                        df, 
                        x_column=x_col, 
                        y_column=y_col, 
                        agg_func=agg_func,
                        dataset_profile=dataset_profile
                    )

            elif intent == 'time_series':
                x_col = spec.get('x_field')
                y_col = spec.get('y_field')
                agg_func = spec.get('agg_func', 'mean')
                if x_col and y_col:
                    chart_data = _build_time_series_data(
                        df, 
                        x_column=x_col, 
                        y_column=y_col, 
                        agg_func=agg_func,
                        dataset_profile=dataset_profile
                    )

            elif intent == 'scatter':
                x_col = spec.get('x_field')
                y_col = spec.get('y_field')
                if x_col and y_col:
                    chart_data = _build_scatter_data(
                        df, 
                        x_column=x_col, 
                        y_column=y_col,
                        dataset_profile=dataset_profile
                    )

            elif intent == 'category_pie':
                col = spec.get('x_field')
                if col:
                    chart_data = _build_pie_data(
                        df, 
                        column=col,
                        max_categories=max_categories,
                        dataset_profile=dataset_profile
                    )

            elif intent == 'box_plot':
                x_col = spec.get('x_field')
                y_col = spec.get('y_field')
                if x_col and y_col:
                    chart_data = _build_box_plot_data(
                        df,
                        x_column=x_col,
                        y_column=y_col,
                        dataset_profile=dataset_profile
                    )

            elif intent == 'correlation_matrix':
                # Use the new correlation heatmap builder that properly filters identifiers
                chart_data = _build_correlation_heatmap_data(
                    df,
                    dataset_profile=dataset_profile,
                    correlation_insights=eda_summary.get('correlation_insights', []) if eda_summary else []
                )

            else:
                logger.warning(f"Unknown chart intent: {intent}")
                continue

            # Add chart data if valid and not already added
            if chart_data and chart_id not in charts:
                charts[chart_id] = chart_data

        except Exception as e:
            logger.error(f"Error building chart with intent '{intent}' and ID '{chart_id}': {e}")
            continue

    logger.info(f"Built {len(charts)} valid charts from {len(chart_specs) if chart_specs else 0} specifications")
    return charts


def build_category_count_charts(
    df: pd.DataFrame,
    chart_specs,
    dataset_profile: Optional[Dict[str, Any]] = None,
    max_categories: int = 10,
    max_charts: int = 20,
) -> Dict[str, Any]:
    """
    Builds multiple category count charts with intelligent filtering and ID exclusion.
    """
    if df.empty:
        logger.warning("DataFrame is empty, returning empty category charts")
        return {}

    category_charts = {}

    if not chart_specs:
        logger.warning("No chart specs provided for category charts")
        return category_charts

    for spec in chart_specs:
        if spec.get("intent") != "category_count":
            continue

        col = spec.get("x_field")
        if not col or col in category_charts:
            continue

        # Check if column is likely an identifier before building chart
        series = df[col]
        if _is_likely_identifier(series, col):
            logger.info(f"Skipping identifier column '{col}' from category count chart")
            continue

        # Check dataset profile for roles
        if dataset_profile:
            col_profile = next((c for c in dataset_profile.get('columns', []) if c['name'] == col), None)
            if col_profile and col_profile.get('role') == 'identifier':
                logger.info(f"Skipping identifier column '{col}' from category count chart")
                continue

        chart_obj = _build_category_count_data(
            df,
            column=col,
            max_categories=max_categories,
            dataset_profile=dataset_profile
        )

        if chart_obj is not None:
            category_charts[col] = chart_obj

        if len(category_charts) >= max_charts:
            break

    logger.info(f"Built {len(category_charts)} category count charts")
    return category_charts