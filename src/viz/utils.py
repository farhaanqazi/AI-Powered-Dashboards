import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re
import math
from src.utils.identifier_detector import is_likely_identifier

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
    # Use the centralized identifier detector
    return is_likely_identifier(series, name)


def _build_category_count_data(
    df: pd.DataFrame,
    column: str,
    max_categories: int = 10,
    dataset_profile: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Builds category count data with intelligent truncation and ID exclusion.
    """
    if df is None or column not in df.columns:
        logger.warning(f"DataFrame is None or column '{column}' not found in dataframe")
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

    # Get value counts, but first ensure the series is clean of problematic values
    # Drop NaN values and ensure we're working with a proper pandas series
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    series_clean = series.dropna()

    # For string-like values, convert to string representation but handle problematic values
    series_clean = series_clean.astype(str)

    # Remove '<NA>' or other string representations of nulls that might remain
    series_clean = series_clean[series_clean != '<NA>']
    series_clean = series_clean[series_clean != 'nan']
    series_clean = series_clean[series_clean != 'NaN']
    series_clean = series_clean[series_clean != 'None']

    # Get value counts
    counts = series_clean.value_counts()

    # Handle high cardinality: truncate and add "Others" category
    if len(counts) > max_categories:
        # Keep top N categories and group the rest as "Others"
        top_counts = counts.head(max_categories - 1)  # Leave room for "Others"
        others_count = counts.iloc[max_categories - 1:].sum()

        if others_count > 0:
            top_counts.loc["Others"] = others_count

        logger.info(f"Truncated categories for '{column}' from {len(counts)} to {len(top_counts)} with 'Others' bucket")
        counts = top_counts

    # Create categories and values, ensuring they're valid Python types
    categories = []
    values = []

    for idx, v in counts.items():
        # Ensure we have valid values
        if idx is not None and pd.notna(idx):
            cat_str = str(idx)
            # Check that cat_str is not a method reference (like the <built-in method title> issue)
            if not callable(idx) and not hasattr(idx, '__func__'):
                categories.append(cat_str)
                if pd.notna(v) and np.isfinite(v):
                    values.append(int(v))
                else:
                    values.append(0)  # Default to 0 if count is invalid

    if not categories:
        logger.warning(f"No valid categories found for column '{column}' after filtering")
        return None

    table_data = [
        {"category": cat, "count": val}
        for cat, val in zip(categories, values)
        if cat and val >= 0  # Filter out any invalid entries
    ]

    if not table_data:
        logger.warning(f"No valid data entries after processing column '{column}'")
        return None

    # Validate and normalize the chart payload
    title = f"Count of {column.replace('_', ' ')}"
    if hasattr(title, 'title'):  # If title is a string, call its title method
        title = title.title()
    else:
        title = str(title)

    chart_payload = ChartPayload(
        title=title,
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
    if df is None or column not in df.columns:
        logger.warning(f"DataFrame is None or column '{column}' not found in dataframe")
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
    # First check if df[column] is valid
    if not isinstance(df[column], pd.Series) and not isinstance(df[column], pd.DataFrame):
        col_series = pd.Series(df[column]) if hasattr(df[column], '__iter__') else pd.Series([df[column]])
    else:
        col_series = df[column]

    # Clean the series by converting to numeric and handling NaN/None values
    series = pd.to_numeric(col_series, errors='coerce')
    # Drop NaN values
    series = series.dropna()

    # Filter out infinite values
    series = series[np.isfinite(series)]

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
        bins = min(bins, n_samples // 10) # At most 10 samples per bin

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
                    # Ensure intervals are valid
                    valid_counts = value_counts[value_counts.index.map(lambda x: pd.notna(x.left) and pd.notna(x.right))]
                    if valid_counts.empty:
                        logger.warning(f"Quantile binning resulted in no valid intervals for {column}")
                        # Fall back to regular binning
                        hist, bin_edges = pd.cut(series, bins=bins, retbins=True)
                        value_counts = hist.value_counts(sort=False)  # Sort by bin order, not by count
                        valid_indices = [i for i, interval in enumerate(value_counts.index) if pd.notna(interval.left)]
                        categories = [f"{value_counts.index[i].left:.2f} - {value_counts.index[i].right:.2f}" for i in valid_indices]
                        values = [int(value_counts.iloc[i]) for i in valid_indices]
                    else:
                        categories = [f"{interval.left:.2f} - {interval.right:.2f}" for interval in valid_counts.index]
                        values = [int(count) for count in valid_counts.values]
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

    # Final validation - ensure all categories and values are valid
    categories = [str(cat) for cat in categories if cat is not None and not pd.isna(cat)]
    values = [int(val) for val in values if val is not None and pd.notna(val) and np.isfinite(val)]

    if not categories or len(categories) != len(values):
        if not series.empty:
            # If we have data but couldn't create proper bins, just use the single value
            single_value = series.iloc[0] if len(series) > 0 else 0
            categories = [str(single_value)]
            values = [len(series)]
        else:
            logger.warning(f"No valid categories found for histogram of column '{column}'")
            return None

    table_data = [
        {"bin_range": cat, "count": val}
        for cat, val in zip(categories, values)
        if cat and isinstance(val, (int, float)) and np.isfinite(val)  # Filter out any invalid entries
    ]

    if not table_data:
        logger.warning(f"No valid data entries after processing histogram for column '{column}'")
        return None

    # Validate and normalize the chart payload
    title = f"Distribution of {column.replace('_', ' ')}"
    if hasattr(title, 'title'):  # If title is a string, call its title method
        title = title.title()
    else:
        title = str(title)

    chart_payload = ChartPayload(
        title=title,
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
    agg_func_lower = agg_func.lower()
    valid_agg_funcs = {'sum', 'mean', 'count', 'min', 'max', 'std', 'median'}
    if agg_func_lower not in valid_agg_funcs:
        logger.warning(f"Invalid aggregation function '{agg_func}', defaulting to 'mean'.")
        agg_func_lower = 'mean'

    try:
        # Ensure y_final is a Series for groupby operations
        if not isinstance(y_final, pd.Series):
            y_final = pd.Series(y_final)
        if not isinstance(x_final, pd.Series):
            x_final = pd.Series(x_final)

        if agg_func_lower == "sum":
            result = y_final.groupby(x_final).sum()
        elif agg_func_lower == "mean":
            result = y_final.groupby(x_final).mean()
        elif agg_func_lower == "count":
            result = y_final.groupby(x_final).count()
        elif agg_func_lower == "min":
            result = y_final.groupby(x_final).min()
        elif agg_func_lower == "max":
            result = y_final.groupby(x_final).max()
        elif agg_func_lower == "std":
            result = y_final.groupby(x_final).std()
        elif agg_func_lower == "median":
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
        agg_display = agg_func_lower.title()
        chart_payload = ChartPayload(
            title=f"{agg_display} of {y_column.replace('_', ' ').title()} by {x_column.replace('_', ' ').title()}",
            column=f"agg_{agg_func_lower}_{x_column}",
            data=table_data,
            type="category_summary"
        )

        return {
            "title": chart_payload.title,
            "x_column": x_column,
            "y_column": y_column,
            "data": chart_payload.data,
            "type": chart_payload.type,
            "agg_func": agg_func_lower
        }
    except Exception as e:
        logger.error(f"Error in aggregation for {x_column} vs {y_column}: {e}")
        return None


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
        agg_func_lower = agg_func.lower()
        valid_agg_funcs = {'sum', 'mean', 'count', 'min', 'max', 'std', 'median'}
        if agg_func_lower not in valid_agg_funcs:
            logger.warning(f"Invalid aggregation function '{agg_func}' for time series, defaulting to 'mean'.")
            agg_func_lower = 'mean'

        try:
            grouped = y_final.groupby(x_final).agg(agg_func_lower)
            dates = [dt.isoformat() for dt in grouped.index]
            values = [float(val) for val in grouped.values]
        except Exception as e:
            logger.error(f"Error in aggregation for time series {x_column} vs {y_column}: {e}")
            return None
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
    x_values = [float(val) for val in x_final if pd.notna(val) and np.isfinite(val)]
    y_values = [float(val) for val in y_final if pd.notna(val) and np.isfinite(val)]

    if not x_values or not y_values or len(x_values) != len(y_values) or len(x_values) < 3:
        logger.warning(f"Insufficient or mismatched valid data points for scatter plot between '{x_column}' and '{y_column}' after cleaning for NaN/Inf")
        return None

    table_data = [
        {"x": x_val, "y": y_val}
        for x_val, y_val in zip(x_values, y_values)
    ]

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
        "column": chart_payload.column,
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
        # Filter out non-numeric or infinite values for the box plot
        numeric_values = [float(v) for v in values if pd.notna(v) and np.isfinite(v)]
        if numeric_values:  # Only add if there are valid numeric values
            table_data.append({
                "category": str(category),
                "values": numeric_values
            })

    if not table_data:
        logger.warning(f"No valid data for box plot between '{x_column}' and '{y_column}' after filtering for numeric values")
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
