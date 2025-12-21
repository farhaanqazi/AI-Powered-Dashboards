import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any
# Import common utility functions from src.viz.utils
from src.viz.utils import (
    _build_category_count_data,
    _build_histogram_data,
    _build_time_series_data,
    _build_scatter_data,
    _build_pie_data,
)

logger = logging.getLogger(__name__)

def generate_all_chart_data(df: pd.DataFrame, dataset_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generates all possible charts based on the dataset using the simple renderer logic.
    """
    charts = []

    if df is None or df.empty or dataset_profile is None or dataset_profile.get('n_rows', 0) == 0:
        logger.warning("DataFrame is None, empty, or dataset_profile is invalid, returning empty charts")
        return charts

    # Ensure df is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # Validate dataset_profile structure
    if not isinstance(dataset_profile, dict) or 'columns' not in dataset_profile:
        logger.error("Invalid dataset_profile: missing 'columns' key")
        return charts

    # Get column information
    numeric_cols = [col['name'] for col in dataset_profile['columns'] if col.get('role') == 'numeric' and col.get('name') in df.columns]
    categorical_cols = [col['name'] for col in dataset_profile['columns'] if col.get('role') == 'categorical' and col.get('name') in df.columns]
    datetime_cols = [col['name'] for col in dataset_profile['columns'] if col.get('role') == 'datetime' and col.get('name') in df.columns]

    # Create category count charts for categorical columns (using simple renderer's logic)
    for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
        if col in df.columns:  # Double-check the column exists
            chart_obj = _build_category_count_data(df, col, dataset_profile=dataset_profile)
            if chart_obj and chart_obj.get('data'):
                charts.append(chart_obj)

    # Create histograms for numeric columns (using simple renderer's logic)
    for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
        if col in df.columns:  # Double-check the column exists
            chart_obj = _build_histogram_data(df, col, dataset_profile=dataset_profile)
            if chart_obj and chart_obj.get('data'):
                charts.append(chart_obj)

    # Create time series charts for datetime vs numeric (using simple renderer's logic)
    for dt_col in datetime_cols:
        if dt_col in df.columns:  # Double-check the column exists
            for num_col in numeric_cols[:2]:  # Limit to first 2 numeric cols per datetime
                if num_col in df.columns:  # Double-check the column exists
                    chart_obj = _build_time_series_data(df, dt_col, num_col, dataset_profile=dataset_profile)
                    if chart_obj and chart_obj.get('data'):
                        charts.append(chart_obj)

    # Create scatter plots for numeric vs numeric (using simple renderer's logic)
    if len(numeric_cols) >= 2:
        # Check that both columns exist before creating scatter plot
        if numeric_cols[0] in df.columns and numeric_cols[1] in df.columns:
            chart_obj = _build_scatter_data(df, numeric_cols[0], numeric_cols[1], dataset_profile=dataset_profile)
            if chart_obj and chart_obj.get('data'):
                charts.append(chart_obj)

    # Create pie charts for categorical columns (using simple renderer's logic)
    for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
        if col in df.columns:  # Double-check the column exists
            # Find the column index to access unique_count
            col_idx = next((i for i, col_dict in enumerate(dataset_profile['columns']) if col_dict.get('name') == col), -1)
            if col_idx != -1 and dataset_profile['columns'][col_idx].get('unique_count', 0) <= 10:
                chart_obj = _build_pie_data(df, col, dataset_profile=dataset_profile)
                if chart_obj and chart_obj.get('data'):
                    charts.append(chart_obj)

    # Fallback: Create a generic chart if no others are generated
    if not charts and numeric_cols and numeric_cols[0] in df.columns:
        chart_obj = _build_histogram_data(df, numeric_cols[0], dataset_profile=dataset_profile)
        if chart_obj and chart_obj.get('data'):
            charts.append(chart_obj)

    logger.info(f"Generated {len(charts)} charts using simple renderer logic")
    return charts
