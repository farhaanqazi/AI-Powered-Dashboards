import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional, Tuple
from plotly.subplots import make_subplots
import re
import math
# Import common utility functions and dataclasses from src.viz.utils
from src.viz.utils import (
    ChartPayload,
    _is_likely_identifier,
    _build_category_count_data,
    _build_histogram_data,
    _build_category_summary_data,
    _build_time_series_data,
    _build_scatter_data,
    _build_pie_data,
    _build_box_plot_data
)

logger = logging.getLogger(__name__)





















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
    if df is None or df.empty:
        logger.warning("DataFrame is None or empty, returning empty charts")
        return {}

    # Ensure df is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

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
        if not isinstance(spec, dict):
            logger.warning(f"Invalid chart spec encountered: {spec}")
            continue

        intent = spec.get('intent')
        chart_id = str(spec.get('id', f'chart_{len(charts)}'))

        # Skip if we've reached the maximum charts
        if len(charts) >= max_charts:
            break

        chart_data = None

        try:
            if intent == 'category_count':
                col = spec.get('x_field')
                if col and col in df.columns:
                    chart_data = _build_category_count_data(
                        df,
                        column=col,
                        max_categories=max_categories,
                        dataset_profile=dataset_profile
                    )

            elif intent == 'histogram':
                col = spec.get('x_field')
                if col and col in df.columns:
                    chart_data = _build_histogram_data(
                        df,
                        column=col,
                        dataset_profile=dataset_profile
                    )

            elif intent == 'category_summary':
                x_col = spec.get('x_field')
                y_col = spec.get('y_field')
                agg_func = spec.get('agg_func', 'mean')
                if x_col and x_col in df.columns and y_col and y_col in df.columns:
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
                if x_col and x_col in df.columns and y_col and y_col in df.columns:
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
                if x_col and x_col in df.columns and y_col and y_col in df.columns:
                    chart_data = _build_scatter_data(
                        df,
                        x_column=x_col,
                        y_column=y_col,
                        dataset_profile=dataset_profile
                    )

            elif intent == 'category_pie':
                col = spec.get('x_field')
                if col and col in df.columns:
                    chart_data = _build_pie_data(
                        df,
                        column=col,
                        max_categories=max_categories,
                        dataset_profile=dataset_profile
                    )

            elif intent == 'box_plot':
                x_col = spec.get('x_field')
                y_col = spec.get('y_field')
                if x_col and x_col in df.columns and y_col and y_col in df.columns:
                    chart_data = _build_box_plot_data(
                        df,
                        x_column=x_col,
                        y_column=y_col,
                        dataset_profile=dataset_profile
                    )

            elif intent == 'correlation_matrix':
                # This logic seems to be missing from the file, assuming it exists elsewhere
                # chart_data = _build_correlation_heatmap_data(...)
                pass

            # Handle 'distribution' as an alias for 'histogram'
            elif intent == 'distribution':
                col = spec.get('x_field')
                if col and col in df.columns:
                    chart_data = _build_histogram_data(
                        df,
                        column=col,
                        dataset_profile=dataset_profile
                    )
            
            # Handle 'group_comparison' as an alias for 'category_summary'
            elif intent == 'group_comparison':
                x_col = spec.get('x_field')
                y_col = spec.get('y_field')
                agg_func = spec.get('agg_func', 'mean')
                if x_col and x_col in df.columns and y_col and y_col in df.columns:
                    chart_data = _build_category_summary_data(
                        df,
                        x_column=x_col,
                        y_column=y_col,
                        agg_func=agg_func,
                        dataset_profile=dataset_profile
                    )

            else:
                logger.warning(f"Unknown chart intent: {intent}")
                continue

            # Add chart data if valid and not already added
            if chart_data and chart_id not in charts:
                # Additional validation for chart data
                if chart_data.get('data'):
                    charts[chart_id] = chart_data
                else:
                    logger.debug(f"Chart {chart_id} has no data, skipping")
            else:
                logger.debug(f"Chart {chart_id} is invalid or already exists, skipping")

        except Exception as e:
            logger.error(f"Error building chart with intent '{intent}' and ID '{chart_id}': {e}")
            import traceback
            traceback.print_exc()
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
    if df is None or df.empty:
        logger.warning("DataFrame is None or empty, returning empty category charts")
        return {}

    # Ensure df is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    category_charts = {}

    if not chart_specs:
        logger.warning("No chart specs provided for category charts")
        return category_charts

    for spec in chart_specs:
        if not isinstance(spec, dict):
            logger.warning(f"Invalid chart spec encountered: {spec}")
            continue

        if spec.get("intent") != "category_count":
            continue

        col = spec.get("x_field")
        if not col or col not in df.columns or col in category_charts:
            continue

        # Check if column is likely an identifier before building chart
        series = df[col]
        if _is_likely_identifier(series, col):
            logger.info(f"Skipping identifier column '{col}' from category count chart")
            continue

        # Check dataset profile for roles
        if dataset_profile:
            col_profile = next((c for c in dataset_profile.get('columns', []) if c.get('name') == col), None)
            if col_profile and col_profile.get('role') == 'identifier':
                logger.info(f"Skipping identifier column '{col}' from category count chart")
                continue

        chart_obj = _build_category_count_data(
            df,
            column=col,
            max_categories=max_categories,
            dataset_profile=dataset_profile
        )

        if chart_obj is not None and chart_obj.get('data'):
            category_charts[col] = chart_obj

        if len(category_charts) >= max_charts:
            break

    logger.info(f"Built {len(category_charts)} category count charts")
    return category_charts