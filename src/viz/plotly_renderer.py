# src/viz/plotly_renderer.py

import pandas as pd
import numpy as np


def _build_category_count_data(
    df: pd.DataFrame,
    column: str,
    max_categories: int = 10,
):

    if column not in df.columns:
        return None

    counts = df[column].value_counts(dropna=False).head(max_categories)

    categories = [str(idx) for idx in counts.index]
    values = [int(v) for v in counts.values]

    if not categories:
        return None

    table_data = [
        {"category": cat, "count": val}
        for cat, val in zip(categories, values)
    ]

    return {
        "title": f"Count of {column}",
        "column": column,
        "data": table_data,
    }


def _build_histogram_data(
    df: pd.DataFrame,
    column: str,
    bins: int = 20,
):

    if column not in df.columns:
        return None

    # Ensure the column is numeric
    series = pd.to_numeric(df[column], errors='coerce')
    # Drop NaN values
    series = series.dropna()
    if series.empty:
        return None

    # Create histogram if there are enough unique values
    if series.nunique() < 2:
        # If only one unique value, just return that as a single bin
        categories = [str(series.iloc[0])]
        values = [len(series)]
    else:
        # Create histogram bins
        hist, bin_edges = pd.cut(series, bins=bins, retbins=True)
        value_counts = hist.value_counts().sort_index()

        # Prepare chart data
        categories = [f"{interval.left:.2f} - {interval.right:.2f}" for interval in value_counts.index]
        values = [int(count) for count in value_counts.values]

    if not categories:
        return None

    table_data = [
        {"category": cat, "count": val}
        for cat, val in zip(categories, values)
    ]

    return {
        "title": f"Distribution of {column}",
        "column": column,
        "data": table_data,
    }


def _build_category_summary_data(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    agg_func: str = "sum",
):

    if x_column not in df.columns or y_column not in df.columns:
        return None

    # Drop rows where either column is NaN
    subset = df[[x_column, y_column]].dropna()
    if subset.empty:
        return None

    # Check if the x_column is suitable for grouping (categorical/low cardinality)
    unique_count = subset[x_column].nunique()
    if unique_count > 50:  # Too many categories for a bar chart
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
        return None

    table_data = [
        {"category": cat, "count": val}
        for cat, val in zip(categories, values)
    ]

    return {
        "title": f"{agg_func.title()} of {y_column} by {x_column}",
        "x_column": x_column,
        "y_column": y_column,
        "data": table_data,
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

    if column not in df.columns:
        return None

    # Get value counts
    counts = df[column].value_counts(dropna=True)

    categories = [str(idx) for idx in counts.index]
    values = [int(val) for val in counts.values]

    if not categories:
        return None

    # Ensure we have positive values
    table_data = [
        {"category": cat, "value": val}
        for cat, val in zip(categories, values) if val > 0
    ]

    if not table_data:
        return None

    return {
        "title": f"Distribution of {column}",
        "column": column,
        "data": table_data,
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


def _build_correlation_data(
    df: pd.DataFrame,
    numeric_columns: list,
):

    if len(numeric_columns) < 2:
        return None

    # Filter dataframe to only include the specified numeric columns
    subset_df = df[numeric_columns]

    # Convert all columns to numeric, handling errors
    for col in subset_df.columns:
        subset_df[col] = pd.to_numeric(subset_df[col], errors='coerce')

    # Drop rows with NaN values for correlation calculation
    subset_df = subset_df.dropna()

    if subset_df.empty or len(subset_df.columns) < 2:
        return None

    # Calculate correlation matrix
    corr_matrix = subset_df.corr()

    # Prepare chart data
    categories = [str(col) for col in corr_matrix.columns]
    values_matrix = corr_matrix.values.tolist()

    if not categories:
        return None

    table_data = {
        "categories": categories,
        "values": values_matrix
    }

    return {
        "title": "Correlation Matrix",
        "data": table_data,
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