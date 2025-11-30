# src/viz/plotly_renderer.py

import pandas as pd


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

    series = df[column]
    if not pd.api.types.is_numeric_dtype(series):
        return None

    # Drop NaN values
    series = series.dropna()
    if series.empty:
        return None

    # Create histogram
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
