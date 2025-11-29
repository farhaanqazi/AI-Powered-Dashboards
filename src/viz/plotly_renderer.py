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
