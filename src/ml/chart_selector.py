# src/ml/chart_selector.py

import re
import pandas as pd


def _is_id_like_name(name: str) -> bool:
    """
    Generic check if a column name looks like an identifier.
    Works for many datasets, not hard-coded to one.
    """
    name_lower = name.lower()
    # Common ID-ish tokens
    patterns = [
        r"\bid\b",
        r"\b_id\b",
        r"id$",
        r"\bkey\b",
        r"\bcode\b",
        r"\bpk\b",
        r"\buid\b",
        r"\buuid\b",
        r"\bno\b",
        r"\bnumber\b",
    ]
    return any(re.search(p, name_lower) for p in patterns)


def _is_geo_like_name(name: str) -> bool:
    """
    Check if the name looks like a geographic coordinate: lat/long etc.
    """
    name_lower = name.lower()
    patterns = [
        r"\blat\b",
        r"\blong\b",
        r"\blng\b",
        r"latitude",
        r"longitude",
    ]
    return any(re.search(p, name_lower) for p in patterns)


def _is_metric_like_name(name: str) -> bool:
    """
    Check if the name looks like a useful metric / measure.
    These are generic patterns that appear in many datasets.
    """
    name_lower = name.lower()
    tokens = [
        "price", "amount", "total", "sum", "revenue", "income", "sales", "profit",
        "cost", "fee", "charge", "value",
        "count", "qty", "quantity", "volume",
        "review", "rating", "score",
        "nights", "days", "duration", "time", "age",
    ]
    return any(t in name_lower for t in tokens)


def _is_time_like_name(name: str) -> bool:
    """
    Check if the name looks like a time-related field.
    """
    name_lower = name.lower()
    tokens = [
        "time", "date", "hour", "minute", "second", "day",
        "week", "month", "year", "season", "period", "interval",
        "morning", "evening", "night", "afternoon", "duration",
        "time", "timestamp", "datetime"
    ]
    return any(t in name_lower for t in tokens)


def _is_percentage_like_name(name: str) -> bool:
    """
    Check if the name looks like a percentage-related field.
    """
    name_lower = name.lower()
    tokens = [
        "percent", "percentage", "pct", "ratio", "proportion",
        "rate", "fraction", "part", "discount", "tax", "interest"
    ]
    return any(t in name_lower for t in tokens)


def _is_ranking_like_name(name: str) -> bool:
    """
    Check if the name looks like a ranking-related field.
    """
    name_lower = name.lower()
    tokens = [
        "rank", "level", "rating", "score", "grade", "point",
        "quality", "satisfaction", "review", "feedback"
    ]
    return any(t in name_lower for t in tokens)


def _choose_main_numeric(dataset_profile):
    """
    Choose a 'good' numeric column for charts on ANY dataset.

    Heuristics:
    - Penalise almost-unique columns (likely identifiers).
    - Penalise ID-like names.
    - Penalise geo-like names (lat/long).
    - Prefer metric-like names (price, count, reviews, etc).
    - Prefer a moderate unique_ratio (0.05 - 0.8), but allow others if name is strong.
    """
    columns = dataset_profile["columns"]
    n_rows = dataset_profile["n_rows"]

    candidates = []
    for col in columns:
        if col["role"] != "numeric":
            continue

        name = col["name"]
        unique_count = col["unique_count"]
        unique_ratio = unique_count / n_rows if n_rows > 0 else 0.0

        score = 0.0

        # 1) Penalise almost-unique columns
        if unique_ratio > 0.95:
            score -= 60

        # 2) Penalise ID-like names
        if _is_id_like_name(name):
            score -= 80

        # 3) Penalise geo-like names (lat/long)
        if _is_geo_like_name(name):
            score -= 40

        # 4) Reward metric-like names
        if _is_metric_like_name(name):
            score += 40

        # 5) Reward percentage/rate names
        if _is_percentage_like_name(name):
            score += 20

        # 6) Reward ranking/score names
        if _is_ranking_like_name(name):
            score += 20

        # 7) Generic reward based on unique_ratio
        if 0.05 <= unique_ratio <= 0.8:
            score += 15
        elif 0.8 < unique_ratio <= 0.95:
            score += 5
        elif unique_ratio < 0.05:
            # low variety but might still be a meaningful metric (e.g. ratings 1–5)
            score += 5

        candidates.append((score, name))

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda x: x[0])
    best_score, best_name = candidates[0]

    if best_score < -50:
        return None

    return best_name


def _choose_numeric_candidates(dataset_profile, max_charts: int = 3):
    """
    Choose multiple numeric columns for various chart types.
    """
    columns = dataset_profile["columns"]
    n_rows = dataset_profile["n_rows"]

    candidates = []
    for col in columns:
        if col["role"] != "numeric":
            continue

        name = col["name"]
        unique_count = col["unique_count"]
        unique_ratio = unique_count / n_rows if n_rows > 0 else 0.0

        score = 0.0

        # Penalise almost-unique columns (likely identifiers)
        if unique_ratio > 0.95:
            continue  # Skip high-cardinality numeric columns

        # Penalise ID-like names
        if _is_id_like_name(name):
            continue  # Skip ID-like columns

        # Reward metric-like names
        if _is_metric_like_name(name):
            score += 40

        # Reward percentage/rate names
        if _is_percentage_like_name(name):
            score += 20

        # Reward ranking/score names
        if _is_ranking_like_name(name):
            score += 20

        # Generic reward based on unique_ratio
        if 0.05 <= unique_ratio <= 0.8:
            score += 15
        elif 0.8 < unique_ratio <= 0.95:
            score += 5
        elif unique_ratio < 0.05:
            # low variety but might still be a meaningful metric (e.g. ratings 1–5)
            score += 5

        candidates.append((score, name))

    if not candidates:
        return []

    candidates.sort(reverse=True, key=lambda x: x[0])

    # Return up to max_charts column names
    return [name for _, name in candidates[:max_charts]]


def _choose_categorical_candidates(dataset_profile, max_charts: int = 3):
    """
    Choose up to `max_charts` categorical columns that are good for
    'category vs count' charts.

    Generic rules:
    - role == 'categorical'
    - at least 2 unique values
    - prefer up to ~50 categories
    """
    columns = dataset_profile["columns"]
    candidates = []

    for col in columns:
        if col["role"] != "categorical":
            continue

        name = col["name"]
        unique_count = col["unique_count"]

        if unique_count < 2:
            continue

        score = 0.0

        # Prefer fewer categories (cleaner bar chart)
        if unique_count <= 10:
            score += 25
        elif unique_count <= 50:
            score += 20
        elif unique_count <= 200:
            score += 5
        else:
            score -= 10  # too many categories, noisy

        candidates.append((score, name, unique_count))

    if not candidates:
        return []

    # Higher score first, then fewer categories
    candidates.sort(key=lambda x: (-x[0], x[2]))

    # Return up to max_charts column names
    return [name for _, name, _ in candidates[:max_charts]]


def _choose_main_categorical(dataset_profile):
    """
    Choose a single 'main' categorical for numeric-by-category charts.
    Just reuse the first from the candidate list.
    """
    cats = _choose_categorical_candidates(dataset_profile, max_charts=1)
    return cats[0] if cats else None


def _choose_datetime_candidates(dataset_profile, max_charts: int = 2):
    """
    Choose datetime columns for time series analysis.
    """
    datetime_cols = []
    for col in dataset_profile["columns"]:
        if col["role"] == "datetime":
            datetime_cols.append(col["name"])

    # Return up to max_charts datetime columns
    return datetime_cols[:max_charts]


def suggest_charts(df, dataset_profile, kpis, max_charts_per_type: int = 5):
    """
    Rule-based chart suggestions that work for ANY dataset with diverse chart types.

    Args:
        df: Input DataFrame
        dataset_profile: Dataset profile from analyser
        kpis: KPIs from kpi_generator
        max_charts_per_type: Maximum number of each chart type to suggest

    ChartSpec-like dicts:

    {
        "id": str,
        "title": str,
        "chart_type": str,   # Chart.js type: 'bar', 'line', 'scatter', 'pie', etc.
        "intent": str,       # 'histogram', 'category_summary', 'time_series', 'category_count', 'scatter', 'pie'
        "x_field": str,
        "y_field": str | None,
        "agg_func": str | None
    }
    """
    charts = []

    # Choose main fields based on generic heuristics
    main_numeric = _choose_main_numeric(dataset_profile)
    main_categorical = _choose_main_categorical(dataset_profile)
    datetime_cols = _choose_datetime_candidates(dataset_profile, max_charts=max_charts_per_type)
    numeric_cols = _choose_numeric_candidates(dataset_profile, max_charts=max_charts_per_type)
    categorical_cols = _choose_categorical_candidates(dataset_profile, max_charts=max_charts_per_type)

    def next_id():
        return f"chart_{len(charts) + 1}"

    # 1) Time series charts: numeric values over time
    time_series_count = 0
    for dt_col in datetime_cols:
        for num_col in numeric_cols[:2]:  # Limit to first 2 numeric columns per datetime
            if time_series_count >= max_charts_per_type:
                break
            charts.append({
                "id": next_id(),
                "title": f"{num_col} over time ({dt_col})",
                "chart_type": "line",
                "intent": "time_series",
                "x_field": dt_col,
                "y_field": num_col,
                "agg_func": "mean",
            })
            time_series_count += 1

    # 2) Scatter plots: relationship between two numeric variables
    scatter_count = 0
    if len(numeric_cols) >= 2:
        for i in range(len(numeric_cols)):
            for j in range(i+1, min(i+2, len(numeric_cols))):  # Limit to avoid too many scatter plots
                if scatter_count >= max_charts_per_type:
                    break
                charts.append({
                    "id": next_id(),
                    "title": f"Relationship: {numeric_cols[i]} vs {numeric_cols[j]}",
                    "chart_type": "scatter",
                    "intent": "scatter",
                    "x_field": numeric_cols[j],
                    "y_field": numeric_cols[i],
                    "agg_func": None,
                })
                scatter_count += 1
            if scatter_count >= max_charts_per_type:
                break

    # 3) Distribution histograms for numeric columns
    histogram_count = 0
    for num_col in numeric_cols[:max_charts_per_type]:
        if histogram_count >= max_charts_per_type:
            break
        charts.append({
            "id": next_id(),
            "title": f"Distribution of {num_col}",
            "chart_type": "histogram",
            "intent": "histogram",
            "x_field": num_col,
            "y_field": None,
            "agg_func": None,
        })
        histogram_count += 1

    # 4) Categorical summary: numeric by categorical (for different aggregation functions)
    if main_numeric and main_categorical:
        agg_count = 0
        for agg_func in ["mean", "sum", "count", "max", "min", "std", "var"]:
            if agg_count >= max_charts_per_type:
                break
            charts.append({
                "id": next_id(),
                "title": f"{main_numeric} by {main_categorical} ({agg_func})",
                "chart_type": "bar",
                "intent": "category_summary",
                "x_field": main_categorical,
                "y_field": main_numeric,
                "agg_func": agg_func,
            })
            agg_count += 1

    # 5) Pie charts for categorical distributions (only for categories with few unique values)
    pie_count = 0
    for cat_col in categorical_cols[:max_charts_per_type]:
        col_info = next((col for col in dataset_profile["columns"] if col["name"] == cat_col), None)
        if col_info and col_info["unique_count"] <= 10 and pie_count < max_charts_per_type:  # Only for low cardinality
            charts.append({
                "id": next_id(),
                "title": f"Distribution of {cat_col}",
                "chart_type": "pie",
                "intent": "category_pie",
                "x_field": cat_col,
                "y_field": None,
                "agg_func": "count",
            })
            pie_count += 1

    # 6) Pure categorical count charts
    cat_count = 0
    for cat_name in categorical_cols[:max_charts_per_type]:
        if cat_count >= max_charts_per_type:
            break
        charts.append({
            "id": next_id(),
            "title": f"Count of {cat_name}",
            "chart_type": "bar",
            "intent": "category_count",
            "x_field": cat_name,
            "y_field": None,
            "agg_func": "count",
        })
        cat_count += 1

    # 7) Box plots for numeric distributions by category (if we have both)
    if main_numeric and main_categorical:
        charts.append({
            "id": next_id(),
            "title": f"Distribution of {main_numeric} by {main_categorical}",
            "chart_type": "box",
            "intent": "box_plot",
            "x_field": main_categorical,
            "y_field": main_numeric,
            "agg_func": None,
        })

    # 8) Correlation heatmaps for all numeric columns
    if len(numeric_cols) >= 2:
        charts.append({
            "id": next_id(),
            "title": "Correlation between numeric variables",
            "chart_type": "heatmap",
            "intent": "correlation",
            "x_field": "variables",
            "y_field": "variables",
            "agg_func": "correlation",
        })

    # 9) Stacked bar charts for categorical relationships
    stacked_bar_count = 0
    for i in range(len(categorical_cols)):
        for j in range(i+1, min(i+2, len(categorical_cols))):
            if stacked_bar_count >= max_charts_per_type:
                break
            charts.append({
                "id": next_id(),
                "title": f"{categorical_cols[i]} vs {categorical_cols[j]}",
                "chart_type": "bar",  # Can be rendered as stacked bar
                "intent": "categorical_relationship",
                "x_field": categorical_cols[i],
                "y_field": categorical_cols[j],
                "agg_func": "count",
            })
            stacked_bar_count += 1
        if stacked_bar_count >= max_charts_per_type:
            break

    # 10) Violin plots for distribution comparison (when we have categorical and numeric)
    if main_numeric and main_categorical:
        charts.append({
            "id": next_id(),
            "title": f"Distribution of {main_numeric} by {main_categorical} (Violin)",
            "chart_type": "violin",
            "intent": "violin_plot",
            "x_field": main_categorical,
            "y_field": main_numeric,
            "agg_func": None,
        })

    # 11) Additional line charts for more numeric trends over time
    # Create additional individual line charts if we have datetime and multiple numerics
    line_count = 0
    for num_col in numeric_cols[1:max_charts_per_type]:  # Create additional line charts for other numeric columns
        if line_count >= max_charts_per_type:
            break
        for dt_col in datetime_cols[:1]:  # Use only the first datetime column to avoid too many charts
            charts.append({
                "id": next_id(),
                "title": f"{num_col} over time ({dt_col})",
                "chart_type": "line",
                "intent": "time_series",
                "x_field": dt_col,
                "y_field": num_col,
                "agg_func": "mean",
            })
            line_count += 1
            if line_count >= max_charts_per_type:
                break

    # 12) Additional scatter plots with more combinations
    if len(numeric_cols) >= 3:
        scatter_extra_count = 0
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                if scatter_extra_count >= max_charts_per_type // 2:  # Limit additional scatter plots
                    break
                charts.append({
                    "id": next_id(),
                    "title": f"Scatter: {numeric_cols[i]} vs {numeric_cols[j]}",
                    "chart_type": "scatter",
                    "intent": "scatter",
                    "x_field": numeric_cols[j],
                    "y_field": numeric_cols[i],
                    "agg_func": None,
                })
                scatter_extra_count += 1
            if scatter_extra_count >= max_charts_per_type // 2:
                break

    return charts
