# src/ml/chart_selector.py

import re


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

        # 5) Generic reward based on unique_ratio
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


def _choose_main_datetime(dataset_profile):
    """
    Choose a datetime column if available (first one is usually fine).
    """
    for col in dataset_profile["columns"]:
        if col["role"] == "datetime":
            return col["name"]
    return None


def suggest_charts(df, dataset_profile, kpis):
    """
    Rule-based chart suggestions that work for ANY dataset.

    ChartSpec-like dicts:

    {
        "id": str,
        "title": str,
        "chart_type": str,   # Chart.js type: 'bar', 'line', etc.
        "intent": str,       # 'histogram', 'category_summary', 'time_series', 'category_count'
        "x_field": str,
        "y_field": str | None,
        "agg_func": str | None
    }
    """
    charts = []

    # Choose main fields based on generic heuristics
    main_numeric = _choose_main_numeric(dataset_profile)
    main_categorical = _choose_main_categorical(dataset_profile)
    main_datetime = _choose_main_datetime(dataset_profile)

    # Also get multiple categorical candidates for pure count charts
    cat_for_counts = _choose_categorical_candidates(dataset_profile, max_charts=3)

    def next_id():
        return f"chart_{len(charts) + 1}"

    # 1) Histogram-style chart for the main numeric column
    if main_numeric:
        charts.append({
            "id": next_id(),
            "title": f"Distribution of {main_numeric}",
            "chart_type": "bar",
            "intent": "histogram",
            "x_field": main_numeric,
            "y_field": None,
            "agg_func": None,
        })

    # 2) Categorical summary: main numeric by main categorical (sum)
    if main_numeric and main_categorical:
        charts.append({
            "id": next_id(),
            "title": f"{main_numeric} by {main_categorical} (sum)",
            "chart_type": "bar",
            "intent": "category_summary",
            "x_field": main_categorical,
            "y_field": main_numeric,
            "agg_func": "sum",
        })

    # 3) Time series: main numeric over main datetime
    if main_numeric and main_datetime:
        charts.append({
            "id": next_id(),
            "title": f"{main_numeric} over time ({main_datetime})",
            "chart_type": "line",
            "intent": "time_series",
            "x_field": main_datetime,
            "y_field": main_numeric,
            "agg_func": "sum",
        })

    # 4) Pure categorical count charts: category vs count
    for cat_name in cat_for_counts:
        charts.append({
            "id": next_id(),
            "title": f"Count of {cat_name}",
            "chart_type": "bar",
            "intent": "category_count",
            "x_field": cat_name,
            "y_field": None,       # we'll derive counts on x in the frontend/backend later
            "agg_func": "count",
        })

    return charts
