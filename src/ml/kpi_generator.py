# src/ml/kpi_generator.py

def generate_basic_kpis(df, dataset_profile):
    """
    KPIs = columns that look important based on their data behaviour,
    not on hard-coded name patterns.

    We highlight:
    - Top numeric columns by variability  -> "metric"
    - Top categorical/text columns by richness -> "category"
    - Datetime columns                    -> "time feature"

    We only show the column name + a simple tag.
    """

    kpis = []
    columns = dataset_profile["columns"]
    n_rows = dataset_profile["n_rows"] if dataset_profile.get("n_rows") else len(df)

    # ----------------------------
    # 1) Numeric "metric" columns
    # ----------------------------
    numeric_candidates = []
    for col in columns:
        if col["role"] != "numeric":
            continue

        stats = col.get("stats") or {}
        std_val = stats.get("std")

        # require some spread; ignore totally constant or missing
        if std_val is None or std_val == 0:
            continue

        numeric_candidates.append((abs(std_val), col["name"]))

    # Sort by variability (std) descending and take top 3
    numeric_candidates.sort(reverse=True, key=lambda x: x[0])
    for _, name in numeric_candidates[:3]:
        kpis.append({
            "label": name,
            "value": "metric",
        })

    # -----------------------------------------
    # 2) Categorical/text "category" columns
    # -----------------------------------------
    cat_candidates = []
    for col in columns:
        if col["role"] not in ("categorical", "text"):
            continue

        uniq = col.get("unique_count", 0)
        if n_rows <= 0:
            continue
        unique_ratio = uniq / n_rows

        # Skip degenerate ones
        if uniq < 2:
            continue

        # Skip almost-all-unique (look more like IDs)
        if unique_ratio > 0.9:
            continue

        # Score: we like mid/medium richness
        # e.g. 3–100 categories is often interesting
        score = 0.0
        if 0.01 <= unique_ratio <= 0.5:
            score += 10
        elif 0.5 < unique_ratio <= 0.9:
            score += 3
        else:
            score += 1

        cat_candidates.append((score, uniq, col["name"]))

    # Sort: higher score, then more categories
    cat_candidates.sort(key=lambda x: (-x[0], -x[1]))
    for _, _, name in cat_candidates[:3]:
        kpis.append({
            "label": name,
            "value": "category",
        })

    # ----------------------------
    # 3) Datetime "time feature"
    # ----------------------------
    time_cols = [col for col in columns if col["role"] == "datetime"]
    for col in time_cols[:2]:
        kpis.append({
            "label": col["name"],
            "value": "time feature",
        })

    return kpis
