"""Phase 15 S15.1 (Tier A) — supervised driver analysis + prediction quality.

Given the cleaned frame and its Layer-2 enriched profiles, this module:

* auto-picks a modelling **target** from the semantic roles — a non-identifier
  numeric measure becomes a *regression* target, a low-cardinality categorical
  becomes a *classification* target (an explicit ``config.ML_TARGET`` overrides);
* drops identifiers and target-leaky columns, one-hot encodes low-cardinality
  categoricals, and assembles a bounded, deterministic feature matrix;
* cross-validates a ``HistGradientBoosting`` model against a linear/logistic
  baseline, reporting an **honest** CV metric (F1 / accuracy, or R²·MAE);
* computes **permutation importance** ("what drives X") with the effect
  *direction* of the leading numeric driver;
* emits the ranked importances as a horizontal-bar ``ChartPayload`` and a
  plain-English verdict.

Defensive by construction: every stage is guarded and the public
:func:`compute_ml_insights` never raises — it returns ``{"available": False,
"reason": ...}`` when it cannot model the data, so the deployed image stays
correct (and slim) even without the heavy ML stack.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src import config
from src.analysis.data_structures import EnrichedProfile

# Year columns are numeric by storage but an axis, not an outcome — recognise
# them by name so we never regress *on* a year. (Kept local so this module
# stays as import-light as ``statistical_depth`` — no heavy contract package.)
_YEAR_NAME_RE = re.compile(r"\b(year|yr|fy|fiscal_year)\b", re.IGNORECASE)

try:
    from src.logger import get_logger
    logger = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


# --- column selection ------------------------------------------------------

def _sample(df: pd.DataFrame) -> pd.DataFrame:
    cap = config.ML_MAX_ROWS
    if len(df) > cap:
        return df.sample(cap, random_state=config.ML_RANDOM_STATE)
    return df


def _is_identifier(col: str, profiles: Dict[str, EnrichedProfile]) -> bool:
    p = profiles.get(col)
    return p is not None and p.role == "identifier"


def _is_year(col: str, profiles: Dict[str, EnrichedProfile]) -> bool:
    p = profiles.get(col)
    return p is not None and p.role in ("numeric", "year") and \
        bool(_YEAR_NAME_RE.search(col))


def _numeric_feature_cols(
    df: pd.DataFrame, profiles: Dict[str, EnrichedProfile]
) -> List[str]:
    """Numeric/ratio/year measures usable as model inputs (ids excluded)."""
    out = []
    for c in df.columns:
        p = profiles.get(c)
        if p is None or _is_identifier(c, profiles):
            continue
        if p.role in ("numeric", "ratio", "year", "boolean") and \
                pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def _categorical_feature_cols(
    df: pd.DataFrame, profiles: Dict[str, EnrichedProfile]
) -> List[str]:
    """Low-cardinality categoricals usable as one-hot model inputs."""
    out = []
    for c in df.columns:
        p = profiles.get(c)
        if p is None or _is_identifier(c, profiles):
            continue
        if p.role in ("categorical", "boolean"):
            nun = int(df[c].nunique(dropna=True))
            if 2 <= nun <= config.ML_MAX_CAT_CARDINALITY:
                out.append(c)
    return out


# --- target selection ------------------------------------------------------

def _regression_candidates(
    df: pd.DataFrame, profiles: Dict[str, EnrichedProfile]
) -> List[str]:
    out = []
    for c in df.columns:
        p = profiles.get(c)
        if p is None or _is_identifier(c, profiles):
            continue
        if _is_year(c, profiles):  # a year is an axis, not an outcome to predict
            continue
        if p.role in ("numeric", "ratio") and pd.api.types.is_numeric_dtype(df[c]):
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if s.nunique() >= 10:  # continuous enough to regress on
                out.append(c)
    return out


def _classification_candidates(
    df: pd.DataFrame, profiles: Dict[str, EnrichedProfile]
) -> List[str]:
    out = []
    for c in df.columns:
        p = profiles.get(c)
        if p is None or _is_identifier(c, profiles):
            continue
        if p.role in ("categorical", "boolean"):
            nun = int(df[c].nunique(dropna=True))
            if 2 <= nun <= config.ML_MAX_CLASSES:
                out.append(c)
    return out


def _pick_target(
    df: pd.DataFrame, profiles: Dict[str, EnrichedProfile]
) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(target, task)`` deterministically.

    An explicit ``config.ML_TARGET`` wins. Otherwise a regression target (the
    highest-variance numeric measure — usually the KPI you want to explain) is
    preferred; failing that, the lowest-cardinality categorical for
    classification. Ties break on column order.
    """
    explicit = config.ML_TARGET
    if explicit and explicit in df.columns:
        if explicit in _classification_candidates(df, profiles):
            return explicit, "classification"
        if pd.api.types.is_numeric_dtype(df[explicit]):
            return explicit, "regression"
        return explicit, "classification"

    order = list(df.columns)
    clf = _classification_candidates(df, profiles)
    reg = _regression_candidates(df, profiles)

    # 1. A *binary* categorical/boolean is an unambiguous labelled outcome
    #    (churn yes/no, converted, pass/fail) — prefer it for classification.
    binary = [c for c in clf if int(df[c].nunique(dropna=True)) == 2]
    if binary:
        best = min(binary, key=lambda c: (
            0 if (profiles.get(c) and profiles[c].role == "boolean") else 1,
            order.index(c)))
        return best, "classification"

    # 2. A continuous numeric measure (the KPI you usually want to explain) —
    #    pick the highest-variance one for regression.
    if reg:
        best, best_var = None, -1.0
        for c in reg:
            v = float(pd.to_numeric(df[c], errors="coerce").dropna().var() or 0.0)
            if v > best_var:
                best, best_var = c, v
        return best, "regression"

    # 3. Otherwise the lowest-cardinality categorical = cleanest classification.
    if clf:
        best = min(clf, key=lambda c: (int(df[c].nunique(dropna=True)),
                                       order.index(c)))
        return best, "classification"
    return None, None


# --- feature matrix --------------------------------------------------------

def _drop_leaky_numeric(
    data: pd.DataFrame, target: str, num_feats: List[str]
) -> Tuple[List[str], List[str]]:
    """Drop numeric features that are essentially a copy of a numeric target.

    A |Pearson| above ``config.ML_LEAKAGE_CORR`` means the feature is a linear
    restatement of the target (a data leak), not a driver. Returns
    ``(kept, dropped)``.
    """
    if not pd.api.types.is_numeric_dtype(data[target]):
        return num_feats, []
    y = pd.to_numeric(data[target], errors="coerce")
    kept, dropped = [], []
    for c in num_feats:
        x = pd.to_numeric(data[c], errors="coerce")
        pair = pd.concat([x, y], axis=1).dropna()
        if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2:
            kept.append(c)
            continue
        r = pair.iloc[:, 0].corr(pair.iloc[:, 1])
        if r is not None and not math.isnan(r) and abs(r) >= config.ML_LEAKAGE_CORR:
            dropped.append(c)
        else:
            kept.append(c)
    return kept, dropped


def _build_matrix(
    data: pd.DataFrame, num_feats: List[str], cat_feats: List[str]
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """One-hot encode categoricals; return ``(X, encoded_col -> source_col)``.

    Categoricals are dropped highest-cardinality-first if the encoded width
    would exceed ``config.ML_MAX_FEATURES``.
    """
    cat_feats = sorted(cat_feats, key=lambda c: int(data[c].nunique(dropna=True)))
    frames: List[pd.DataFrame] = []
    source: Dict[str, str] = {}
    width = 0
    for c in num_feats:
        frames.append(pd.to_numeric(data[c], errors="coerce").rename(c).to_frame())
        source[c] = c
        width += 1
    for c in cat_feats:
        d = pd.get_dummies(
            data[c].astype("object"), prefix=c, prefix_sep="=", dummy_na=False
        )
        if width + d.shape[1] > config.ML_MAX_FEATURES:
            continue
        for dc in d.columns:
            source[dc] = c
        frames.append(d)
        width += d.shape[1]
    if not frames:
        return pd.DataFrame(index=data.index), source
    X = pd.concat(frames, axis=1)
    return X, source


# --- modelling -------------------------------------------------------------

def _regression_report(
    X: pd.DataFrame, y: pd.Series, k: int
) -> Tuple[Dict[str, Any], Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    rs = config.ML_RANDOM_STATE
    cv = KFold(n_splits=k, shuffle=True, random_state=rs)
    model = HistGradientBoostingRegressor(random_state=rs)
    Xv, yv = X.to_numpy(dtype=float), y.to_numpy(dtype=float)

    r2 = cross_val_score(model, Xv, yv, cv=cv, scoring="r2")
    mae = -cross_val_score(model, Xv, yv, cv=cv, scoring="neg_mean_absolute_error")
    baseline = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), LinearRegression()
    )
    base_r2 = cross_val_score(baseline, Xv, yv, cv=cv, scoring="r2")

    model.fit(Xv, yv)  # fit on full sample for permutation importance
    metrics = {
        "cv_folds": k,
        "r2_mean": round(float(np.mean(r2)), 4),
        "r2_std": round(float(np.std(r2)), 4),
        "mae_mean": round(float(np.mean(mae)), 4),
        "model": "HistGradientBoostingRegressor",
        "baseline_model": "LinearRegression",
        "baseline_r2_mean": round(float(np.mean(base_r2)), 4),
        "scoring": "r2",
    }
    return metrics, model


def _classification_report(
    X: pd.DataFrame, y: pd.Series, k: int
) -> Tuple[Dict[str, Any], Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    rs = config.ML_RANDOM_STATE
    # Stratified CV needs every fold to see every class.
    min_class = int(y.value_counts().min())
    k = max(2, min(k, min_class))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=rs)
    model = HistGradientBoostingClassifier(random_state=rs)
    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy()

    f1 = cross_val_score(model, Xv, yv, cv=cv, scoring="f1_weighted")
    acc = cross_val_score(model, Xv, yv, cv=cv, scoring="accuracy")
    baseline = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    base_f1 = cross_val_score(baseline, Xv, yv, cv=cv, scoring="f1_weighted")

    counts = y.value_counts()
    majority = counts.index[0]
    majority_acc = float(counts.iloc[0] / counts.sum())

    model.fit(Xv, yv)
    metrics = {
        "cv_folds": k,
        "f1_mean": round(float(np.mean(f1)), 4),
        "f1_std": round(float(np.std(f1)), 4),
        "accuracy_mean": round(float(np.mean(acc)), 4),
        "model": "HistGradientBoostingClassifier",
        "baseline_model": "LogisticRegression",
        "baseline_f1_mean": round(float(np.mean(base_f1)), 4),
        "majority_class": str(majority),
        "majority_baseline_accuracy": round(majority_acc, 4),
        "n_classes": int(counts.size),
        "scoring": "f1_weighted",
    }
    return metrics, model


def _importances(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    source: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Permutation importance aggregated back to original source columns."""
    from sklearn.inspection import permutation_importance

    rs = config.ML_RANDOM_STATE
    scoring = "f1_weighted" if task == "classification" else "r2"
    yv = y.to_numpy() if task == "classification" else y.to_numpy(dtype=float)
    pi = permutation_importance(
        model, X.to_numpy(dtype=float), yv,
        n_repeats=config.ML_PERMUTATION_REPEATS,
        random_state=rs, scoring=scoring, n_jobs=1,
    )
    agg: Dict[str, float] = {}
    for col, imp in zip(X.columns, pi.importances_mean):
        src = source.get(col, col)
        agg[src] = agg.get(src, 0.0) + float(imp)
    ranked = sorted(
        ({"feature": f, "importance": round(v, 4)} for f, v in agg.items()),
        key=lambda d: d["importance"], reverse=True,
    )
    return ranked[: config.ML_TOP_FEATURES]


def _direction(
    data: pd.DataFrame, target: str, feature: str, task: str
) -> Optional[str]:
    """Sign of the leading numeric driver's effect on the target."""
    if feature not in data.columns or not pd.api.types.is_numeric_dtype(data[feature]):
        return None
    x = pd.to_numeric(data[feature], errors="coerce")
    if task == "regression":
        y = pd.to_numeric(data[target], errors="coerce")
    else:  # binary-ish: correlate with the majority-vs-rest indicator
        top = data[target].value_counts().index[0]
        y = (data[target] == top).astype(float)
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2:
        return None
    r = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    if r is None or math.isnan(r) or abs(r) < 0.05:
        return "flat"
    return "positive" if r > 0 else "negative"


# --- verdict + chart -------------------------------------------------------

def _verdict(task: str, target: str, metrics: Dict[str, Any]) -> str:
    if task == "regression":
        r2 = metrics["r2_mean"]
        mae = metrics["mae_mean"]
        base = metrics["baseline_r2_mean"]
        if r2 >= 0.5:
            strength = "a strong, dependable signal"
        elif r2 >= 0.2:
            strength = "a moderate signal — useful but not precise"
        else:
            strength = "little predictable structure — treat predictions cautiously"
        beats = "beats" if r2 > base + 0.02 else "roughly matches"
        return (
            f"The model explains about {max(r2, 0) * 100:.0f}% of the variation in "
            f"'{target}' (cross-validated R² = {r2:.2f}), with a typical error of "
            f"±{mae:.3g}. That is {strength}. The gradient-boosted model {beats} a "
            f"simple linear baseline (R² {base:.2f})."
        )
    f1 = metrics["f1_mean"]
    acc = metrics["accuracy_mean"]
    maj = metrics["majority_baseline_accuracy"]
    cls = metrics["majority_class"]
    lift = "a clear, useful lift over guessing" if acc > maj + 0.05 \
        else "barely better than guessing the most common class — a weak signal"
    return (
        f"The model predicts '{target}' with a cross-validated F1 of {f1:.2f} "
        f"(accuracy {acc * 100:.0f}%), versus {maj * 100:.0f}% for always guessing "
        f"the most common class '{cls}'. That is {lift}."
    )


def _importance_chart(target: str, importances: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Horizontal-bar ChartPayload of the ranked drivers (frontend + PDF)."""
    data = [
        {
            "feature": d["feature"], "importance": d["importance"],
            "category": d["feature"], "value": d["importance"],
        }
        for d in importances
    ]
    return {
        "title": f"What drives {target}",
        "column": "feature",
        "type": "bar",
        "intent": "feature_importance",
        "section": "Predictions",
        "x_column": "feature",
        "y_column": "importance",
        "x_title": "Driver",
        "y_title": "Permutation importance",
        "data": data,
    }


# --- entrypoint ------------------------------------------------------------

def compute_ml_insights(
    df: pd.DataFrame,
    enriched_profiles: Dict[str, EnrichedProfile],
    fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    """Tier-A supervised driver analysis. Never raises.

    Returns ``{"available": False, "reason": ...}`` when modelling is disabled,
    the ML stack is missing, or the data cannot support an honest model.

    When ``fingerprint`` is given, the fitted model + the metadata needed to
    align a single feature row is cached (S15.4 what-if path) — no retrain at
    predict time.
    """
    if not config.ML_INSIGHTS_ENABLED:
        return {"available": False, "reason": "disabled"}
    if df is None or df.empty:
        return {"available": False, "reason": "empty"}
    try:
        import sklearn  # noqa: F401
    except Exception:
        return {"available": False, "reason": "sklearn-unavailable"}

    try:
        work = _sample(df)
        target, task = _pick_target(work, enriched_profiles)
        if target is None:
            return {"available": False, "reason": "no-suitable-target"}

        num_feats = [c for c in _numeric_feature_cols(work, enriched_profiles)
                     if c != target]
        cat_feats = [c for c in _categorical_feature_cols(work, enriched_profiles)
                     if c != target]
        num_feats, leaked = _drop_leaky_numeric(work, target, num_feats)
        if not num_feats and not cat_feats:
            return {"available": False, "reason": "no-features", "target": target}

        X, source = _build_matrix(work, num_feats, cat_feats)
        frame = X.copy()
        frame["__target__"] = (
            pd.to_numeric(work[target], errors="coerce")
            if task == "regression" else work[target].astype("object")
        )
        frame = frame.dropna(subset=["__target__"])
        # HistGradientBoosting tolerates NaN inputs; rows need only a valid y.
        y = frame.pop("__target__")
        X = frame

        if len(X) < config.ML_MIN_ROWS:
            return {"available": False, "reason": "too-few-rows",
                    "target": target, "n_rows": int(len(X))}
        if task == "classification":
            vc = y.value_counts()
            if vc.size < 2 or int(vc.min()) < 2:
                return {"available": False, "reason": "degenerate-target",
                        "target": target}
        elif y.nunique() < 5:
            return {"available": False, "reason": "degenerate-target",
                    "target": target}

        k = max(2, min(config.ML_CV_FOLDS, len(X) // 2))
        if task == "regression":
            metrics, model = _regression_report(X, y, k)
        else:
            metrics, model = _classification_report(X, y, k)

        importances = _importances(model, X, y, task, source)
        top_feature = importances[0]["feature"] if importances else None
        if top_feature is not None:
            for entry in importances:
                d = _direction(work, target, entry["feature"], task)
                if d is not None:
                    entry["direction"] = d

        notes: List[str] = []
        if leaked:
            notes.append(
                "Dropped as target leakage (near-perfect correlation): "
                + ", ".join(sorted(leaked))
            )

        # S15.4 — cache the fitted model + row-alignment metadata for what-if.
        if fingerprint:
            try:
                from src.analysis.ml import model_cache

                medians = {c: float(pd.to_numeric(work[c], errors="coerce").median())
                           for c in num_feats}
                model_cache.put(fingerprint, target, {
                    "model": model,
                    "task": task,
                    "target": target,
                    "columns": list(X.columns),
                    "numeric_features": num_feats,
                    "categorical_features": cat_feats,
                    "prefix_sep": "=",
                    "medians": medians,
                    "mae": metrics.get("mae_mean"),
                    "classes": [str(c) for c in getattr(model, "classes_", [])],
                })
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("model cache store failed: %s", e)

        return {
            "available": True,
            "task": task,
            "target": target,
            "n_rows_used": int(len(X)),
            "n_features": len(num_feats) + len(cat_feats),
            "numeric_features": num_feats,
            "categorical_features": cat_feats,
            "metrics": metrics,
            "importances": importances,
            "top_driver": top_feature,
            "verdict": _verdict(task, target, metrics),
            "chart": _importance_chart(target, importances),
            "notes": notes,
        }
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("ml_insights failed: %s", e)
        return {"available": False, "reason": f"error: {e}"}
