"""Phase 15 S15.2 (Tier B+C) — unsupervised segmentation + anomaly detection.

* :func:`compute_segments` — KMeans (k chosen by silhouette) on scaled numeric
  + one-hot categorical features, a PCA→2D scatter for the dashboard, and the
  *distinguishing* features of each segment (how its mean deviates from the
  whole). HDBSCAN is reported alongside when available.
* :func:`compute_anomalies` — IsolationForest row-level outlier detection with
  the features that most contribute to the flagged rows, complementing the
  column-level Data-Quality tab.

Both mirror the ``statistical_depth`` discipline: deterministic (seeded, row
capped), import-light, and defensive — a missing scikit-learn or a degenerate
frame yields ``{"available": False, "reason": ...}`` rather than an exception.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src import config
from src.analysis.data_structures import EnrichedProfile
from src.analysis.ml.supervised import (
    _categorical_feature_cols,
    _numeric_feature_cols,
    _sample,
)

try:
    from src.logger import get_logger
    logger = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


def _scaled_matrix(
    work: pd.DataFrame,
    num_feats: List[str],
    cat_feats: List[str],
) -> Tuple[Any, List[str], Any, pd.Index]:
    """Impute (median) + standard-scale a numeric/one-hot matrix.

    Returns ``(X_scaled, feature_names, scaler, row_index)`` or
    ``(None, [], None, None)`` when nothing usable remains.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    frames: List[pd.DataFrame] = []
    for c in num_feats:
        frames.append(pd.to_numeric(work[c], errors="coerce").rename(c).to_frame())
    width = len(num_feats)
    for c in sorted(cat_feats, key=lambda c: int(work[c].nunique(dropna=True))):
        d = pd.get_dummies(work[c].astype("object"), prefix=c, prefix_sep="=",
                           dummy_na=False)
        if width + d.shape[1] > config.ML_MAX_FEATURES:
            continue
        frames.append(d)
        width += d.shape[1]
    if not frames:
        return None, [], None, None
    X = pd.concat(frames, axis=1)
    # Drop rows that are entirely missing across the matrix.
    X = X.dropna(how="all")
    if X.empty:
        return None, [], None, None
    names = list(X.columns)
    imputed = SimpleImputer(strategy="median").fit_transform(X.to_numpy(dtype=float))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(imputed)
    return Xs, names, scaler, X.index


def _distinguishing(
    Xs: Any, names: List[str], labels: Any, cluster: int, top: int = 3
) -> List[Dict[str, Any]]:
    """Features whose standardized mean in `cluster` deviates most from 0."""
    mask = labels == cluster
    if not mask.any():
        return []
    means = Xs[mask].mean(axis=0)
    order = np.argsort(-np.abs(means))
    out = []
    for idx in order[:top]:
        z = float(means[idx])
        if abs(z) < 0.1:
            continue
        out.append({
            "feature": names[idx],
            "direction": "higher" if z > 0 else "lower",
            "z": round(z, 3),
        })
    return out


def compute_segments(
    df: pd.DataFrame,
    enriched_profiles: Dict[str, EnrichedProfile],
    columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """KMeans segmentation + PCA scatter + per-segment distinguishing features.

    When ``columns`` is given (S15.4 re-segment), the feature set is restricted
    to those columns and split numeric/categorical by dtype — so it works from
    just the cleaned frame, without Layer-2 profiles."""
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
        if columns:
            present = [c for c in columns if c in work.columns]
            num_feats = [c for c in present
                         if pd.api.types.is_numeric_dtype(work[c])]
            cat_feats = [c for c in present
                         if not pd.api.types.is_numeric_dtype(work[c])
                         and 2 <= int(work[c].nunique(dropna=True))
                         <= config.ML_MAX_CAT_CARDINALITY]
        else:
            num_feats = _numeric_feature_cols(work, enriched_profiles)
            cat_feats = _categorical_feature_cols(work, enriched_profiles)
        if len(num_feats) < 2:
            return {"available": False, "reason": "too-few-numeric-features"}

        Xs, names, _, _ = _scaled_matrix(work, num_feats, cat_feats)
        if Xs is None or len(Xs) < config.ML_MIN_ROWS:
            n = 0 if Xs is None else int(len(Xs))
            return {"available": False, "reason": "too-few-rows", "n_rows": n}

        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        rs = config.ML_RANDOM_STATE
        best = None
        upper = min(config.ML_MAX_SEGMENTS + 1, len(Xs))
        for k in range(2, max(3, upper)):
            km = KMeans(n_clusters=k, random_state=rs, n_init=10)
            labels = km.fit_predict(Xs)
            if len(set(labels)) < 2:
                continue
            sil = float(silhouette_score(Xs, labels))
            if best is None or sil > best["silhouette"]:
                best = {"k": k, "silhouette": sil, "labels": labels}
        if best is None:
            return {"available": False, "reason": "no-separation"}

        labels = best["labels"]
        k = best["k"]
        sizes = pd.Series(labels).value_counts().sort_index()
        segments = []
        for c in range(k):
            segments.append({
                "label": f"Segment {c + 1}",
                "size": int(sizes.get(c, 0)),
                "share": round(float(sizes.get(c, 0)) / len(labels), 4),
                "distinguishing": _distinguishing(Xs, names, labels, c),
            })

        # PCA → 2D scatter (capped points for a lean payload).
        chart = None
        try:
            from sklearn.decomposition import PCA

            coords = PCA(n_components=2, random_state=rs).fit_transform(Xs)
            cap = config.ML_SCATTER_MAX_POINTS
            if len(coords) > cap:
                rng = np.random.default_rng(rs)
                sel = rng.choice(len(coords), size=cap, replace=False)
            else:
                sel = np.arange(len(coords))
            data = [
                {"x": round(float(coords[i, 0]), 4),
                 "y": round(float(coords[i, 1]), 4),
                 "group": f"Segment {int(labels[i]) + 1}"}
                for i in sel
            ]
            chart = {
                "title": "Customer / row segments (PCA projection)",
                "type": "scatter",
                "intent": "segment_scatter",
                "section": "Predictions",
                "x_title": "Principal component 1",
                "y_title": "Principal component 2",
                "data": data,
            }
        except Exception:
            pass

        result = {
            "available": True,
            "method": "KMeans",
            "k": k,
            "silhouette": round(best["silhouette"], 4),
            "n_rows_used": int(len(Xs)),
            "n_features": len(names),
            "segments": segments,
        }
        if chart is not None:
            result["chart"] = chart

        try:
            from sklearn.cluster import HDBSCAN

            hdb = HDBSCAN(min_cluster_size=max(5, len(Xs) // 50))
            hl = hdb.fit_predict(Xs)
            result["hdbscan"] = {
                "n_clusters": int(len({l for l in hl if l != -1})),
                "n_noise": int((hl == -1).sum()),
            }
        except Exception:
            pass
        return result
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("ml segments failed: %s", e)
        return {"available": False, "reason": f"error: {e}"}


def compute_anomalies(
    df: pd.DataFrame,
    enriched_profiles: Dict[str, EnrichedProfile],
) -> Dict[str, Any]:
    """IsolationForest row-level outliers + top contributing features."""
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
        num_feats = _numeric_feature_cols(work, enriched_profiles)
        if len(num_feats) < 2:
            return {"available": False, "reason": "too-few-numeric-features"}

        Xs, names, _, index = _scaled_matrix(work, num_feats, [])
        if Xs is None or len(Xs) < config.ML_MIN_ROWS:
            n = 0 if Xs is None else int(len(Xs))
            return {"available": False, "reason": "too-few-rows", "n_rows": n}

        from sklearn.ensemble import IsolationForest

        rs = config.ML_RANDOM_STATE
        iso = IsolationForest(random_state=rs, contamination="auto")
        pred = iso.fit_predict(Xs)
        out_mask = pred == -1
        n_out = int(out_mask.sum())
        if n_out == 0:
            return {
                "available": True, "method": "IsolationForest",
                "n_rows_used": int(len(Xs)), "n_outliers": 0, "fraction": 0.0,
                "top_features": [], "example_rows": [],
            }

        # Top contributing features: where outliers deviate most (|mean z|).
        contrib = np.abs(Xs[out_mask].mean(axis=0))
        order = np.argsort(-contrib)
        top_features = [
            {"feature": names[i], "contribution": round(float(contrib[i]), 4)}
            for i in order[:config.ML_TOP_FEATURES] if contrib[i] > 0.1
        ]
        # A few example flagged source-row indices (most anomalous first).
        scores = iso.score_samples(Xs)  # lower = more anomalous
        flagged = np.where(out_mask)[0]
        worst = flagged[np.argsort(scores[flagged])][:10]
        example_rows = [int(index[i]) for i in worst]

        return {
            "available": True,
            "method": "IsolationForest",
            "n_rows_used": int(len(Xs)),
            "n_outliers": n_out,
            "fraction": round(n_out / len(Xs), 4),
            "top_features": top_features,
            "example_rows": example_rows,
        }
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("ml anomalies failed: %s", e)
        return {"available": False, "reason": f"error: {e}"}
