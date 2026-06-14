"""Phase 15 S15.4 — deterministic what-if prediction + re-segment.

The supervised model fitted at analysis time (S15.1) is reused from the
in-process model cache to score user-supplied feature values. There is **no
retrain and no LLM** on this path — given the same inputs the answer is
identical, and every result carries a provenance token. Re-segment reruns the
deterministic KMeans (S15.2) over a user-chosen column subset.

Both entrypoints never raise: a cache miss / bad input comes back as a status,
mirroring the ``df_cache`` "data expired → re-run" UX.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

import pandas as pd

from src import config
from src.analysis.ml import model_cache


def _token(prefix: str, *parts: Any) -> str:
    raw = "|".join(json.dumps(p, sort_keys=True, default=str) for p in parts)
    return f"{prefix}:" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def run_predict(
    fingerprint: str,
    feature_values: Dict[str, Any],
    target: Optional[str] = None,
) -> Dict[str, Any]:
    """Score one row of user-supplied feature values against the cached model.

    Returns ``status='unavailable'`` on a cache miss (model expired → re-run),
    ``status='ok'`` with the prediction + a confidence band/probability."""
    if not config.ML_INSIGHTS_ENABLED:
        return {"status": "disabled", "result": None}
    bundle = model_cache.get(fingerprint, target)
    if bundle is None:
        return {
            "status": "unavailable",
            "error": "The trained model for this analysis has expired. "
                     "Re-run the analysis to predict again.",
        }
    try:
        cols = bundle["columns"]
        sep = bundle["prefix_sep"]
        row = {c: 0.0 for c in cols}
        used: Dict[str, Any] = {}

        for c in bundle["numeric_features"]:
            v = feature_values.get(c)
            if v is None or v == "":
                row[c] = bundle["medians"].get(c, 0.0)
            else:
                try:
                    row[c] = float(v)
                    used[c] = float(v)
                except (TypeError, ValueError):
                    row[c] = bundle["medians"].get(c, 0.0)

        unknown_cats: List[str] = []
        for c in bundle["categorical_features"]:
            v = feature_values.get(c)
            if v is None or v == "":
                continue
            key = f"{c}{sep}{v}"
            if key in row:
                row[key] = 1.0
                used[c] = str(v)
            else:
                unknown_cats.append(f"{c}={v}")

        X = pd.DataFrame([[row[c] for c in cols]], columns=cols).to_numpy(dtype=float)
        model = bundle["model"]
        task = bundle["task"]
        if task == "regression":
            yhat = float(model.predict(X)[0])
            mae = float(bundle.get("mae") or 0.0)
            result = {
                "prediction": round(yhat, 4),
                "lower": round(yhat - mae, 4),
                "upper": round(yhat + mae, 4),
                "confidence_basis": "±1 cross-validated MAE",
            }
        else:
            cls = str(model.predict(X)[0])
            proba = None
            try:
                p = model.predict_proba(X)[0]
                proba = round(float(max(p)), 4)
            except Exception:
                pass
            result = {"prediction": cls, "probability": proba}

        notes = []
        if unknown_cats:
            notes.append("Unseen categories treated as baseline: "
                         + ", ".join(unknown_cats))
        return {
            "status": "ok",
            "task": task,
            "target": bundle["target"],
            "inputs_used": used,
            "result": result,
            "notes": notes,
            "provenance": {
                "token": _token("predict", fingerprint, bundle["target"], used),
                "verified": True,
                "retrained": False,
            },
            "numbers_traceable": True,
        }
    except Exception:  # noqa: BLE001 - never break the request
        return {"status": "error", "error": "Prediction failed for these inputs."}


def run_resegment(
    fingerprint: str,
    df: pd.DataFrame,
    columns: List[str],
) -> Dict[str, Any]:
    """Re-run KMeans segmentation over a user-chosen column subset (no LLM)."""
    if not config.ML_INSIGHTS_ENABLED:
        return {"status": "disabled", "result": None}
    if not columns or len(columns) < 2:
        return {"status": "error",
                "error": "Pick at least two columns to segment on."}
    from src.analysis.ml import compute_segments

    seg = compute_segments(df, {}, columns=list(columns))
    if not seg.get("available"):
        return {"status": "unavailable", "error": seg.get("reason"), "segments": seg}
    return {
        "status": "ok",
        "segments": seg,
        "provenance": {
            "token": _token("resegment", fingerprint, sorted(columns)),
            "verified": True,
            "retrained": False,
        },
        "numbers_traceable": True,
    }
