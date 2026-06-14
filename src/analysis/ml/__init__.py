"""Phase 15 — Machine Learning Insights.

Deterministic, contract-aware modelling layered on top of the analysis
pipeline. Every figure is computed (cross-validated metrics, permutation
importance, silhouette, forecasts), never produced by the LLM, mirroring the
``statistical_depth`` invariant. Each public entrypoint never raises: a missing
scikit-learn / statsmodels or a degenerate dataset degrades to
``{"available": False, "reason": ...}``.

The pipeline folds these into ``eda_summary["ml_insights"]`` as
``{"supervised", "segments", "anomalies", "forecast"}``.
"""
from src.analysis.ml.forecast import compute_forecast
from src.analysis.ml.supervised import compute_ml_insights
from src.analysis.ml.unsupervised import compute_anomalies, compute_segments

__all__ = [
    "compute_ml_insights",
    "compute_segments",
    "compute_anomalies",
    "compute_forecast",
]
