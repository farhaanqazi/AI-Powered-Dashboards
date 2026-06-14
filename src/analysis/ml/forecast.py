"""Phase 15 S15.3 (Tier D) — forecasting.

When Layer 2 detects a datetime column and a numeric measure, fit a
Holt-Winters / ETS model (statsmodels — *never* Prophet) and project a short
forecast with a confidence band that extends the existing time-series chart.

Defensive and deterministic: a missing statsmodels, no datetime/measure pair,
or too short a history yields ``{"available": False, "reason": ...}`` — never
an exception.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src import config
from src.analysis.data_structures import EnrichedProfile
from src.analysis.ml.supervised import _numeric_feature_cols

try:
    from src.logger import get_logger
    logger = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


def _pick_datetime(
    df: pd.DataFrame, profiles: Dict[str, EnrichedProfile]
) -> Optional[str]:
    for c in df.columns:
        p = profiles.get(c)
        if p is not None and p.role == "datetime":
            return c
    return None


def _pick_measure(
    df: pd.DataFrame, profiles: Dict[str, EnrichedProfile], exclude: str
) -> Optional[str]:
    best, best_var = None, -1.0
    for c in _numeric_feature_cols(df, profiles):
        if c == exclude:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.nunique() < 5:
            continue
        v = float(s.var() or 0.0)
        if v > best_var:
            best, best_var = c, v
    return best


def _build_series(
    df: pd.DataFrame, dt_col: str, measure: str
) -> Tuple[Optional[pd.Series], Optional[str]]:
    """Ordered, de-duplicated numeric series indexed by parsed datetime."""
    sub = df[[dt_col, measure]].copy()
    sub[dt_col] = pd.to_datetime(sub[dt_col], errors="coerce")
    sub[measure] = pd.to_numeric(sub[measure], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return None, None
    # Collapse duplicate timestamps (panel data) to a daily-mean grain.
    s = sub.groupby(dt_col)[measure].mean().sort_index()
    if len(s) < 2:
        return None, None
    freq = pd.infer_freq(s.index)
    return s, freq


def _verdict(measure: str, last: float, mean_fc: float) -> str:
    if last == 0:
        trend = "changing"
    else:
        change = (mean_fc - last) / abs(last)
        trend = "rising" if change > 0.02 else ("falling" if change < -0.02 else "roughly flat")
    return (
        f"Projecting '{measure}' forward with Holt-Winters exponential smoothing: "
        f"the near-term outlook is {trend}. The shaded band is a ~95% confidence "
        f"range — it widens with the forecast horizon, so treat the far end as "
        f"indicative, not exact."
    )


def compute_forecast(
    df: pd.DataFrame,
    enriched_profiles: Dict[str, EnrichedProfile],
) -> Dict[str, Any]:
    """Short Holt-Winters forecast + confidence band for a datetime+measure."""
    if not config.ML_INSIGHTS_ENABLED:
        return {"available": False, "reason": "disabled"}
    if df is None or df.empty:
        return {"available": False, "reason": "empty"}
    try:
        import statsmodels  # noqa: F401
    except Exception:
        return {"available": False, "reason": "statsmodels-unavailable"}

    try:
        dt_col = _pick_datetime(df, enriched_profiles)
        if dt_col is None:
            return {"available": False, "reason": "no-datetime"}
        measure = _pick_measure(df, enriched_profiles, exclude=dt_col)
        if measure is None:
            return {"available": False, "reason": "no-measure"}

        s, freq = _build_series(df, dt_col, measure)
        if s is None or len(s) < config.ML_FORECAST_MIN_POINTS:
            n = 0 if s is None else int(len(s))
            return {"available": False, "reason": "too-short-history",
                    "n_points": n, "target": measure}

        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        y = s.to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                y, trend="add", damped_trend=True, seasonal=None,
                initialization_method="estimated",
            ).fit()
        horizon = config.ML_FORECAST_HORIZON
        fc = np.asarray(model.forecast(horizon), dtype=float)
        resid = np.asarray(model.resid, dtype=float)
        sigma = float(np.std(resid)) if resid.size else 0.0
        # Band widens with the square root of the horizon step.
        steps = np.sqrt(np.arange(1, horizon + 1))
        margin = 1.96 * sigma * steps

        # Future timestamps continue the inferred cadence (fallback: +1 day).
        step = (s.index[-1] - s.index[-2]) if len(s) >= 2 else pd.Timedelta(days=1)
        future_idx = [s.index[-1] + step * (i + 1) for i in range(horizon)]

        hist_cap = 120
        hist = s.iloc[-hist_cap:]
        history = [{"date": d.isoformat(), "value": round(float(v), 4)}
                   for d, v in hist.items()]
        forecast = [
            {"date": future_idx[i].isoformat(),
             "yhat": round(float(fc[i]), 4),
             "lower": round(float(fc[i] - margin[i]), 4),
             "upper": round(float(fc[i] + margin[i]), 4)}
            for i in range(horizon)
        ]
        chart_data = (
            [{"date": h["date"], "value": h["value"], "kind": "history"} for h in history]
            + [{"date": f["date"], "value": f["yhat"], "kind": "forecast"} for f in forecast]
        )
        chart = {
            "title": f"{measure} — forecast",
            "type": "time_series",
            "intent": "forecast",
            "section": "Predictions",
            "x_title": dt_col,
            "y_title": measure,
            "data": chart_data,
        }
        return {
            "available": True,
            "method": "Holt-Winters (ETS, damped)",
            "target": measure,
            "date_column": dt_col,
            "frequency": freq,
            "horizon": horizon,
            "n_history": int(len(s)),
            "history": history,
            "forecast": forecast,
            "chart": chart,
            "verdict": _verdict(measure, float(y[-1]), float(np.mean(fc))),
        }
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("ml forecast failed: %s", e)
        return {"available": False, "reason": f"error: {e}"}
