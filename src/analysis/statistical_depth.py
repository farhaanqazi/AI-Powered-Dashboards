"""Phase 9 (S9.2) — deterministic statistical depth.

Every figure here is computed from the data, never by the LLM (the
"deterministic numbers, AI decorative" invariant). The module is defensive by
construction: it is one big collection of independently-guarded blocks. A
missing optional dependency (scikit-learn / statsmodels) or a degenerate column
yields ``{}`` for that block — never an exception — so the deployed image stays
correct (and slim) even without the heavy stack.

Public entrypoint: :func:`compute_statistical_depth`.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src import config
from src.analysis.data_structures import EnrichedProfile

try:
    from src.logger import get_logger
    logger = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


# --- optional engine probes ------------------------------------------------

def _engines() -> Dict[str, bool]:
    have = {"scipy": False, "sklearn": False, "statsmodels": False}
    try:
        import scipy  # noqa: F401
        have["scipy"] = True
    except Exception:
        pass
    try:
        import sklearn  # noqa: F401
        have["sklearn"] = True
    except Exception:
        pass
    try:
        import statsmodels  # noqa: F401
        have["statsmodels"] = True
    except Exception:
        pass
    return have


# --- column selection ------------------------------------------------------

def _numeric_cols(df: pd.DataFrame, profiles: Dict[str, EnrichedProfile]) -> List[str]:
    out = []
    for c in df.columns:
        p = profiles.get(c)
        if p is None or p.role == "identifier":
            continue
        if p.role in ("numeric", "ratio", "year", "boolean") or pd.api.types.is_numeric_dtype(df[c]):
            if pd.api.types.is_numeric_dtype(df[c]):
                out.append(c)
    return out[: config.STAT_DEPTH_MAX_COLS]


def _categorical_cols(df: pd.DataFrame, profiles: Dict[str, EnrichedProfile]) -> List[str]:
    out = []
    for c in df.columns:
        p = profiles.get(c)
        if p is None or p.role == "identifier":
            continue
        if p.role in ("categorical", "boolean"):
            nun = df[c].nunique(dropna=True)
            if 2 <= nun <= 50:
                out.append(c)
    return out[: config.STAT_DEPTH_MAX_COLS]


def _sample(df: pd.DataFrame) -> pd.DataFrame:
    cap = config.STAT_DEPTH_MAX_ROWS
    if len(df) > cap:
        return df.sample(cap, random_state=config.STAT_DEPTH_RANDOM_STATE)
    return df


# --- distributions ---------------------------------------------------------

def _distributions(df: pd.DataFrame, num_cols: List[str]) -> Dict[str, Any]:
    try:
        from scipy import stats
    except Exception:
        return {}
    out: Dict[str, Any] = {}
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 8 or s.nunique() < 3:
            continue
        try:
            skew = float(stats.skew(s))
            kurt = float(stats.kurtosis(s))  # excess kurtosis (Fisher)
            # D'Agostino K^2 is robust for n>=8 and not capped at 5000 like
            # Shapiro; fall back to Shapiro for tiny samples.
            if len(s) >= 20:
                stat, p = stats.normaltest(s)
                test = "dagostino_k2"
            else:
                stat, p = stats.shapiro(s)
                test = "shapiro"
            out[c] = {
                "skew": round(skew, 4),
                "kurtosis": round(kurt, 4),
                "normality_test": test,
                "normality_p": round(float(p), 6),
                "is_normal": bool(p > 0.05),
            }
        except Exception:
            continue
    return out


# --- associations ----------------------------------------------------------

def _spearman(df: pd.DataFrame, num_cols: List[str]) -> List[Dict[str, Any]]:
    try:
        from scipy import stats
    except Exception:
        return []
    out = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            a, b = num_cols[i], num_cols[j]
            pair = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 8:
                continue
            try:
                rho, p = stats.spearmanr(pair[a], pair[b])
                if math.isnan(rho):
                    continue
                out.append({"a": a, "b": b, "rho": round(float(rho), 4),
                            "p_value": round(float(p), 6)})
            except Exception:
                continue
    out.sort(key=lambda d: abs(d["rho"]), reverse=True)
    return out[:25]


def _cramers_v(df: pd.DataFrame, cat_cols: List[str]) -> List[Dict[str, Any]]:
    try:
        from scipy.stats import chi2_contingency
    except Exception:
        return []
    out = []
    for i in range(len(cat_cols)):
        for j in range(i + 1, len(cat_cols)):
            a, b = cat_cols[i], cat_cols[j]
            tbl = pd.crosstab(df[a], df[b])
            if tbl.size == 0 or tbl.shape[0] < 2 or tbl.shape[1] < 2:
                continue
            try:
                chi2, _, _, _ = chi2_contingency(tbl)
                n = tbl.to_numpy().sum()
                phi2 = chi2 / n
                r, k = tbl.shape
                denom = min(k - 1, r - 1)
                if denom <= 0 or n <= 0:
                    continue
                v = math.sqrt(phi2 / denom)
                out.append({"a": a, "b": b, "cramers_v": round(float(v), 4)})
            except Exception:
                continue
    out.sort(key=lambda d: d["cramers_v"], reverse=True)
    return out[:25]


def _correlation_ratio_eta(
    df: pd.DataFrame, cat_cols: List[str], num_cols: List[str]
) -> List[Dict[str, Any]]:
    """η (eta): strength of a categorical → numeric association."""
    out = []
    for cat in cat_cols:
        for num in num_cols:
            sub = df[[cat, num]].copy()
            sub[num] = pd.to_numeric(sub[num], errors="coerce")
            sub = sub.dropna()
            if len(sub) < 8 or sub[cat].nunique() < 2:
                continue
            try:
                grand = sub[num].mean()
                ss_total = float(((sub[num] - grand) ** 2).sum())
                if ss_total <= 0:
                    continue
                ss_between = 0.0
                for _, g in sub.groupby(cat, observed=True)[num]:
                    ss_between += len(g) * (g.mean() - grand) ** 2
                eta = math.sqrt(ss_between / ss_total)
                out.append({"category": cat, "numeric": num,
                            "eta": round(float(eta), 4)})
            except Exception:
                continue
    out.sort(key=lambda d: d["eta"], reverse=True)
    return out[:25]


# --- trend -----------------------------------------------------------------

def _mann_kendall(series: pd.Series) -> Dict[str, Any]:
    """Non-parametric monotonic-trend test (no statsmodels needed)."""
    try:
        from scipy.stats import norm
    except Exception:
        return {}
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    n = len(x)
    if n < 8:
        return {}
    s = 0
    for k in range(n - 1):
        s += np.sign(x[k + 1:] - x[k]).sum()
    # Variance with no tie correction (sufficient for a directional verdict).
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if var_s <= 0:
        return {}
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * (1 - norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))
    trend = "increasing" if (p < 0.05 and s > 0) else (
        "decreasing" if (p < 0.05 and s < 0) else "no-trend")
    return {"trend": trend, "tau": round(float(tau), 4),
            "p_value": round(float(p), 6), "z": round(float(z), 4)}


def _trend(
    df: pd.DataFrame,
    profiles: Dict[str, EnrichedProfile],
    num_cols: List[str],
) -> Dict[str, Any]:
    dt_col = None
    for c in df.columns:
        p = profiles.get(c)
        if p is not None and p.role == "datetime":
            dt_col = c
            break
    out: Dict[str, Any] = {}
    if dt_col is not None:
        try:
            order = pd.to_datetime(df[dt_col], errors="coerce")
            ordered = df.assign(_dt=order).dropna(subset=["_dt"]).sort_values("_dt")
        except Exception:
            ordered = df
    else:
        ordered = df

    have_sm = False
    try:
        from statsmodels.tsa.seasonal import STL
        have_sm = True
    except Exception:
        STL = None  # type: ignore

    for c in num_cols[:10]:
        mk = _mann_kendall(ordered[c])
        if not mk:
            continue
        entry: Dict[str, Any] = {"mann_kendall": mk}
        if have_sm and dt_col is not None:
            try:
                s = pd.to_numeric(ordered[c], errors="coerce").dropna()
                period = max(2, min(12, len(s) // 2 - 1))
                if len(s) >= 2 * period + 1:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = STL(s.to_numpy(), period=period, robust=True).fit()
                    var_resid = np.var(res.resid)
                    var_detrend = np.var(res.resid + res.seasonal)
                    var_deseason = np.var(res.resid + res.trend)
                    entry["stl"] = {
                        "period": period,
                        "seasonal_strength": round(
                            float(max(0.0, 1 - var_resid / var_detrend))
                            if var_detrend > 0 else 0.0, 4),
                        "trend_strength": round(
                            float(max(0.0, 1 - var_resid / var_deseason))
                            if var_deseason > 0 else 0.0, 4),
                    }
            except Exception:
                pass
        out[c] = entry
    return out


# --- anomalies -------------------------------------------------------------

def _numeric_matrix(df: pd.DataFrame, num_cols: List[str]):
    if len(num_cols) < 2:
        return None
    m = df[num_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(m) < 16:
        return None
    return m


def _anomalies(df: pd.DataFrame, num_cols: List[str]) -> Dict[str, Any]:
    m = _numeric_matrix(df, num_cols)
    if m is None:
        return {}
    out: Dict[str, Any] = {}
    rs = config.STAT_DEPTH_RANDOM_STATE
    try:
        from sklearn.ensemble import IsolationForest

        iso = IsolationForest(random_state=rs, contamination="auto")
        pred = iso.fit_predict(m.to_numpy())
        n_out = int((pred == -1).sum())
        out["isolation_forest"] = {
            "n_outliers": n_out,
            "fraction": round(n_out / len(m), 4),
        }
    except Exception:
        pass
    try:
        from sklearn.neighbors import LocalOutlierFactor

        lof = LocalOutlierFactor(n_neighbors=min(20, len(m) - 1))
        pred = lof.fit_predict(m.to_numpy())
        n_out = int((pred == -1).sum())
        out["lof"] = {
            "n_outliers": n_out,
            "fraction": round(n_out / len(m), 4),
        }
    except Exception:
        pass
    return out


# --- clustering ------------------------------------------------------------

def _clustering(df: pd.DataFrame, num_cols: List[str]) -> Dict[str, Any]:
    m = _numeric_matrix(df, num_cols)
    if m is None:
        return {}
    out: Dict[str, Any] = {}
    rs = config.STAT_DEPTH_RANDOM_STATE
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        X = StandardScaler().fit_transform(m.to_numpy())
        best = None
        for k in range(2, min(8, len(m) - 1)):
            km = KMeans(n_clusters=k, random_state=rs, n_init=10)
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            sil = silhouette_score(X, labels)
            if best is None or sil > best["silhouette"]:
                best = {"k": k, "silhouette": round(float(sil), 4)}
        if best:
            out["kmeans"] = best
        try:
            from sklearn.cluster import HDBSCAN

            hdb = HDBSCAN(min_cluster_size=max(5, len(m) // 50))
            labels = hdb.fit_predict(X)
            n_clusters = len({l for l in labels if l != -1})
            out["hdbscan"] = {
                "n_clusters": int(n_clusters),
                "n_noise": int((labels == -1).sum()),
            }
        except Exception:
            pass
    except Exception:
        pass
    return out


# --- driver analysis -------------------------------------------------------

def _pick_target(
    df: pd.DataFrame,
    profiles: Dict[str, EnrichedProfile],
    num_cols: List[str],
) -> str | None:
    additive = [
        c for c in num_cols
        if profiles.get(c) and "additive" in (profiles[c].semantic_tags or [])
    ]
    pool = additive or num_cols
    best, best_var = None, -1.0
    for c in pool:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.nunique() < 5:
            continue
        v = float(s.var())
        if v > best_var:
            best, best_var = c, v
    return best


def _drivers(
    df: pd.DataFrame,
    profiles: Dict[str, EnrichedProfile],
    num_cols: List[str],
) -> Dict[str, Any]:
    if len(num_cols) < 2:
        return {}
    target = _pick_target(df, profiles, num_cols)
    if target is None:
        return {}
    features = [c for c in num_cols if c != target]
    if not features:
        return {}
    data = df[[target] + features].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 24:
        return {}
    rs = config.STAT_DEPTH_RANDOM_STATE
    out: Dict[str, Any] = {"target": target}
    try:
        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(
            n_estimators=120, random_state=rs, n_jobs=1, max_depth=12
        )
        rf.fit(data[features].to_numpy(), data[target].to_numpy())
        imp = sorted(
            (
                {"feature": f, "importance": round(float(v), 4)}
                for f, v in zip(features, rf.feature_importances_)
            ),
            key=lambda d: d["importance"],
            reverse=True,
        )
        out["method"] = "RandomForestRegressor"
        out["importances"] = imp[:15]
    except Exception:
        return {}
    try:
        from sklearn.feature_selection import mutual_info_regression

        mi = mutual_info_regression(
            data[features].to_numpy(), data[target].to_numpy(), random_state=rs
        )
        out["mutual_information"] = sorted(
            (
                {"feature": f, "mi": round(float(v), 4)}
                for f, v in zip(features, mi)
            ),
            key=lambda d: d["mi"],
            reverse=True,
        )[:15]
    except Exception:
        pass
    return out


# --- entrypoint ------------------------------------------------------------

def compute_statistical_depth(
    df: pd.DataFrame,
    enriched_profiles: Dict[str, EnrichedProfile],
) -> Dict[str, Any]:
    """Compute the full deterministic statistical-depth report.

    Never raises: every block is independently guarded and degrades to an
    empty section. Returns ``{"available": False}`` when disabled by config.
    """
    if not config.STATISTICAL_DEPTH_ENABLED:
        return {"available": False, "reason": "disabled"}
    if df is None or df.empty:
        return {"available": False, "reason": "empty"}

    engines = _engines()
    work = _sample(df)
    num_cols = _numeric_cols(work, enriched_profiles)
    cat_cols = _categorical_cols(work, enriched_profiles)

    report: Dict[str, Any] = {
        "available": True,
        "engines": engines,
        "n_rows_sampled": int(len(work)),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
    }
    try:
        report["distributions"] = _distributions(work, num_cols)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("statistical_depth.distributions failed: %s", e)
        report["distributions"] = {}
    try:
        report["associations"] = {
            "spearman": _spearman(work, num_cols),
            "cramers_v": _cramers_v(work, cat_cols),
            "correlation_ratio_eta": _correlation_ratio_eta(
                work, cat_cols, num_cols
            ),
        }
    except Exception as e:  # pragma: no cover
        logger.warning("statistical_depth.associations failed: %s", e)
        report["associations"] = {}
    for name, fn in (
        ("trend", lambda: _trend(work, enriched_profiles, num_cols)),
        ("anomalies", lambda: _anomalies(work, num_cols)),
        ("clustering", lambda: _clustering(work, num_cols)),
        ("drivers", lambda: _drivers(work, enriched_profiles, num_cols)),
    ):
        try:
            report[name] = fn()
        except Exception as e:  # pragma: no cover
            logger.warning("statistical_depth.%s failed: %s", name, e)
            report[name] = {}
    return report
