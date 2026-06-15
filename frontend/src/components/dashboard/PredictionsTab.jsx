import React, { useState } from 'react';
import ChartRenderer from '../charts/ChartRenderer';
import LazyMount from '../charts/LazyMount';
import { runInteraction } from '../../services/api';

// Phase 15 — ML insights. Every number here is computed server-side
// (cross-validated metrics, permutation importance, silhouette, Holt-Winters)
// and shipped in eda_summary.ml_insights = {supervised, segments, anomalies,
// forecast}; this tab only renders it. Reason codes map to honest empty-state
// copy rather than implying failure.
const REASON_COPY = {
  disabled: 'Predictive modelling is turned off for this workspace.',
  'no-suitable-target':
    'No column looks like an outcome to predict — add a numeric measure or a labelled category.',
  'no-features': 'There are no usable predictor columns once identifiers are excluded.',
  'too-few-rows': 'There are too few rows to cross-validate a trustworthy model.',
  'too-few-numeric-features': 'Segmentation needs at least two numeric columns.',
  'degenerate-target': 'The target column does not vary enough to model.',
  'too-short-history': 'The time series is too short to forecast reliably.',
  'no-datetime': 'No date/time column was detected for a forecast.',
  'no-measure': 'No numeric measure was available to forecast.',
  'sklearn-unavailable': 'The modelling engine is not available in this deployment.',
  'statsmodels-unavailable': 'The forecasting engine is not available in this deployment.',
  empty: 'No data is available for this dataset.',
};

const directionTone = (d) => {
  switch (d) {
    case 'positive': return { cls: 'neon-emerald', icon: 'fa-arrow-trend-up', label: 'raises' };
    case 'negative': return { cls: 'neon-rose', icon: 'fa-arrow-trend-down', label: 'lowers' };
    case 'flat':     return { cls: 'neon-amber', icon: 'fa-arrows-left-right', label: 'mixed' };
    default:         return null;
  }
};

// --- Plain-language helpers (translate ML output for non-experts) ----------
// The silhouette score (0–1) measures how cleanly the groups separate. Laymen
// don't know the scale, so map it to an honest phrase; keep the number on hover.
const separationLabel = (s) => {
  const v = Number(s);
  if (!Number.isFinite(v)) return { word: 'rough grouping', tone: 'neon-amber' };
  if (v < 0.25) return { word: 'groups overlap a lot', tone: 'neon-amber' };
  if (v < 0.5)  return { word: 'loosely separated', tone: 'neon-blue' };
  if (v < 0.7)  return { word: 'clearly separated', tone: 'neon-emerald' };
  return { word: 'very distinct groups', tone: 'neon-emerald' };
};

// Turn an internal feature signal (incl. one-hot "Col=Value") into a sentence.
const humanizeFeature = (feature, direction) => {
  const up = direction === 'higher';
  if (typeof feature === 'string' && feature.includes('=')) {
    const idx = feature.indexOf('=');
    const col = feature.slice(0, idx).replace(/_/g, ' ');
    const val = feature.slice(idx + 1);
    return `${up ? 'More often' : 'Less often'} ${col} = ${val}`;
  }
  const nice = String(feature).replace(/_/g, ' ');
  return `${up ? 'Higher' : 'Lower'} ${nice} than average`;
};

const prettyName = (s) => String(s).replace(/_/g, ' ');

const Metric = ({ label, value, hint }) => (
  <div className="glass-soft p-4 rounded-xl">
    <div className="text-[10px] uppercase tracking-[0.28em] text-slate-400 mb-1">{label}</div>
    <div className="text-2xl font-semibold text-slate-100">{value}</div>
    {hint && <div className="text-xs text-slate-400 mt-1">{hint}</div>}
  </div>
);

const ChartCard = ({ icon, color, title, badge, badgeTone, chart }) => (
  <div className="glass-card p-6 chart-card">
    <div className="flex items-start justify-between mb-5 gap-4">
      <h2 className="text-base md:text-lg font-semibold text-slate-100 flex items-center gap-3">
        <span className="section-icon"><i className={`fas ${icon}`} style={{ color }} /></span>
        <span className="truncate">{title}</span>
      </h2>
      {badge && <span className={`neon-badge ${badgeTone} flex-shrink-0`}>{badge}</span>}
    </div>
    <div className="chart-container chart-shell h-96">
      <LazyMount minHeight={300}><ChartRenderer chartData={chart} /></LazyMount>
    </div>
  </div>
);

// --- What-if predict (S15.4) ----------------------------------------------
const WhatIfPanel = ({ ml }) => {
  // The top numeric drivers make the most useful what-if knobs.
  const numeric = ml.numeric_features || [];
  const ranked = (ml.importances || []).map((i) => i.feature).filter((f) => numeric.includes(f));
  const fields = (ranked.length ? ranked : numeric).slice(0, 6);
  const [values, setValues] = useState({});
  const [state, setState] = useState({ status: 'idle' });

  if (!fields.length) return null;

  const submit = async () => {
    setState({ status: 'loading' });
    try {
      const features = {};
      for (const f of fields) if (values[f] !== '' && values[f] != null) features[f] = Number(values[f]);
      const res = await runInteraction({ calculation: 'predict', params: { features } });
      setState({ status: 'done', res });
    } catch (e) {
      setState({ status: 'error', error: e?.response?.data?.detail || e.message });
    }
  };

  const r = state.res?.result;
  return (
    <div className="glass-card p-6">
      <div className="flex items-center justify-between mb-4 gap-3">
        <h3 className="text-base font-semibold text-slate-100 flex items-center gap-3">
          <span className="section-icon"><i className="fas fa-sliders" style={{ color: '#34d399' }} /></span>
          What-if — predict {ml.target}
        </h3>
        <span className="neon-badge neon-emerald flex-shrink-0">No retrain</span>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-4">
        {fields.map((f) => (
          <label key={f} className="block">
            <span className="block text-xs text-slate-400 mb-1 truncate">{f}</span>
            <input
              type="number"
              className="w-full bg-slate-900/60 border border-white/10 rounded-lg px-3 py-2 text-slate-100 text-sm"
              value={values[f] ?? ''}
              placeholder="median"
              onChange={(e) => setValues((v) => ({ ...v, [f]: e.target.value }))}
            />
          </label>
        ))}
      </div>
      <button
        type="button"
        onClick={submit}
        disabled={state.status === 'loading'}
        className="neon-badge neon-blue px-4 py-2 text-sm disabled:opacity-50"
      >
        {state.status === 'loading' ? 'Predicting…' : 'Predict'}
      </button>

      {state.status === 'done' && state.res?.status === 'ok' && (
        <div className="glass-soft p-4 rounded-xl mt-4">
          <div className="text-[10px] uppercase tracking-[0.28em] text-slate-400 mb-1">Predicted {ml.target}</div>
          <div className="text-2xl font-semibold text-slate-100">
            {typeof r.prediction === 'number' ? r.prediction.toLocaleString() : r.prediction}
            {r.probability != null && <span className="text-sm text-slate-400 ml-2">({Math.round(r.probability * 100)}% confidence)</span>}
          </div>
          {r.lower != null && (
            <div className="text-xs text-slate-400 mt-1">Range {r.lower.toLocaleString()} – {r.upper.toLocaleString()} ({r.confidence_basis})</div>
          )}
        </div>
      )}
      {state.status === 'done' && state.res?.status === 'unavailable' && (
        <p className="text-xs text-amber-300 mt-3">{state.res.error}</p>
      )}
      {state.status === 'error' && <p className="text-xs text-rose-300 mt-3">{state.error}</p>}
    </div>
  );
};

// --- Supervised (S15.1) ---------------------------------------------------
const SupervisedSection = ({ ml }) => {
  const m = ml.metrics || {};
  const isReg = ml.task === 'regression';
  const importances = ml.importances || [];
  const headlineMetrics = isReg
    ? [
        { label: 'R² (cross-validated)', value: (m.r2_mean ?? 0).toFixed(2),
          hint: `explains ~${Math.round(Math.max(0, m.r2_mean ?? 0) * 100)}% of the variation` },
        { label: 'Typical error (MAE)', value: `±${Number(m.mae_mean ?? 0).toPrecision(3)}`,
          hint: `units of ${ml.target}` },
        { label: 'Linear baseline R²', value: (m.baseline_r2_mean ?? 0).toFixed(2), hint: m.baseline_model },
      ]
    : [
        { label: 'F1 (cross-validated)', value: (m.f1_mean ?? 0).toFixed(2),
          hint: `weighted across ${m.n_classes} classes` },
        { label: 'Accuracy', value: `${Math.round((m.accuracy_mean ?? 0) * 100)}%`,
          hint: `vs ${Math.round((m.majority_baseline_accuracy ?? 0) * 100)}% guessing '${m.majority_class}'` },
        { label: 'Logistic baseline F1', value: (m.baseline_f1_mean ?? 0).toFixed(2), hint: m.baseline_model },
      ];
  return (
    <>
      <div className="glass-card p-6">
        <div className="flex items-start justify-between gap-4 mb-3">
          <div className="min-w-0">
            <span className="block text-[10px] uppercase tracking-[0.32em] text-slate-400 mb-1">Predictive model</span>
            <h2 className="text-xl md:text-2xl font-semibold text-slate-100 truncate">
              What drives <span className="text-cyan-300">{ml.target}</span>
            </h2>
          </div>
          <span className={`neon-badge ${isReg ? 'neon-blue' : 'neon-purple'} flex-shrink-0`}>
            {isReg ? 'Regression' : 'Classification'}
          </span>
        </div>
        <p className="text-sm md:text-base text-slate-200 leading-relaxed">{ml.verdict}</p>
        <div className="text-xs text-slate-400 mt-3">
          {m.model} · {ml.n_rows_used?.toLocaleString()} rows · {ml.n_features} candidate drivers · {m.cv_folds}-fold CV
        </div>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {headlineMetrics.map((mt) => <Metric key={mt.label} {...mt} />)}
      </div>
      {ml.chart && (
        <ChartCard icon="fa-ranking-star" color="#22d3ee" title="Driver importance"
          badge="Permutation" badgeTone="neon-emerald" chart={ml.chart} />
      )}
      <div className="glass-card p-6">
        <h3 className="text-base font-semibold text-slate-100 mb-4">Ranked drivers</h3>
        <ul className="space-y-2">
          {importances.map((it, i) => {
            const tone = directionTone(it.direction);
            return (
              <li key={it.feature} className="flex items-center justify-between gap-3 py-2 border-b border-white/5 last:border-0">
                <span className="flex items-center gap-3 min-w-0">
                  <span className="text-slate-500 text-sm w-5 text-right">{i + 1}</span>
                  <span className="text-slate-100 truncate">{it.feature}</span>
                  {tone && (
                    <span className={`neon-badge ${tone.cls} text-[10px]`}>
                      <i className={`fas ${tone.icon} mr-1`} />{tone.label} {ml.target}
                    </span>
                  )}
                </span>
                <span className="text-slate-300 text-sm tabular-nums flex-shrink-0">{Number(it.importance).toFixed(3)}</span>
              </li>
            );
          })}
        </ul>
        {(ml.notes || []).map((n) => (
          <p key={n} className="text-xs text-slate-400 mt-3"><i className="fas fa-circle-info mr-1" />{n}</p>
        ))}
      </div>
    </>
  );
};

// --- Segments (S15.2) -----------------------------------------------------
const SegmentsSection = ({ seg }) => {
  const sep = separationLabel(seg.silhouette);
  return (
  <>
    <div className="glass-card p-6">
      <div className="flex items-start justify-between gap-4 mb-2">
        <h2 className="text-lg md:text-xl font-semibold text-slate-100">
          We found {seg.k} natural groups in your data
        </h2>
        <span
          className={`neon-badge ${sep.tone} flex-shrink-0`}
          title={`Silhouette score ${seg.silhouette} (0–1; higher means the groups separate more cleanly)`}
        >
          {sep.word}
        </span>
      </div>
      <p className="text-sm text-slate-300">
        Your {seg.n_rows_used?.toLocaleString()} rows were sorted into {seg.k} groups of
        rows that behave alike, by comparing {seg.n_features} columns at once. Nobody
        labelled these — the groups emerge from the data itself.
      </p>
    </div>
    {seg.chart && (
      <ChartCard icon="fa-shapes" color="#a78bfa" title="Group map"
        badge="Auto-grouped" badgeTone="neon-purple" chart={seg.chart} />
    )}
    <div className="glass-soft p-4 rounded-xl">
      <p className="text-xs text-slate-400 leading-relaxed">
        <i className="fas fa-circle-info mr-2 text-slate-500" />
        Each dot is one row. Rows that are alike sit close together, so look for
        the separate clouds — and how much they overlap. The two axes are just a
        compressed summary of all {seg.n_features} columns, so the exact position
        of any single dot isn't meant to be read directly.
      </p>
    </div>
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {(seg.segments || []).map((s) => (
        <div key={s.label} className="glass-soft p-4 rounded-xl">
          <div className="flex items-center justify-between mb-2">
            <span className="font-semibold text-slate-100">{s.label}</span>
            <span className="text-xs text-slate-400">{Math.round(s.share * 100)}% · {s.size.toLocaleString()} rows</span>
          </div>
          <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500 mb-1.5">What stands out here</div>
          <ul className="space-y-1">
            {(s.distinguishing || []).map((d) => (
              <li key={d.feature} className="text-xs text-slate-300 flex items-center gap-2">
                <i className={`fas ${d.direction === 'higher' ? 'fa-arrow-up text-emerald-300' : 'fa-arrow-down text-rose-300'} flex-shrink-0`} />
                <span className="truncate" title={humanizeFeature(d.feature, d.direction)}>
                  {humanizeFeature(d.feature, d.direction)}
                </span>
              </li>
            ))}
            {!(s.distinguishing || []).length && (
              <li className="text-xs text-slate-500">Pretty average across the board.</li>
            )}
          </ul>
        </div>
      ))}
    </div>
  </>
  );
};

// --- Anomalies (S15.2) ----------------------------------------------------
const AnomaliesSection = ({ an }) => {
  const pct = Math.round(an.fraction * 100);
  const many = pct >= 20;
  return (
  <div className="glass-card p-6">
    <div className="flex items-start justify-between gap-4 mb-3">
      <h2 className="text-lg font-semibold text-slate-100 flex items-center gap-3">
        <span className="section-icon"><i className="fas fa-triangle-exclamation" style={{ color: '#fb7185' }} /></span>
        Rows that look unusual
      </h2>
      <span
        className="neon-badge neon-rose flex-shrink-0"
        title="Found automatically with an Isolation Forest outlier detector"
      >
        {an.n_outliers.toLocaleString()} · {pct}%
      </span>
    </div>
    <p className="text-sm text-slate-300 mb-1">
      {an.n_outliers.toLocaleString()} of {an.n_rows_used?.toLocaleString()} rows ({pct}%) stand
      out as different from the typical pattern in your data.
    </p>
    {many && (
      <p className="text-xs text-amber-300/80 mb-2">
        That's a large share — usually a sign the data is naturally varied, not that
        a third of it is wrong. Treat it as worth a look, not an alarm.
      </p>
    )}
    {!!(an.top_features || []).length && (
      <>
        <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500 mb-2 mt-3">What makes them stand out</div>
        <div className="flex flex-wrap gap-2">
          {an.top_features.map((f, i) => (
            <span
              key={f.feature}
              className="neon-badge neon-amber text-[11px]"
              title={`Contribution score ${f.contribution}`}
            >
              {i + 1}. {prettyName(f.feature)}
            </span>
          ))}
        </div>
      </>
    )}
  </div>
  );
};

// --- Forecast (S15.3) -----------------------------------------------------
const ForecastSection = ({ fc }) => (
  <>
    <div className="glass-card p-6">
      <div className="flex items-start justify-between gap-4 mb-2">
        <h2 className="text-lg md:text-xl font-semibold text-slate-100">
          Forecast — <span className="text-cyan-300">{fc.target}</span>
        </h2>
        <span className="neon-badge neon-blue flex-shrink-0">{fc.method}</span>
      </div>
      <p className="text-sm text-slate-300">{fc.verdict}</p>
    </div>
    {fc.chart && (
      <ChartCard icon="fa-chart-line" color="#60a5fa" title={`${fc.target} — history + ${fc.horizon}-step forecast`}
        badge="Holt-Winters" badgeTone="neon-blue" chart={fc.chart} />
    )}
  </>
);

const Unavailable = ({ title, reason }) => (
  <div className="glass-soft p-4 rounded-xl">
    <div className="text-sm text-slate-300 font-medium mb-1">{title}</div>
    <div className="text-xs text-slate-400">{REASON_COPY[reason] || 'Not available for this dataset.'}</div>
  </div>
);

const PredictionsTab = ({ data, loading, error }) => {
  const bundle = data?.eda_summary?.ml_insights;

  if (loading) {
    return (
      <section id="predictions-section" className="analysis-section">
        <div className="flex justify-center items-center py-20"><div className="dash-spinner" /></div>
      </section>
    );
  }
  if (error) {
    return (
      <section id="predictions-section" className="analysis-section">
        <div className="glass-soft p-5 text-rose-200 border-rose-400/30">{error}</div>
      </section>
    );
  }

  const sup = bundle?.supervised;
  const seg = bundle?.segments;
  const an = bundle?.anomalies;
  const fc = bundle?.forecast;
  const anyAvailable = [sup, seg, an, fc].some((x) => x?.available);

  if (!bundle || !anyAvailable) {
    const reason = REASON_COPY[sup?.reason] || 'Predictive modelling is not available for this dataset.';
    return (
      <section id="predictions-section" className="analysis-section">
        <div className="empty-state">
          <i className="fas fa-wand-magic-sparkles empty-icon" />
          <p>No predictive insights for this dataset.</p>
          <p className="text-sm text-slate-400 mt-1">{reason}</p>
        </div>
      </section>
    );
  }

  return (
    <section id="predictions-section" className="analysis-section space-y-6">
      {sup?.available ? <SupervisedSection ml={sup} /> : <Unavailable title="Driver model" reason={sup?.reason} />}
      {sup?.available && <WhatIfPanel ml={sup} />}
      {seg?.available && <SegmentsSection seg={seg} />}
      {an?.available && an.n_outliers > 0 && <AnomaliesSection an={an} />}
      {fc?.available && <ForecastSection fc={fc} />}
    </section>
  );
};

export default PredictionsTab;
