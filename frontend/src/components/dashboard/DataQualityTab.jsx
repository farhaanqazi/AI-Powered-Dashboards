import React from 'react';

// Phase 7 (S7.4): surfaces the backend DataQualityReport — cleaning manifest,
// invariant vetoes/flags, PII verdict and the review status. Read-only; the
// editable role corrections live in the Columns tab (S7.3).

const STATUS_TONE = {
  ok: { c: 'neon-emerald', i: 'fa-circle-check', t: 'Auto-accepted' },
  review: { c: 'neon-amber', i: 'fa-user-pen', t: 'Needs schema review' },
  blocked: { c: 'neon-rose', i: 'fa-shield-halved', t: 'PII-blocked' },
};

const Card = ({ title, icon, children }) => (
  <div className="glass-card p-5 space-y-3">
    <div className="flex items-center gap-2 text-slate-200 font-semibold">
      <i className={`fas ${icon} text-sky-300`} /> {title}
    </div>
    {children}
  </div>
);

const DataQualityTab = ({ data, onGoToColumns }) => {
  const dq = data?.dataset_profile?.data_quality || {};
  const report = dq.report;

  if (!report) {
    return (
      <section className="analysis-section">
        <div className="empty-state">
          <i className="fas fa-clipboard-check empty-icon" />
          <p>No data-quality report for this dataset.</p>
        </div>
      </section>
    );
  }

  const tone = STATUS_TONE[report.status] || STATUS_TONE.review;
  const cleaning = report.cleaning || dq.cleaning || {};
  const vetoes = report.vetoes || dq.vetoes || [];
  const flags = report.flags || dq.flags || [];
  const piiCols = report.pii_columns || dq.pii_columns || {};

  return (
    <section className="analysis-section space-y-6">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <div className="text-[11px] uppercase tracking-[0.32em] text-slate-400 mb-1">
            Data Quality
          </div>
          <h2 className="text-xl md:text-2xl font-semibold text-slate-100">
            Contract &amp; quality report
          </h2>
        </div>
        <span className={`neon-badge ${tone.c} text-sm`}>
          <i className={`fas ${tone.i} mr-1.5`} /> {tone.t}
        </span>
      </div>

      {report.reasons?.length > 0 && (
        <div className="glass-soft p-4 border-amber-400/30 text-amber-100 text-sm space-y-1">
          {report.reasons.map((r, i) => (
            <div key={i}><i className="fas fa-circle-info mr-2" />{r}</div>
          ))}
        </div>
      )}

      {/* Close the loop: tell the user WHERE to act and take them there. */}
      <div className="glass-soft p-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-sky-400/30">
        <div className="text-sm text-slate-200">
          <i className="fas fa-circle-arrow-right text-sky-300 mr-2" />
          {report.status === 'ok'
            ? 'Column types look right. You can still review and adjust them in the Columns tab.'
            : 'The column types need a human check. Open the Columns tab to correct any roles and confirm the schema.'}
        </div>
        <button
          type="button"
          onClick={onGoToColumns}
          className="inline-flex items-center gap-2 rounded-lg text-white text-sm font-semibold px-5 py-2.5 whitespace-nowrap"
          style={{
            background: '#0ea5e9',
            border: '1px solid rgba(125,211,252,0.6)',
            boxShadow: '0 4px 14px rgba(14,165,233,0.35)',
            cursor: 'pointer',
          }}
        >
          <i className="fas fa-table-columns" />
          Review &amp; correct column roles
        </button>
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        <Card title="Cleaning applied" icon="fa-broom">
          <ul className="text-sm text-slate-300 space-y-1">
            <li>Original shape: {JSON.stringify(cleaning.original_shape || [])}</li>
            <li>Cleaned shape: {JSON.stringify(cleaning.cleaned_shape || [])}</li>
            <li>Null rows dropped: {cleaning.dropped_null_rows ?? 0}</li>
            <li>Numeric-coerced: {Object.keys(cleaning.coerced_numeric || {}).join(', ') || '—'}</li>
            <li>Sentinels nulled: {Object.keys(cleaning.sentinels_nulled || {}).join(', ') || '—'}</li>
          </ul>
        </Card>

        <Card title="Sensitivity" icon="fa-user-shield">
          <div className="text-sm text-slate-300 space-y-1">
            <div>Sensitivity: <b>{report.sensitivity}</b></div>
            <div>PII blocked: <b>{String(report.pii_blocked)}</b></div>
            <div>
              PII columns:{' '}
              {Object.keys(piiCols).length
                ? Object.entries(piiCols).map(([c, e]) => `${c} (${e.join('/')})`).join(', ')
                : '—'}
            </div>
            <div>Mean confidence: {report.mean_confidence}</div>
          </div>
        </Card>
      </div>

      {vetoes.length > 0 && (
        <Card title="Invariant vetoes (auto-corrected roles)" icon="fa-gavel">
          <ul className="text-sm text-slate-300 space-y-1">
            {vetoes.map((v, i) => (
              <li key={i}>
                <b>{v.column}</b>: {v.from_role} → {v.to_role} — {v.reason}
              </li>
            ))}
          </ul>
        </Card>
      )}

      {flags.length > 0 && (
        <Card title="Quality flags" icon="fa-flag">
          <ul className="text-sm text-slate-300 space-y-1">
            {flags.map((f, i) => (
              <li key={i}>
                <span className="neon-badge neon-amber mr-2">{f.type}</span>
                {f.detail}
              </li>
            ))}
          </ul>
        </Card>
      )}
    </section>
  );
};

export default DataQualityTab;
