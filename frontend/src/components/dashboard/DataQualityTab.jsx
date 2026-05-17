import React from 'react';
import { useDashboardStore } from '../../dashboardStore';

// Phase 7 (S7.4): surfaces the backend DataQualityReport — cleaning manifest,
// invariant vetoes/flags, PII verdict and the review status. The PII consent
// opt-in (AI Insights gate) lives here too — the dashboard itself always
// builds; only AI egress needs the user's explicit consent.

const STATUS_TONE = {
  ok: { c: 'neon-emerald', i: 'fa-circle-check', t: 'Looks good' },
  review: { c: 'neon-amber', i: 'fa-user-pen', t: 'Needs your review' },
};

const FLAG_LABEL = {
  total_vs_components: 'double-count risk',
  share_sum: 'percentages',
  std_gg_mean: 'wide spread',
};

const fmtShape = (shape) => {
  if (!Array.isArray(shape) || shape.length < 2) return '—';
  const [rows, cols] = shape;
  return `${Number(rows).toLocaleString()} rows × ${Number(cols).toLocaleString()} columns`;
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
          <p>No data check is available for this dataset yet.</p>
        </div>
      </section>
    );
  }

  const tone = STATUS_TONE[report.status] || STATUS_TONE.review;
  const cleaning = report.cleaning || dq.cleaning || {};
  const vetoes = report.vetoes || dq.vetoes || [];
  const flags = report.flags || dq.flags || [];
  const piiCols = report.pii_columns || dq.pii_columns || {};

  const piiPresent = !!(report.pii_present ?? report.pii_blocked);
  const aiConsent = !!(report.ai_consent ?? dq.ai_consent);
  const consentNeeded = piiPresent && !aiConsent;
  const piiColNames = Object.keys(piiCols);

  const grantAiConsent = useDashboardStore((s) => s.grantAiConsent);
  const aiConsentSubmitting = useDashboardStore((s) => s.aiConsentSubmitting);
  const aiConsentError = useDashboardStore((s) => s.aiConsentError);

  return (
    <section className="analysis-section space-y-6">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <div className="text-[11px] uppercase tracking-[0.32em] text-slate-400 mb-1">
            Data Check
          </div>
          <h2 className="text-xl md:text-2xl font-semibold text-slate-100">
            Data check &amp; cleanup report
          </h2>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {piiPresent && (
            <span className={`neon-badge ${aiConsent ? 'neon-emerald' : 'neon-rose'} text-sm`}>
              <i className={`fas ${aiConsent ? 'fa-robot' : 'fa-user-shield'} mr-1.5`} />
              {aiConsent ? 'AI enabled for this data' : 'Personal data — AI needs your OK'}
            </span>
          )}
          <span className={`neon-badge ${tone.c} text-sm`}>
            <i className={`fas ${tone.i} mr-1.5`} /> {tone.t}
          </span>
        </div>
      </div>

      {report.reasons?.length > 0 && (
        <div className="glass-soft p-4 border-amber-400/30 text-amber-100 text-sm space-y-1">
          {report.reasons.map((r, i) => (
            <div key={i}><i className="fas fa-circle-info mr-2" />{r}</div>
          ))}
        </div>
      )}

      {/* PII consent gate — the dashboard already built with no data leaving
          the server; only AI Insights needs the user's explicit go-ahead. */}
      {piiPresent && (
        <div className={`glass-soft p-4 space-y-3 ${aiConsent ? 'border-emerald-400/30' : 'border-rose-400/40'}`}>
          {aiConsent ? (
            <div className="text-sm text-emerald-100">
              <i className="fas fa-circle-check mr-2 text-emerald-300" />
              You allowed AI to use this dataset. AI Insights are generated from
              your data, including the personal columns.
            </div>
          ) : (
            <>
              <div className="text-sm text-slate-200">
                <i className="fas fa-user-shield mr-2 text-rose-300" />
                We spotted what looks like <b>personal data</b> in{' '}
                <span className="text-rose-200 font-semibold">
                  {piiColNames.length ? piiColNames.join(', ') : 'one or more columns'}
                </span>
                {Object.values(piiCols).some((e) => e?.length) && (
                  <span className="text-slate-400">
                    {' '}({Object.entries(piiCols)
                      .map(([c, e]) => `${c}: ${(e || []).join('/')}`)
                      .join('; ')})
                  </span>
                )}
                .
              </div>
              <div className="text-sm text-slate-300">
                Your dashboard — charts, metrics and the data check — is{' '}
                <b>already built</b> and never left this server. Only the{' '}
                <b>AI Insights</b> tab sends data to an external AI service.
                That’s your call: if you turn it on, the full dataset
                (including the columns above) is sent for analysis. Sharing
                your own data is your decision and responsibility.
              </div>
              {aiConsentError && (
                <div className="text-sm text-rose-200">
                  <i className="fas fa-triangle-exclamation mr-2" />{aiConsentError}
                </div>
              )}
              <div className="flex flex-wrap gap-3">
                <button
                  type="button"
                  disabled={aiConsentSubmitting}
                  onClick={() => { grantAiConsent().catch(() => {}); }}
                  className="inline-flex items-center gap-2 rounded-lg text-white text-sm font-semibold px-5 py-2.5 whitespace-nowrap disabled:opacity-50"
                  style={{
                    background: aiConsentSubmitting ? '#9f1239' : '#e11d48',
                    border: '1px solid rgba(253,164,175,0.6)',
                    boxShadow: '0 4px 14px rgba(225,29,72,0.35)',
                    cursor: aiConsentSubmitting ? 'not-allowed' : 'pointer',
                  }}
                >
                  <i className={`fas ${aiConsentSubmitting ? 'fa-spinner fa-spin' : 'fa-robot'}`} />
                  {aiConsentSubmitting ? 'Enabling AI…' : 'Yes, use AI on this data'}
                </button>
                <span className="text-xs text-slate-400 self-center">
                  Prefer not to? Just leave it — everything else works without AI.
                </span>
              </div>
            </>
          )}
        </div>
      )}

      {/* Close the loop: tell the user WHERE to act and take them there. */}
      <div className="glass-soft p-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-sky-400/30">
        <div className="text-sm text-slate-200">
          <i className="fas fa-circle-arrow-right text-sky-300 mr-2" />
          {report.status === 'ok'
            ? 'Column types look right. You can still review and adjust them in the Columns tab.'
            : 'Some column types need a quick check. Open the Columns tab to fix any of them and confirm.'}
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
          Review &amp; fix column types
        </button>
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        <Card title="Cleanup applied" icon="fa-broom">
          <ul className="text-sm text-slate-300 space-y-1">
            <li>Size before cleanup: {fmtShape(cleaning.original_shape)}</li>
            <li>Size after cleanup: {fmtShape(cleaning.cleaned_shape)}</li>
            <li>Empty rows removed: {cleaning.dropped_null_rows ?? 0}</li>
            <li>Reformatted to numbers: {Object.keys(cleaning.coerced_numeric || {}).join(', ') || '—'}</li>
            <li>Placeholder values cleared: {Object.keys(cleaning.sentinels_nulled || {}).join(', ') || '—'}</li>
          </ul>
        </Card>

        <Card title="Privacy" icon="fa-user-shield">
          <div className="text-sm text-slate-300 space-y-1">
            <div>Privacy level:{' '}
              <b>{report.sensitivity === 'sensitive'
                ? 'contains personal data'
                : 'no personal data found'}</b>
            </div>
            <div>AI Insights on this data:{' '}
              <b>{!piiPresent
                ? 'allowed (no personal data)'
                : aiConsent
                  ? 'allowed (you consented)'
                  : 'off until you allow it'}</b>
            </div>
            <div>
              Personal-data columns:{' '}
              {Object.keys(piiCols).length
                ? Object.entries(piiCols).map(([c, e]) => `${c} (${e.join('/')})`).join(', ')
                : '—'}
            </div>
            <div>Type-detection accuracy: {Math.round((report.mean_confidence || 0) * 100)}%</div>
          </div>
        </Card>
      </div>

      {vetoes.length > 0 && (
        <Card title="Auto-corrected column types" icon="fa-gavel">
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
        <Card title="Things to check" icon="fa-flag">
          <ul className="text-sm text-slate-300 space-y-1">
            {flags.map((f, i) => (
              <li key={i}>
                <span className="neon-badge neon-amber mr-2">{FLAG_LABEL[f.type] || f.type}</span>
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
