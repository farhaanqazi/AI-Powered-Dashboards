import React, { useState } from 'react';
import { useDashboardStore } from '../../dashboardStore';

const EDITABLE_ROLES = [
  'numeric', 'categorical', 'datetime', 'identifier',
  'text', 'boolean', 'year', 'ratio',
];

const roleTone = (role) => {
  switch (role) {
    case 'numeric':     return 'neon-blue';
    case 'categorical': return 'neon-purple';
    case 'datetime':    return 'neon-emerald';
    case 'identifier':  return 'neon-rose';
    case 'text':        return 'neon-amber';
    default:            return 'neon-cyan';
  }
};

// Mirrors backend AUTO_ACCEPT_CONFIDENCE — used only as a display hint to
// decide which rows to flag for the user; not a pipeline threshold.
const LOW_CONFIDENCE = 0.7;

const SORTABLE = [
  { key: 'name',          label: 'Column Name' },
  { key: 'dtype',         label: 'Data Type' },
  { key: 'null_count',    label: 'Empty cells' },
  { key: 'unique_count',  label: 'Unique' },
  { key: 'role',          label: 'Type' },
];

const ColumnsTab = ({ data }) => {
  const { dataset_profile } = data || {};
  const [searchTerm, setSearchTerm] = useState('');
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });
  const [edits, setEdits] = useState({});           // colName -> new role
  const [acknowledged, setAcknowledged] = useState(false);
  const [editing, setEditing] = useState(false);    // explicit edit mode

  const submitSchemaReview = useDashboardStore((s) => s.submitSchemaReview);
  const reviewSubmitting = useDashboardStore((s) => s.reviewSubmitting);
  const reviewError = useDashboardStore((s) => s.reviewError);

  const contract = dataset_profile?.contract;
  const dq = dataset_profile?.data_quality;
  const report = dq?.report;
  const needsReview = !!report && report.status && report.status !== 'ok';

  // Which specific columns the user should actually look at: any column the
  // critic vetoed/flagged, plus any classified with low certainty. This is
  // the same evidence the backend used for the "needs review" verdict — we
  // just make it visible per row instead of leaving it as "some columns".
  const flagInfo = React.useMemo(() => {
    const m = new Map();   // name -> reason
    const add = (n, why) => { if (n && !m.has(n)) m.set(n, why); };
    (report?.vetoes || dq?.vetoes || []).forEach((v) =>
      add(v?.column, 'Type was auto-corrected — please confirm it’s right.'));
    (report?.flags || dq?.flags || []).forEach((f) =>
      (f?.columns || []).forEach((c) => add(c, f?.detail || 'Flagged for a quick check.')));
    (dataset_profile.columns || []).forEach((c) => {
      if (typeof c.confidence === 'number' && c.confidence > 0 && c.confidence < LOW_CONFIDENCE)
        add(c.name, `Type detected with low certainty (${Math.round(c.confidence * 100)}%).`);
    });
    return m;
  }, [report, dq, dataset_profile]);
  const flagSet = flagInfo;
  const flaggedNames = [...flagInfo.keys()];
  const reasons = report?.reasons || [];
  // Read-only by default; the editor appears only when the dataset needs
  // review OR the user explicitly enters edit mode (no silent regression of
  // the clean profile view).
  const canEdit = !!contract;
  const editable = canEdit && (needsReview || editing);

  const setRole = (name, role) =>
    setEdits((e) => ({ ...e, [name]: role }));

  const confirmReview = async () => {
    const overrides = Object.entries(edits)
      .filter(([, role]) => role)
      .map(([name, role]) => ({ name, role }));
    try {
      await submitSchemaReview(overrides);
      setEdits({});
      setAcknowledged(true);
      setEditing(false);
    } catch { /* reviewError is surfaced from the store */ }
  };

  if (!dataset_profile || !dataset_profile.columns) {
    return (
      <section id="column_profiling-section" className="analysis-section">
        <div className="empty-state">
          <i className="fas fa-table empty-icon" />
          <p>No column profiling data available.</p>
        </div>
      </section>
    );
  }

  const filtered = dataset_profile.columns.filter(c =>
    c.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    c.role.toLowerCase().includes(searchTerm.toLowerCase()) ||
    c.dtype.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const sorted = [...filtered].sort((a, b) => {
    if (sortConfig.key) {
      const av = a[sortConfig.key], bv = b[sortConfig.key];
      if (av < bv) return sortConfig.direction === 'asc' ? -1 : 1;
      if (av > bv) return sortConfig.direction === 'asc' ? 1 : -1;
      return 0;
    }
    // Default order: columns that need a look come first.
    const fa = flagSet.has(a.name) ? 0 : 1;
    const fb = flagSet.has(b.name) ? 0 : 1;
    return fa - fb;
  });

  const handleSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') direction = 'desc';
    setSortConfig({ key, direction });
  };

  const total = dataset_profile.columns.length;
  const numeric = dataset_profile.columns.filter(c => c.role === 'numeric').length;
  const categorical = dataset_profile.columns.filter(c => c.role === 'categorical').length;
  const identifiers = dataset_profile.columns.filter(c => c.role === 'identifier').length;

  return (
    <section id="column_profiling-section" className="analysis-section space-y-6">
      <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
        <div>
          <div className="text-[11px] uppercase tracking-[0.32em] text-slate-400 mb-1">Column types</div>
          <h2 className="text-xl md:text-2xl font-semibold text-slate-100">Column profile</h2>
          <p className="text-sm text-slate-400 mt-1">
            Inspect data types, missing values and statistics across every column.
            {canEdit && !editable && (
              <span className="block mt-1 text-sky-300/90">
                <i className="fas fa-circle-info mr-1.5" />
                Wrong type on a column? Use “Edit column types” to fix it and confirm.
              </span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-3 w-full md:w-auto">
          {canEdit && !needsReview && (
            <button
              type="button"
              onClick={() => setEditing((v) => !v)}
              className="inline-flex items-center gap-2 rounded-lg border border-sky-400/40 bg-sky-500/10 hover:bg-sky-500/20 text-sky-200 text-sm px-3 py-2 whitespace-nowrap transition-colors"
            >
              <i className={`fas ${editing ? 'fa-xmark' : 'fa-pen'}`} />
              {editing ? 'Cancel' : 'Edit column types'}
            </button>
          )}
          <div className="relative w-full md:w-72">
            <i className="fas fa-search absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 text-sm" />
            <input
              type="text"
              placeholder="Search columns..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="dash-input"
            />
          </div>
        </div>
      </div>

      {/* Stats summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MiniStat tone="blue"    icon="fa-table"        label="Total"        value={total} />
        <MiniStat tone="emerald" icon="fa-calculator"   label="Numeric"      value={numeric} />
        <MiniStat tone="purple"  icon="fa-tags"         label="Categorical"  value={categorical} />
        <MiniStat tone="amber"   icon="fa-fingerprint"  label="Identifiers"  value={identifiers} />
      </div>

      {/* S7.3: non-skippable schema-review confirm */}
      {editable && (
        <div className={`glass-soft p-4 flex flex-col md:flex-row md:items-center md:justify-between gap-3 ${needsReview ? 'border-amber-400/40' : 'border-emerald-400/30'}`}>
          <div className="text-sm text-slate-200 space-y-1.5">
            <div>
              <i className={`fas ${needsReview ? 'fa-user-pen text-amber-300' : 'fa-circle-check text-emerald-300'} mr-2`} />
              {needsReview
                ? (flaggedNames.length
                    ? <>Check {flaggedNames.length === 1 ? 'this column' : `these ${flaggedNames.length} columns`} (highlighted below, listed first): <span className="font-semibold text-amber-200">{flaggedNames.join(', ')}</span>. Fix the Type if it’s wrong, then confirm — this step can’t be skipped.</>
                    : <>Review the column types below and correct any that look wrong, then confirm — this step can’t be skipped.</>)
                : 'Column types look right. You can still adjust them and re-save.'}
              {Object.keys(edits).length > 0 && (
                <span className="ml-2 neon-badge neon-amber">{Object.keys(edits).length} change{Object.keys(edits).length > 1 ? 's' : ''}</span>
              )}
            </div>
            {needsReview && reasons.length > 0 && (
              <ul className="text-xs text-slate-400 list-disc list-inside">
                {reasons.map((r, i) => <li key={i}>{r}</li>)}
              </ul>
            )}
          </div>
          <button
            type="button"
            className="inline-flex items-center gap-2 rounded-lg text-white text-sm font-semibold px-5 py-2.5 whitespace-nowrap disabled:opacity-50 shadow-lg"
            style={{
              background: reviewSubmitting ? '#0369a1' : '#0ea5e9',
              border: '1px solid rgba(125,211,252,0.6)',
              boxShadow: '0 4px 14px rgba(14,165,233,0.35)',
              cursor: reviewSubmitting ? 'not-allowed' : 'pointer',
            }}
            disabled={reviewSubmitting}
            onClick={confirmReview}
          >
            <i className="fas fa-check" />
            {reviewSubmitting ? 'Saving…'
              : Object.keys(edits).length ? 'Apply & confirm types'
              : 'Confirm column types'}
          </button>
        </div>
      )}
      {reviewError && (
        <div className="glass-soft p-3 text-rose-200 border-rose-400/30 text-sm">
          <i className="fas fa-triangle-exclamation mr-2" />{reviewError}
        </div>
      )}
      {acknowledged && !needsReview && (
        <div className="glass-soft p-3 text-emerald-200 border-emerald-400/30 text-sm">
          <i className="fas fa-circle-check mr-2" />Column types confirmed and saved.
        </div>
      )}

      {/* Table */}
      <div className="glass-card p-0 overflow-hidden">
        <div className="dash-scroll overflow-x-auto">
          <table className="dash-table">
            <thead>
              <tr>
                {SORTABLE.map(col => (
                  <th key={col.key} onClick={() => handleSort(col.key)}>
                    <div className="flex items-center gap-2">
                      {col.label}
                      {sortConfig.key === col.key && (
                        <i className={`fas fa-${sortConfig.direction === 'asc' ? 'arrow-up' : 'arrow-down'} text-sky-300 text-[10px]`} />
                      )}
                    </div>
                  </th>
                ))}
                <th>Min</th>
                <th>Max</th>
                <th>Mean</th>
                <th>Top Categories</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((col, index) => {
                const flagged = flagSet.has(col.name);
                return (
                <tr
                  key={index}
                  style={flagged ? {
                    background: 'rgba(245,158,11,0.08)',
                    boxShadow: 'inset 3px 0 0 #f59e0b',
                  } : undefined}
                >
                  <td className="font-medium text-slate-100">
                    {col.name}
                    {flagged && (
                      <span
                        className="ml-2 neon-badge neon-amber align-middle"
                        title={flagInfo.get(col.name)}
                      >
                        <i className="fas fa-circle-exclamation mr-1" />Check
                      </span>
                    )}
                  </td>
                  <td><span className="mono">{col.dtype}</span></td>
                  <td>
                    <div className="text-slate-200">{col.null_count ?? 0}</div>
                    {col.null_count > 0 && dataset_profile.n_rows > 0 && (
                      <div className="text-xs text-slate-500">
                        {((col.null_count / dataset_profile.n_rows) * 100).toFixed(1)}%
                      </div>
                    )}
                  </td>
                  <td className="text-slate-200">{col.unique_count}</td>
                  <td>
                    {editable ? (
                      <select
                        className="text-xs rounded-md"
                        style={{
                          background: '#1e293b',
                          color: '#f1f5f9',
                          border: '1px solid rgba(148,163,184,0.35)',
                          padding: '0.35rem 0.5rem',
                          minWidth: '128px',
                          cursor: 'pointer',
                        }}
                        value={edits[col.name] ?? col.role}
                        onChange={(e) => setRole(col.name, e.target.value)}
                      >
                        {/* keep the current role selectable even if it's not
                            in the standard editable set (e.g. 'unknown') */}
                        {(EDITABLE_ROLES.includes(col.role)
                          ? EDITABLE_ROLES
                          : [col.role, ...EDITABLE_ROLES]
                        ).map((r) => (
                          <option key={r} value={r} style={{ background: '#1e293b', color: '#f1f5f9' }}>
                            {r}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <span className={`neon-badge ${roleTone(col.role)}`}>{col.role}</span>
                    )}
                  </td>
                  <td className="text-slate-300 tabular-nums">{col.stats?.min !== undefined ? col.stats.min : '—'}</td>
                  <td className="text-slate-300 tabular-nums">{col.stats?.max !== undefined ? col.stats.max : '—'}</td>
                  <td className="text-slate-300 tabular-nums">{col.stats?.mean !== undefined ? col.stats.mean.toFixed(2) : '—'}</td>
                  <td>
                    <div className="flex flex-wrap gap-1.5">
                      {col.top_categories && col.top_categories.length > 0 ? (
                        col.top_categories.slice(0, 3).map((cat, ci) => (
                          <span key={ci} className="metric-chip max-w-[10rem] truncate" title={String(cat.value)}>
                            {String(cat.value).substring(0, 15)}{String(cat.value).length > 15 ? '…' : ''} ({cat.count})
                          </span>
                        ))
                      ) : (
                        <span className="text-xs text-slate-500">—</span>
                      )}
                    </div>
                  </td>
                </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {filtered.length === 0 && (
        <div className="empty-state">
          <i className="fas fa-search empty-icon" />
          <p>No columns match your search criteria.</p>
        </div>
      )}
    </section>
  );
};

const MiniStat = ({ tone, icon, label, value }) => (
  <div className={`stat-tile stat-${tone}`}>
    <div className="stat-glow" />
    <div className="stat-icon"><i className={`fas ${icon}`} /></div>
    <div className="stat-label">{label}</div>
    <div className="stat-value">{value}</div>
  </div>
);

export default ColumnsTab;
