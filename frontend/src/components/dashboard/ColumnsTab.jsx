import React, { useState } from 'react';

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

const SORTABLE = [
  { key: 'name',          label: 'Column Name' },
  { key: 'dtype',         label: 'Data Type' },
  { key: 'missing_count', label: 'Missing' },
  { key: 'unique_count',  label: 'Unique' },
  { key: 'role',          label: 'Role' },
];

const ColumnsTab = ({ data }) => {
  const { dataset_profile } = data || {};
  const [searchTerm, setSearchTerm] = useState('');
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });

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
    if (!sortConfig.key) return 0;
    const av = a[sortConfig.key], bv = b[sortConfig.key];
    if (av < bv) return sortConfig.direction === 'asc' ? -1 : 1;
    if (av > bv) return sortConfig.direction === 'asc' ? 1 : -1;
    return 0;
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
          <div className="text-[11px] uppercase tracking-[0.32em] text-slate-400 mb-1">Schema</div>
          <h2 className="text-xl md:text-2xl font-semibold text-slate-100">Column profile</h2>
          <p className="text-sm text-slate-400 mt-1">Inspect data types, missing values and statistics across every column.</p>
        </div>
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

      {/* Stats summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MiniStat tone="blue"    icon="fa-table"        label="Total"        value={total} />
        <MiniStat tone="emerald" icon="fa-calculator"   label="Numeric"      value={numeric} />
        <MiniStat tone="purple"  icon="fa-tags"         label="Categorical"  value={categorical} />
        <MiniStat tone="amber"   icon="fa-fingerprint"  label="Identifiers"  value={identifiers} />
      </div>

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
              {sorted.map((col, index) => (
                <tr key={index}>
                  <td className="font-medium text-slate-100">{col.name}</td>
                  <td><span className="mono">{col.dtype}</span></td>
                  <td>
                    <div className="text-slate-200">{col.missing_count}</div>
                    {col.missing_count > 0 && (
                      <div className="text-xs text-slate-500">
                        {((col.missing_count / dataset_profile.n_rows) * 100).toFixed(1)}%
                      </div>
                    )}
                  </td>
                  <td className="text-slate-200">{col.unique_count}</td>
                  <td>
                    <span className={`neon-badge ${roleTone(col.role)}`}>{col.role}</span>
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
              ))}
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
