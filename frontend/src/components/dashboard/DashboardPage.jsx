import React, { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { useNavigate } from 'react-router-dom';
import { useDashboardStore } from '../../dashboardStore';
// The PDF/export stack (jsPDF + html-to-image, ~hundreds of KB) is loaded
// on-demand only when the user actually clicks Export — never in the
// dashboard's initial chunk.
import OverviewTab from './OverviewTab';
import InteractionBar from './InteractionBar';
import EDATab from './EDATab';
import VisualizationsTab from './VisualizationsTab';
import ColumnsTab from './ColumnsTab';
import DataQualityTab from './DataQualityTab';
import '../../styles/dashboard-futuristic.css';

const TABS = [
  { key: 'data_quality',      label: 'Data Check',   icon: 'fa-clipboard-check' },
  { key: 'overview',          label: 'Overview',     icon: 'fa-chart-line' },
  { key: 'eda',               label: 'AI Insights',  icon: 'fa-brain' },
  { key: 'visualizations',    label: 'Charts',       icon: 'fa-chart-bar' },
  { key: 'column_profiling',  label: 'Columns',      icon: 'fa-table' },
];

const isMonetary = (key) => {
  const k = key.toLowerCase();
  return ['amount','revenue','cost','expense','profit','fee','charge','payment','income','value']
    .some(t => k.includes(t));
};

const cleanDatasetName = (raw) => {
  if (!raw) return 'Untitled dataset';
  // strip url path / extension, normalise separators
  const last = String(raw).split(/[\\/]/).pop() || raw;
  const noExt = last.replace(/\.(csv|tsv|xlsx?|json)$/i, '');
  const spaced = noExt.replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim();
  if (!spaced) return 'Untitled dataset';
  return spaced.replace(/\b\w/g, (c) => c.toUpperCase());
};

const buildDescription = (data) => {
  if (!data) return '';
  const useCase = data?.eda_summary?.use_cases?.[0]?.description;
  if (useCase && typeof useCase === 'string' && useCase.length > 8) return useCase;
  const profile = data?.dataset_profile;
  if (!profile) return '';
  const rc = profile.role_counts || {};
  const parts = [];
  if (rc.numeric)     parts.push(`${rc.numeric} numeric`);
  if (rc.categorical) parts.push(`${rc.categorical} categorical`);
  if (rc.datetime)    parts.push(`${rc.datetime} datetime`);
  if (rc.text)        parts.push(`${rc.text} text`);
  if (rc.identifier)  parts.push(`${rc.identifier} identifier`);
  const breakdown = parts.length ? ` — ${parts.join(', ')} fields` : '';
  return `${(profile.n_rows ?? 0).toLocaleString()} rows × ${profile.n_cols ?? 0} columns${breakdown}.`;
};

const DashboardPage = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const { data: dashboardData, loading, error, refresh, lastUpdated } = useDashboardStore();
  const theme = useDashboardStore((s) => s.theme);
  const setExportHandler = useDashboardStore((s) => s.setExportHandler);
  const setExportProgress = useDashboardStore((s) => s.setExportProgress);
  const exporting = useDashboardStore((s) => s.exporting);
  const exportProgress = useDashboardStore((s) => s.exportProgress);
  const hasMounted = useRef(false);
  const navigate = useNavigate();
  const captureRef = useRef(null);
  const activeTabRef = useRef(activeTab);
  const particleRootRef = useRef(null);
  // Off-screen export: a tab the PDF capture drives WITHOUT changing the
  // visible `activeTab`, so the user keeps their view + full interactivity.
  const [exportTab, setExportTab] = useState(null);
  const exportSurfaceRef = useRef(null);

  useEffect(() => { activeTabRef.current = activeTab; }, [activeTab]);

  // S7.4: when the backend flags the dataset for schema review, land the user
  // on Data Quality first (only auto-switch from the initial Overview).
  const reviewedRef = useRef(false);
  useEffect(() => {
    const status = dashboardData?.dataset_profile?.data_quality?.report?.status;
    if (status && status !== 'ok' && !reviewedRef.current
        && activeTabRef.current === 'overview') {
      reviewedRef.current = true;
      setActiveTab('data_quality');
    }
  }, [dashboardData]);

  useEffect(() => {
    setExportHandler(async () => {
      try {
        const { exportDashboardToPDF } = await import('../../services/pdfExport');
        // Drive the OFF-SCREEN surface, never the visible tab — the user
        // keeps their current view and can keep clicking around while the
        // PDF builds in the background.
        await exportDashboardToPDF({
          setActiveTab: setExportTab,
          getCaptureEl: () => exportSurfaceRef.current,
          onProgress: setExportProgress,
        });
      } finally {
        setExportTab(null);
        setExportProgress(null);
      }
    });
    return () => setExportHandler(null);
  }, [setExportHandler, setExportProgress]);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    if (!hasMounted.current) { hasMounted.current = true; return; }
    // While a PDF export is cycling tabs, do NOT refetch on every switch —
    // the store also guards this, but skipping here avoids the wasted render.
    if (useDashboardStore.getState().exporting) return;
    refresh();
  }, [activeTab, refresh]);

  useEffect(() => {
    window.dispatchEvent(new Event('resize'));
  }, [activeTab, lastUpdated]);

  // Seed particles once
  useEffect(() => {
    const root = particleRootRef.current;
    if (!root || root.dataset.seeded === '1') return;
    root.dataset.seeded = '1';
    // Slimmed from 40 → 12: decorative only; many independently-animated DOM
    // nodes are pure compositor cost for negligible visual gain.
    const N = 12;
    for (let i = 0; i < N; i++) {
      const p = document.createElement('div');
      p.className = 'dash-particle';
      const size = Math.random() * 2.5 + 1;
      p.style.width = size + 'px';
      p.style.height = size + 'px';
      p.style.left = (Math.random() * 100) + '%';
      p.style.bottom = '-10px';
      p.style.animationDuration = (Math.random() * 18 + 16) + 's';
      p.style.animationDelay = (Math.random() * 12) + 's';
      p.style.opacity = (Math.random() * 0.45 + 0.15).toFixed(2);
      root.appendChild(p);
    }
  }, []);

  const renderTabComponent = (tabKey) => {
    switch (tabKey) {
      case 'overview':
        return <OverviewTab data={dashboardData} loading={loading} error={error} refreshKey={lastUpdated} />;
      case 'eda':
        return <EDATab data={dashboardData} loading={loading} error={error} />;
      case 'visualizations':
        return <VisualizationsTab data={dashboardData} loading={loading} error={error} refreshKey={lastUpdated} />;
      case 'column_profiling':
        return <ColumnsTab data={dashboardData} />;
      case 'data_quality':
        return (
          <DataQualityTab
            data={dashboardData}
            onGoToColumns={() => setActiveTab('column_profiling')}
          />
        );
      default:
        return <OverviewTab data={dashboardData} loading={loading} error={error} refreshKey={lastUpdated} />;
    }
  };

  const renderTabContent = () => {
    if (loading) {
      return (
        <div className="flex justify-center items-center py-20">
          <div className="dash-spinner" />
        </div>
      );
    }
    if (error) {
      return (
        <div className="glass-soft p-5 flex items-center gap-3 text-rose-200 border-rose-400/30">
          <i className="fas fa-triangle-exclamation text-rose-300 text-lg" />
          <span>{error}</span>
        </div>
      );
    }
    if (!dashboardData) {
      return (
        <div className="empty-state">
          <i className="fas fa-chart-bar empty-icon" />
          <h3 className="text-lg font-semibold text-slate-100 mb-1">No Dashboard Data Available</h3>
          <p className="text-sm">Please upload a dataset to generate insights.</p>
        </div>
      );
    }

    return renderTabComponent(activeTab);
  };

  const profile = dashboardData?.dataset_profile;
  const hasData = !!dashboardData?.original_filename;

  return (
    <div className={`dash-shell${theme === 'light' ? ' theme-light' : ''}`}>
      {/* Full-screen export overlay — portaled to <body> so (a) it is never
          inside the captured element, and (b) the .dash-shell export-mode
          CSS (which kills animations) can't freeze its spinner. It hides the
          tab-cycling that the capture loop performs underneath. */}
      {/* Non-blocking progress pill. pointer-events:none so it never steals a
          click — the dashboard underneath stays fully interactive while the
          PDF builds. Portaled to <body> so it is outside any captured node. */}
      {exporting && createPortal(
        <div
          role="status"
          aria-live="polite"
          style={{
            position: 'fixed',
            right: '1.25rem',
            bottom: '1.25rem',
            zIndex: 200,
            display: 'flex',
            alignItems: 'center',
            gap: '0.6rem',
            padding: '0.7rem 1rem',
            borderRadius: '0.75rem',
            pointerEvents: 'none',
            background: theme === 'light' ? 'rgba(255,255,255,0.96)' : 'rgba(15,23,42,0.95)',
            color: theme === 'light' ? '#0f172a' : '#e2e8f0',
            boxShadow: '0 10px 30px rgba(0,0,0,0.35)',
            border: '1px solid rgba(148,163,184,0.25)',
          }}
        >
          <i className="fas fa-circle-notch fa-spin" style={{ color: '#60a5fa' }} />
          <div style={{ fontSize: '0.85rem', fontWeight: 600 }}>
            Exporting PDF…
            <span style={{ fontWeight: 400, opacity: 0.75 }}>
              {exportProgress
                ? ` ${exportProgress.label}${exportProgress.total ? ` — ${exportProgress.index}/${exportProgress.total}` : ''}`
                : ' Preparing…'}
            </span>
          </div>
        </div>,
        document.body,
      )}
      {/* Off-screen capture surface. Rendered fully laid out (NOT display:none —
          Plotly needs real dimensions to size + paint) but parked far
          off-screen, so the capture loop drives THIS, never the visible
          dashboard. It carries .dash-shell.dash-export-mode so the PDF
          export CSS (solid fills, no backdrop-filter) applies to its
          subtree only. */}
      {exporting && dashboardData && createPortal(
        <div
          id="pdf-export-surface"
          ref={exportSurfaceRef}
          className="dash-shell dash-export-mode"
          aria-hidden="true"
          style={{
            position: 'fixed',
            left: '-100000px',
            top: 0,
            width: '1280px',
            pointerEvents: 'none',
          }}
        >
          <div className="glass-card p-6 md:p-7">
            {exportTab ? renderTabComponent(exportTab) : null}
          </div>
        </div>,
        document.body,
      )}
      {/* Aurora gradient mesh */}
      <div className="dash-aurora dash-aurora-a" />
      <div className="dash-aurora dash-aurora-b" />
      <div className="dash-aurora dash-aurora-c" />
      <div className="dash-aurora dash-aurora-d" />
      <div className="dash-grid" />
      <div ref={particleRootRef} className="pointer-events-none absolute inset-0" />

      <div ref={captureRef} className="relative z-10 w-full mx-auto px-4 sm:px-6 lg:px-10 py-8 max-w-[1920px]">

        {/* Page header — dataset name + AI-generated one-liner */}
        <div className="mb-8 flex flex-col md:flex-row md:items-end md:justify-between gap-4">
          <div className="min-w-0">
            <div className="text-[11px] uppercase tracking-[0.35em] text-slate-400 mb-2 flex items-center gap-2">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-emerald-400 shadow-[0_0_8px_#34d399]" />
              {hasData ? 'Active dataset' : 'Mission Control'}
            </div>
            <h1 className="text-3xl md:text-4xl font-bold leading-[1.2] tracking-tight truncate pb-1" title={dashboardData?.original_filename || ''}>
              <span className="bg-gradient-to-r from-sky-300 via-fuchsia-300 to-amber-200 bg-clip-text text-transparent">
                {hasData ? cleanDatasetName(dashboardData.original_filename) : 'Dataset Intelligence'}
              </span>
            </h1>
            <p className="mt-2 text-sm text-slate-400 max-w-2xl">
              {hasData
                ? buildDescription(dashboardData) || 'A live, AI-generated view across the entire dataset.'
                : 'A live, AI-generated view across the entire dataset — statistics, distributions, correlations and patterns.'}
            </p>
            {hasData && (
              <p className="mt-1 text-xs text-slate-500 truncate" title={dashboardData.original_filename}>
                <i className="fas fa-file-csv mr-1.5 text-slate-500" />
                {dashboardData.original_filename}
              </p>
            )}
          </div>

          {hasData && (
            <div className="flex items-center gap-2 flex-wrap flex-shrink-0">
              <span className="metric-chip">
                <i className="fas fa-database text-sky-300" />
                {profile?.n_rows?.toLocaleString() || 0} rows
              </span>
              <span className="metric-chip">
                <i className="fas fa-grip-vertical text-fuchsia-300" />
                {profile?.n_cols || 0} columns
              </span>
              {lastUpdated && (
                <span className="metric-chip">
                  <i className="fas fa-bolt text-amber-300" />
                  Updated {new Date(lastUpdated).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              )}
            </div>
          )}
        </div>

        {/* Critical Totals */}
        {dashboardData?.critical_totals && Object.keys(dashboardData.critical_totals).length > 0 && (
          <div className="glass-card p-6 mb-7 dash-section-enter">
            <div className="flex items-center justify-between mb-5">
              <div>
                <span className="section-eyebrow text-[11px] tracking-[0.32em] text-slate-400 uppercase block mb-1">
                  Headline metrics
                </span>
                <h2 className="text-xl font-semibold text-slate-100 flex items-center gap-3">
                  <span className="section-icon"><i className="fas fa-coins text-amber-300" /></span>
                  Key Financial Metrics
                </h2>
              </div>
              <span className="neon-badge neon-amber">Pre-sampling</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {Object.entries(dashboardData.critical_totals).map(([key, value]) => (
                <div key={key} className="kpi-tile">
                  <div className="kpi-label">{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</div>
                  <div className={`kpi-value ${isMonetary(key) ? 'text-emerald-300' : 'text-sky-300'}`}>
                    {isMonetary(key)
                      ? `$${typeof value === 'number' ? value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : value}`
                      : typeof value === 'number'
                          ? value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
                          : value}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tabs */}
        <div className="dash-tabs mb-6 dash-section-enter">
          {TABS.map(t => (
            <button
              key={t.key}
              className={`dash-tab ${activeTab === t.key ? 'is-active' : ''}`}
              onClick={() => setActiveTab(t.key)}
            >
              <i className={`fas ${t.icon}`} />
              <span>{t.label}</span>
              <span className="tab-dot" />
            </button>
          ))}
        </div>

        {/* Active cross-highlight / filter chips (S14.3) — renders only when
            an interaction is set; portal-free so it sits above the tab body. */}
        {dashboardData && <InteractionBar />}

        {/* Tab content */}
        <div key={activeTab} className="glass-card p-6 md:p-7 dash-section-enter">
          {renderTabContent()}
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;
