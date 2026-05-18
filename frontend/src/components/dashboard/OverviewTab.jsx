import React, { useState } from 'react';
import ChartRenderer from '../charts/ChartRenderer';
import ChartModal from '../charts/ChartModal';

// Phase 13 S13.2 — dashboard composition. Charts are grouped into themed,
// navigable sections instead of one flat dump. `section` is tagged by the
// backend (S13.1); the type-fallback keeps older persisted payloads working.
const SECTION_ORDER = ['Breakdowns', 'Distributions', 'Trends', 'Relationships', 'Other'];
const SECTION_META = {
  Breakdowns:    { icon: 'fa-chart-bar',   color: '#93c5fd' },
  Distributions: { icon: 'fa-chart-area',  color: '#6ee7b7' },
  Trends:        { icon: 'fa-chart-line',  color: '#fcd34d' },
  Relationships: { icon: 'fa-link',        color: '#d8b4fe' },
  Other:         { icon: 'fa-th-large',    color: '#67e8f9' },
};
const TYPE_FALLBACK = {
  histogram: 'Distributions', box: 'Distributions',
  bar: 'Breakdowns', pie: 'Breakdowns',
  line: 'Trends', scatter: 'Relationships', heatmap: 'Relationships',
};
const sectionOf = (c) => c.section || TYPE_FALLBACK[c && c.type] || 'Other';
const PER_SECTION_CAP = 4;

const OverviewTab = ({ data, loading, error, refreshKey }) => {
  const safeData = data || {};
  const { kpis, primary_chart, category_charts, all_charts, eda_summary } = safeData;
  const aiNarrative = eda_summary && eda_summary.ai_narrative;
  const [modalChart, setModalChart] = useState(null);
  const [activeSection, setActiveSection] = useState(null);
  const [expanded, setExpanded] = useState({});

  // Group charts into ordered, non-empty sections.
  const chartGroups = React.useMemo(() => {
    const g = {};
    (all_charts || []).forEach((c) => {
      const s = sectionOf(c);
      (g[s] = g[s] || []).push(c);
    });
    return g;
  }, [all_charts]);
  const sections = SECTION_ORDER.filter((s) => (chartGroups[s] || []).length > 0);
  const currentSection =
    activeSection && chartGroups[activeSection] ? activeSection : sections[0];

  if (loading) {
    return (
      <section id="overview-section" className="analysis-section">
        <div className="flex justify-center items-center py-20">
          <div className="dash-spinner" />
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section id="overview-section" className="analysis-section">
        <div className="glass-soft p-5 text-rose-200 border-rose-400/30">{error}</div>
      </section>
    );
  }

  return (
    <section id="overview-section" className="analysis-section space-y-8">
      {aiNarrative && (
        <div className="glass-soft p-5 border-sky-400/20">
          <h2 className="section-heading mb-3">
            <span className="section-icon"><i className="fas fa-wand-magic-sparkles text-sky-300" /></span>
            <span>AI Executive Summary</span>
          </h2>
          <p className="text-slate-200 leading-relaxed text-[15px] w-full">{aiNarrative}</p>
        </div>
      )}

      {/* KPIs */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="section-heading">
            <span className="section-icon"><i className="fas fa-chart-pie text-sky-300" /></span>
            <span>Key metrics</span>
          </h2>
          {kpis && kpis.length > 0 && (
            <span className="neon-badge neon-blue">{kpis.length} signals</span>
          )}
        </div>
        {kpis && kpis.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {kpis.map((kpi, index) => (
              <div key={index} className="kpi-tile">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <p className="kpi-label" title={kpi.label}>{kpi.label}</p>
                    <p className="kpi-value" title={kpi.value}>{kpi.value}</p>
                  </div>
                  <div className="p-2 rounded-lg bg-sky-400/15 text-sky-300 border border-sky-400/25 flex-shrink-0">
                    <i className="fas fa-chart-line" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-state">
            <i className="fas fa-info-circle empty-icon" />
            <p>No meaningful KPIs could be generated for this dataset.</p>
          </div>
        )}
      </div>

      {/* Primary Chart */}
      {primary_chart && (
        <div>
          <h2 className="section-heading mb-4">
            <span className="section-icon"><i className="fas fa-star text-amber-300" /></span>
            <span>Primary Chart</span>
          </h2>
          <div
            className="glass-card p-6 chart-card chart-card-clickable"
            onClick={() => setModalChart(primary_chart)}
            title="Click to enlarge"
          >
            <div className="flex items-center justify-between mb-4 gap-4">
              <h3 className="text-lg font-semibold text-slate-100 truncate" title={primary_chart.title || primary_chart.column}>
                {primary_chart.title || primary_chart.column || 'Primary Chart'}
              </h3>
              <span className="neon-badge neon-blue">Featured</span>
            </div>
            <div className="chart-container chart-shell h-96">
              <ChartRenderer chartData={primary_chart} key={`primary-${refreshKey || 0}`} />
            </div>
          </div>
        </div>
      )}

      {/* Category Charts */}
      {category_charts && Object.keys(category_charts).length > 0 && (
        <div>
          <h2 className="section-heading mb-4">
            <span className="section-icon"><i className="fas fa-chart-bar text-fuchsia-300" /></span>
            <span>Category Charts</span>
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {Object.entries(category_charts).map(([column, chart_data]) => (
              <div
                key={column}
                className="glass-card p-5 chart-card chart-card-clickable"
                onClick={() => setModalChart(chart_data)}
                title="Click to enlarge"
              >
                <h4 className="text-base font-medium text-slate-100 mb-3 truncate" title={chart_data.title || column}>
                  {chart_data.title || column.charAt(0).toUpperCase() + column.slice(1)}
                </h4>
                <div className="chart-container chart-shell h-64">
                  <ChartRenderer chartData={chart_data} key={`${column}-${refreshKey || 0}`} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Composed, navigable chart sections (S13.2) */}
      {sections.length > 0 && (
        <div>
          <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-3 mb-4">
            <h2 className="section-heading">
              <span className="section-icon"><i className="fas fa-layer-group text-emerald-300" /></span>
              <span>Explore the data</span>
            </h2>
            {/* Section navigation — themed, like a real dashboard */}
            <div className="dash-section-nav flex flex-wrap gap-2">
              {sections.map((s) => {
                const m = SECTION_META[s] || SECTION_META.Other;
                const active = s === currentSection;
                return (
                  <button
                    key={s}
                    type="button"
                    onClick={() => setActiveSection(s)}
                    className="inline-flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm whitespace-nowrap transition-colors"
                    style={{
                      background: active ? 'rgba(14,165,233,0.18)' : 'rgba(148,163,184,0.08)',
                      border: `1px solid ${active ? 'rgba(125,211,252,0.55)' : 'rgba(148,163,184,0.22)'}`,
                      color: active ? '#e0f2fe' : '#cbd5e1',
                      cursor: 'pointer',
                    }}
                  >
                    <i className={`fas ${m.icon}`} style={{ color: m.color }} />
                    {s}
                    <span className="text-xs text-slate-400">{chartGroups[s].length}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {(() => {
            const charts = chartGroups[currentSection] || [];
            const isOpen = !!expanded[currentSection];
            const shown = isOpen ? charts : charts.slice(0, PER_SECTION_CAP);
            return (
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  {shown.map((chart, index) => (
                    <div
                      key={`${currentSection}-${chart.id || index}`}
                      className="glass-card p-5 chart-card chart-card-clickable"
                      onClick={() => setModalChart(chart)}
                      title="Click to enlarge"
                    >
                      <div className="flex items-center justify-between mb-3 gap-3">
                        <h4 className="text-sm font-medium text-slate-100 truncate" title={chart.title || chart.id || 'Chart'}>
                          {chart.title || chart.id || 'Chart'}
                        </h4>
                        <i className="fas fa-up-right-and-down-left-from-center text-xs text-slate-500 flex-shrink-0" />
                      </div>
                      <div className="chart-container chart-shell h-64">
                        <ChartRenderer chartData={chart} key={`${chart.id || index}-${refreshKey || 0}`} />
                      </div>
                    </div>
                  ))}
                </div>
                {charts.length > PER_SECTION_CAP && (
                  <div className="mt-4 text-center">
                    <button
                      type="button"
                      onClick={() => setExpanded((e) => ({ ...e, [currentSection]: !isOpen }))}
                      className="inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm text-sky-200 transition-colors"
                      style={{ background: 'rgba(14,165,233,0.10)', border: '1px solid rgba(125,211,252,0.4)', cursor: 'pointer' }}
                    >
                      <i className={`fas ${isOpen ? 'fa-chevron-up' : 'fa-chevron-down'}`} />
                      {isOpen ? 'Show fewer' : `Show all ${charts.length} in ${currentSection}`}
                    </button>
                  </div>
                )}
              </>
            );
          })()}
        </div>
      )}

      {!primary_chart && (!category_charts || Object.keys(category_charts).length === 0) && (!all_charts || all_charts.length === 0) && (
        <div className="empty-state">
          <i className="fas fa-chart-bar empty-icon" />
          <p>No charts could be generated for this dataset.</p>
        </div>
      )}

      {modalChart && (
        <ChartModal chart={modalChart} onClose={() => setModalChart(null)} />
      )}
    </section>
  );
};

export default OverviewTab;
