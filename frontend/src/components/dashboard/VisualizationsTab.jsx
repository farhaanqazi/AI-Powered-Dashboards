import React from 'react';
import ChartRenderer from '../charts/ChartRenderer';

const ChartPanel = ({ icon, iconColor, title, badgeTone, badgeLabel, eyebrow, children, height = 'h-96' }) => (
  <div className="glass-card p-6 chart-card">
    <div className="flex items-start justify-between mb-5 gap-4">
      <div className="min-w-0">
        {eyebrow && (
          <span className="block text-[10px] uppercase tracking-[0.32em] text-slate-400 mb-1">
            {eyebrow}
          </span>
        )}
        <h2 className="text-base md:text-lg font-semibold text-slate-100 flex items-center gap-3">
          <span className="section-icon"><i className={`fas ${icon}`} style={{ color: iconColor }} /></span>
          <span className="truncate">{title}</span>
        </h2>
      </div>
      <span className={`neon-badge ${badgeTone} flex-shrink-0`}>{badgeLabel}</span>
    </div>
    <div className={`chart-container chart-shell ${height}`}>
      {children}
    </div>
  </div>
);

const VisualizationsTab = ({ data, loading, error, refreshKey }) => {
  const safeData = data || {};
  const { eda_summary } = safeData;

  if (loading) {
    return (
      <section id="visualizations-section" className="analysis-section">
        <div className="flex justify-center items-center py-20">
          <div className="dash-spinner" />
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section id="visualizations-section" className="analysis-section">
        <div className="glass-soft p-5 text-rose-200 border-rose-400/30">{error}</div>
      </section>
    );
  }

  if (!eda_summary || !eda_summary.patterns_and_relationships) {
    return (
      <section id="visualizations-section" className="analysis-section">
        <div className="empty-state">
          <i className="fas fa-chart-bar empty-icon" />
          <p>Advanced visualizations not available for this dataset.</p>
        </div>
      </section>
    );
  }

  const patterns_and_relationships = eda_summary.patterns_and_relationships || {};
  const correlations = patterns_and_relationships.correlations || [];
  const trends = patterns_and_relationships.trends || [];
  const outliers = patterns_and_relationships.outliers || [];
  const use_cases = eda_summary.use_cases || [];
  const key_indicators = eda_summary.key_indicators || [];

  const buildCorrelationMatrix = (correlations) => {
    if (!correlations || correlations.length === 0) return null;
    const vars = new Set();
    correlations.forEach(c => { vars.add(c.variable1); vars.add(c.variable2); });
    const varArray = Array.from(vars).sort();
    const n = varArray.length;
    const varIndex = Object.fromEntries(varArray.map((v, i) => [v, i]));
    const z = Array(n).fill(0).map(() => Array(n).fill(0));
    varArray.forEach((v, i) => z[i][i] = 1.0);
    correlations.forEach(c => {
      const i = varIndex[c.variable1];
      const j = varIndex[c.variable2];
      if (i !== undefined && j !== undefined) { z[i][j] = c.correlation; z[j][i] = c.correlation; }
    });
    return { z, variables: varArray };
  };

  const correlationMatrix = buildCorrelationMatrix(correlations);

  const noViz =
    (!correlations || correlations.length === 0) &&
    (!trends || trends.length === 0) &&
    (!outliers || outliers.length === 0) &&
    (!use_cases || use_cases.length === 0) &&
    (!key_indicators || key_indicators.length === 0);

  return (
    <section id="visualizations-section" className="analysis-section">
      {/* Section intro */}
      <div className="mb-6 flex flex-col sm:flex-row sm:items-end sm:justify-between gap-3">
        <div>
          <div className="text-[11px] uppercase tracking-[0.32em] text-slate-400 mb-1">Visualization Gallery</div>
          <h2 className="text-xl md:text-2xl font-semibold text-slate-100">
            Patterns the model surfaced
          </h2>
          <p className="text-sm text-slate-400 mt-1 max-w-xl">
            Each panel highlights a different kind of signal — correlations, trends, anomalies, and the use cases this data unlocks.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="metric-chip">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-sky-400 shadow-[0_0_8px_#60a5fa]" />
            {[correlationMatrix, key_indicators?.length, trends?.length, outliers?.length, use_cases?.length].filter(Boolean).length} active panels
          </span>
        </div>
      </div>

      <div className="space-y-6">
        {/* Correlation Heatmap */}
        {correlationMatrix && (
          <ChartPanel
            eyebrow="Variable relationships"
            icon="fa-link"
            iconColor="#d8b4fe"
            title="Correlation Heatmap"
            badgeTone="neon-purple"
            badgeLabel="Correlation"
          >
            <ChartRenderer
              chartData={{
                id: 'correlation-heatmap',
                title: '',
                type: 'heatmap',
                data: {
                  z: correlationMatrix.z,
                  x: correlationMatrix.variables,
                  y: correlationMatrix.variables,
                  type: 'heatmap',
                  zmin: -1,
                  zmax: 1,
                },
                layout: {
                  xaxis: { title: 'Variables' },
                  yaxis: { title: 'Variables' },
                },
              }}
              key={`correlation-heatmap-${refreshKey || 0}`}
            />
          </ChartPanel>
        )}

        {/* Key Indicators */}
        {key_indicators && key_indicators.length > 0 && (
          <ChartPanel
            eyebrow="Top signals"
            icon="fa-chart-bar"
            iconColor="#93c5fd"
            title="Key Indicators"
            badgeTone="neon-blue"
            badgeLabel="Significance"
          >
            <ChartRenderer
              chartData={{
                id: 'key-indicators-chart',
                title: '',
                type: 'bar',
                data: {
                  x: key_indicators.map(ki => ki.indicator || ki.label),
                  y: key_indicators.map(ki => ki.value || ki.score),
                  type: 'bar',
                },
                layout: {
                  xaxis: { title: 'Indicator' },
                  yaxis: { title: 'Value' },
                },
              }}
              key={`key-indicators-${refreshKey || 0}`}
            />
          </ChartPanel>
        )}

        {/* Trend Analysis */}
        {trends && trends.length > 0 && (
          <ChartPanel
            eyebrow="Movement over time"
            icon="fa-chart-line"
            iconColor="#6ee7b7"
            title="Trend Analysis"
            badgeTone="neon-emerald"
            badgeLabel="Time series"
          >
            <ChartRenderer
              chartData={{
                id: 'trends-chart',
                title: '',
                type: 'scatter',
                data: {
                  x: trends.map(t => t.datetime_column || t.period),
                  y: trends.map(t => t.numeric_column || t.value),
                  type: 'scatter',
                  mode: 'lines+markers',
                },
                layout: {
                  xaxis: { title: 'Time Period' },
                  yaxis: { title: 'Value' },
                },
              }}
              key={`trends-${refreshKey || 0}`}
            />
          </ChartPanel>
        )}

        {/* Outliers */}
        {outliers && outliers.length > 0 && (
          <ChartPanel
            eyebrow="Distribution tails"
            icon="fa-triangle-exclamation"
            iconColor="#fda4af"
            title="Outlier Visualization"
            badgeTone="neon-rose"
            badgeLabel="Anomaly"
          >
            <ChartRenderer
              chartData={{
                id: 'outliers-chart',
                title: '',
                type: 'box',
                data: {
                  y: outliers.map(o => o.outlier_count || o.value),
                  type: 'box',
                },
                layout: { yaxis: { title: 'Outlier Count' } },
              }}
              key={`outliers-${refreshKey || 0}`}
            />
          </ChartPanel>
        )}

        {/* Use cases */}
        {use_cases && use_cases.length > 0 && (
          <ChartPanel
            eyebrow="Where this data shines"
            icon="fa-briefcase"
            iconColor="#fcd34d"
            title="Use Cases Overview"
            badgeTone="neon-amber"
            badgeLabel="Business"
          >
            <ChartRenderer
              chartData={{
                id: 'use-cases-chart',
                title: '',
                type: 'pie',
                data: {
                  labels: use_cases.map(uc => uc.use_case || uc.title),
                  values: use_cases.map(uc => uc.confidence || uc.score || 1),
                  type: 'pie',
                },
                layout: {},
              }}
              key={`use-cases-${refreshKey || 0}`}
            />
          </ChartPanel>
        )}

        {noViz && (
          <div className="empty-state">
            <i className="fas fa-info-circle empty-icon" />
            <p>No advanced visualizations available for this dataset.</p>
            <p className="text-xs text-slate-500 mt-2">The EDA analysis did not detect significant patterns that warrant specialized visualizations.</p>
          </div>
        )}
      </div>
    </section>
  );
};

export default VisualizationsTab;
