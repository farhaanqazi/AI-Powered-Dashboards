import React from 'react';

const corrTone = (v) => {
  const a = Math.abs(v);
  if (a > 0.7) return { chip: 'neon-rose', bar: 'corr-strong', label: 'Strong' };
  if (a > 0.5) return { chip: 'neon-amber', bar: 'corr-mid', label: 'Moderate' };
  return { chip: 'neon-emerald', bar: 'corr-weak', label: 'Weak' };
};

const priorityTone = (p) => {
  switch ((p || '').toLowerCase()) {
    case 'high':   return 'neon-rose';
    case 'medium': return 'neon-amber';
    default:       return 'neon-emerald';
  }
};

const EDATab = ({ data }) => {
  const { eda_summary } = data || {};

  if (!eda_summary) {
    return (
      <div className="empty-state">
        <i className="fas fa-search empty-icon" />
        <p>No EDA summary available for this dataset.</p>
      </div>
    );
  }

  const {
    key_indicators = [],
    use_cases = [],
    patterns_and_relationships,
    recommendations = [],
  } = eda_summary;

  const correlations = patterns_and_relationships?.correlations || [];
  const outliers = patterns_and_relationships?.outliers || [];
  const trends = patterns_and_relationships?.trends || [];

  const hasAnything =
    key_indicators.length || use_cases.length ||
    correlations.length || outliers.length || trends.length ||
    recommendations.length;

  return (
    <section id="eda-section" className="analysis-section">
      <div className="mb-6">
        <div className="text-[11px] uppercase tracking-[0.32em] text-slate-400 mb-1">Exploratory analysis</div>
        <h2 className="text-xl md:text-2xl font-semibold text-slate-100">What the AI noticed</h2>
        <p className="text-sm text-slate-400 mt-1 max-w-xl">
          Patterns, relationships and recommendations distilled from your dataset.
        </p>
      </div>

      <div className="space-y-7">
        {/* Key indicators */}
        {key_indicators.length > 0 && (
          <div className="glass-card p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="section-heading">
                <span className="section-icon"><i className="fas fa-lightbulb text-amber-300" /></span>
                <span>Key Indicators</span>
              </h2>
              <span className="neon-badge neon-amber">{key_indicators.length} found</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {key_indicators.slice(0, 6).map((indicator, i) => (
                <div key={i} className="glass-soft p-4">
                  <p className="font-medium text-slate-100">{indicator.indicator}</p>
                  <p className="text-sm text-slate-400 mt-1 leading-relaxed">{indicator.description}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Use cases */}
        {use_cases.length > 0 && (
          <div className="glass-card p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="section-heading">
                <span className="section-icon"><i className="fas fa-briefcase text-sky-300" /></span>
                <span>Potential Use Cases</span>
              </h2>
              <span className="neon-badge neon-blue">{use_cases.length} ideas</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {use_cases.slice(0, 4).map((uc, i) => (
                <div key={i} className="kpi-tile">
                  <p className="font-semibold text-slate-100">{uc.use_case}</p>
                  <p className="text-sm text-slate-400 mt-1 leading-relaxed">{uc.description}</p>
                  {uc.key_inputs && uc.key_inputs.length > 0 && (
                    <div className="mt-3">
                      <p className="text-[10px] uppercase tracking-[0.25em] text-slate-500 mb-1.5">
                        <i className="fas fa-key mr-1.5 text-amber-300/70" /> Key inputs
                      </p>
                      <div className="flex flex-wrap gap-1.5">
                        {uc.key_inputs.slice(0, 4).map((input, idx) => (
                          <span key={idx} className="metric-chip">{input}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Patterns & Relationships */}
        {(correlations.length > 0 || outliers.length > 0 || trends.length > 0) && (
          <div className="glass-card p-6">
            <h2 className="section-heading mb-5">
              <span className="section-icon"><i className="fas fa-project-diagram text-fuchsia-300" /></span>
              <span>Patterns &amp; Relationships</span>
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {correlations.length > 0 && (
                <div className="glass-soft p-4">
                  <h3 className="font-medium text-slate-100 mb-3 flex items-center gap-2">
                    <i className="fas fa-link text-fuchsia-300" />
                    Correlations
                  </h3>
                  <div className="space-y-3">
                    {correlations.slice(0, 4).map((corr, i) => {
                      const t = corrTone(corr.correlation);
                      return (
                        <div key={i}>
                          <div className="flex justify-between items-center mb-1.5 gap-2">
                            <span className="text-sm text-slate-200 truncate">
                              {corr.variable1} <span className="text-slate-500">↔</span> {corr.variable2}
                            </span>
                            <span className={`neon-badge ${t.chip}`}>
                              {corr.correlation.toFixed(3)}
                            </span>
                          </div>
                          <div className="corr-bar-track">
                            <div
                              className={`corr-bar-fill ${t.bar}`}
                              style={{ width: `${Math.abs(corr.correlation) * 100}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {outliers.length > 0 && (
                <div className="glass-soft p-4">
                  <h3 className="font-medium text-slate-100 mb-3 flex items-center gap-2">
                    <i className="fas fa-triangle-exclamation text-rose-300" />
                    Outliers
                  </h3>
                  <div className="space-y-2">
                    {outliers.slice(0, 4).map((o, i) => (
                      <div key={i} className="flex justify-between items-center gap-2">
                        <div className="min-w-0">
                          <p className="text-sm text-slate-200 truncate">{o.column}</p>
                          {Number.isFinite(o.outlier_percentage) && (
                            <p className="text-xs text-slate-500">{o.outlier_percentage.toFixed(2)}% of data</p>
                          )}
                        </div>
                        <span className="neon-badge neon-rose flex-shrink-0">
                          {o.outlier_count} outliers
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {trends.length > 0 && (
              <div className="mt-5 glass-soft p-4">
                <h3 className="font-medium text-slate-100 mb-3 flex items-center gap-2">
                  <i className="fas fa-chart-line text-emerald-300" />
                  Trends
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {trends.slice(0, 4).map((t, i) => (
                    <div key={i} className="flex justify-between items-center gap-2">
                      <span className="text-sm text-slate-200 truncate">
                        {t.datetime_column} <span className="text-slate-500">vs</span> {t.numeric_column}
                      </span>
                      <span className="neon-badge neon-emerald flex-shrink-0">{t.trend_type}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Recommendations */}
        {recommendations.length > 0 && (
          <div className="glass-card p-6">
            <h2 className="section-heading mb-4">
              <span className="section-icon"><i className="fas fa-rocket text-purple-300" /></span>
              <span>Recommendations</span>
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {recommendations.map((rec, i) => (
                <div key={i} className="kpi-tile">
                  <div className="flex items-start gap-3">
                    <span className={`neon-badge ${priorityTone(rec.priority)} flex-shrink-0`}>
                      {rec.priority || 'low'}
                    </span>
                    <div className="min-w-0">
                      <p className="font-medium text-slate-100">{rec.title}</p>
                      <p className="text-sm text-slate-400 mt-1 leading-relaxed">{rec.description}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {!hasAnything && (
          <div className="empty-state">
            <i className="fas fa-info-circle empty-icon" />
            <p>No EDA insights available for this dataset.</p>
          </div>
        )}
      </div>
    </section>
  );
};

export default EDATab;
