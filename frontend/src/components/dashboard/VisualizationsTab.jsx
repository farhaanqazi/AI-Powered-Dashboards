import React from 'react';
import ChartRenderer from '../charts/ChartRenderer';

const VisualizationsTab = ({ data, loading, error, refreshKey }) => {
  const safeData = data || {};
  const { eda_summary } = safeData;

  if (loading) {
    return (
      <section id="visualizations-section" className="analysis-section">
        <div className="flex justify-center items-center py-20">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section id="visualizations-section" className="analysis-section">
        <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-red-700">
          {error}
        </div>
      </section>
    );
  }

  if (!eda_summary || !eda_summary.patterns_and_relationships) {
    return (
      <section id="visualizations-section" className="analysis-section">
        <div className="bg-gray-50 rounded-xl p-8 text-center border-2 border-dashed border-gray-200">
          <i className="fas fa-chart-bar text-gray-400 text-2xl mb-3"></i>
          <p className="text-gray-500">Advanced visualizations not available for this dataset.</p>
        </div>
      </section>
    );
  }

  // Extract data from the EDA summary to create actual chart data
  const patterns_and_relationships = eda_summary.patterns_and_relationships || {};
  const correlations = patterns_and_relationships.correlations || [];
  const trends = patterns_and_relationships.trends || [];
  const outliers = patterns_and_relationships.outliers || [];
  const anomalies = patterns_and_relationships.anomalies || [];
  const use_cases = eda_summary.use_cases || [];
  const key_indicators = eda_summary.key_indicators || [];
  const recommendations = eda_summary.recommendations || [];

  return (
    <section id="visualizations-section" className="analysis-section">
      <div className="space-y-8">
        {/* Correlation Heatmap */}
        {correlations && correlations.length > 0 && (
          <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 chart-card">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-900 flex items-center">
                <i className="fas fa-link text-purple-500 mr-2"></i> Correlation Heatmap
              </h2>
              <span className="badge bg-purple-100 text-purple-700 text-xs px-3 py-1 rounded-full">Correlation</span>
            </div>
            <div className="chart-container h-96">
              <ChartRenderer
                chartData={{
                  id: 'correlation-heatmap',
                  title: 'Variable Correlations',
                  type: 'heatmap',
                  data: {
                    z: correlations.map(corr => [corr.correlation]),
                    x: correlations.map(corr => corr.variable1),
                    y: correlations.map(corr => corr.variable2),
                    type: 'heatmap',
                    colorscale: 'Viridis'
                  },
                  layout: {
                    title: 'Variable Correlations',
                    xaxis: { title: 'Variables' },
                    yaxis: { title: 'Variables' }
                  }
                }}
                key={`correlation-heatmap-${refreshKey || 0}`}
              />
            </div>
          </div>
        )}

        {/* Key Indicators */}
        {key_indicators && key_indicators.length > 0 && (
          <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 chart-card">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-900 flex items-center">
                <i className="fas fa-chart-bar text-blue-500 mr-2"></i> Key Indicators
              </h2>
              <span className="badge bg-blue-100 text-blue-700 text-xs px-3 py-1 rounded-full">Significance</span>
            </div>
            <div className="chart-container h-96">
              <ChartRenderer
                chartData={{
                  id: 'key-indicators-chart',
                  title: 'Key Indicators',
                  type: 'bar',
                  data: {
                    x: key_indicators.map(ki => ki.indicator || ki.label),
                    y: key_indicators.map(ki => ki.value || ki.score),
                    type: 'bar'
                  },
                  layout: {
                    title: 'Key Indicators',
                    xaxis: { title: 'Indicator' },
                    yaxis: { title: 'Value' }
                  }
                }}
                key={`key-indicators-${refreshKey || 0}`}
              />
            </div>
          </div>
        )}

        {/* Trend Analysis */}
        {trends && trends.length > 0 && (
          <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 chart-card">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-900 flex items-center">
                <i className="fas fa-chart-line text-green-500 mr-2"></i> Trend Analysis
              </h2>
              <span className="badge bg-green-100 text-green-700 text-xs px-3 py-1 rounded-full">Time Series</span>
            </div>
            <div className="chart-container h-96">
              <ChartRenderer
                chartData={{
                  id: 'trends-chart',
                  title: 'Trend Analysis',
                  type: 'scatter',
                  data: {
                    x: trends.map(t => t.datetime_column || t.period),
                    y: trends.map(t => t.numeric_column || t.value),
                    type: 'scatter',
                    mode: 'lines+markers'
                  },
                  layout: {
                    title: 'Trend Analysis',
                    xaxis: { title: 'Time Period' },
                    yaxis: { title: 'Value' }
                  }
                }}
                key={`trends-${refreshKey || 0}`}
              />
            </div>
          </div>
        )}

        {/* Outlier Visualization */}
        {outliers && outliers.length > 0 && (
          <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 chart-card">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-900 flex items-center">
                <i className="fas fa-exclamation-triangle text-red-500 mr-2"></i> Outlier Visualization
              </h2>
              <span className="badge bg-red-100 text-red-700 text-xs px-3 py-1 rounded-full">Anomaly</span>
            </div>
            <div className="chart-container h-96">
              <ChartRenderer
                chartData={{
                  id: 'outliers-chart',
                  title: 'Outlier Distribution',
                  type: 'box',
                  data: {
                    y: outliers.map(o => o.outlier_count || o.value),
                    type: 'box'
                  },
                  layout: {
                    title: 'Outlier Distribution',
                    yaxis: { title: 'Outlier Count' }
                  }
                }}
                key={`outliers-${refreshKey || 0}`}
              />
            </div>
          </div>
        )}

        {/* Use Cases Overview */}
        {use_cases && use_cases.length > 0 && (
          <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 chart-card">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-900 flex items-center">
                <i className="fas fa-briefcase text-amber-500 mr-2"></i> Use Cases Overview
              </h2>
              <span className="badge bg-amber-100 text-amber-700 text-xs px-3 py-1 rounded-full">Business</span>
            </div>
            <div className="chart-container h-96">
              <ChartRenderer
                chartData={{
                  id: 'use-cases-chart',
                  title: 'Use Cases Distribution',
                  type: 'pie',
                  data: {
                    labels: use_cases.map(uc => uc.use_case || uc.title),
                    values: use_cases.map(uc => uc.confidence || uc.score || 1),
                    type: 'pie'
                  },
                  layout: {
                    title: 'Use Cases Distribution'
                  }
                }}
                key={`use-cases-${refreshKey || 0}`}
              />
            </div>
          </div>
        )}

        {/* Show message if no visualizations are available */}
        {(!correlations || correlations.length === 0) &&
         (!trends || trends.length === 0) &&
         (!outliers || outliers.length === 0) &&
         (!use_cases || use_cases.length === 0) &&
         (!key_indicators || key_indicators.length === 0) && (
          <div className="bg-gray-50 rounded-xl p-8 text-center border-2 border-dashed border-gray-200">
            <i className="fas fa-info-circle text-gray-400 text-2xl mb-3"></i>
            <p className="text-gray-500">No advanced visualizations available for this dataset.</p>
            <p className="text-sm text-gray-400 mt-2">The EDA analysis did not detect significant patterns that warrant specialized visualizations.</p>
          </div>
        )}
      </div>
    </section>
  );
};

export default VisualizationsTab;
