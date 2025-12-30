import React from 'react';
import ChartRenderer from '../charts/ChartRenderer';

const VisualizationsTab = ({ data }) => {
  const { eda_summary } = data;

  if (!eda_summary || !eda_summary.patterns_and_relationships) {
    return (
      <section id="visualizations-section" className="analysis-section">
        <div className="light-card rounded-3xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-semibold text-gray-900">Advanced Visuals</h2>
            <span className="badge badge-soft">EDA-driven</span>
          </div>
          <div className="alert alert-warning light-card bg-amber-50 text-amber-700 border border-amber-200">
            <div className="flex-1 text-amber-700">
              Advanced visualizations not available for this dataset.
            </div>
          </div>
        </div>
      </section>
    );
  }

  // Mock chart data for visualization examples
  const correlationHeatmapData = {
    id: 'correlation-heatmap',
    title: 'Correlation Heatmap',
    type: 'heatmap',
    data: {
      z: [
        [1.0, 0.85, 0.2, -0.1],
        [0.85, 1.0, 0.1, -0.05],
        [0.2, 0.1, 1.0, 0.3],
        [-0.1, -0.05, 0.3, 1.0]
      ],
      x: ['Price', 'Revenue', 'Quantity', 'Rating'],
      y: ['Price', 'Revenue', 'Quantity', 'Rating'],
      type: 'heatmap'
    },
    layout: {
      title: 'Correlation Heatmap',
      xaxis: { title: 'Variables' },
      yaxis: { title: 'Variables' }
    }
  };

  const keyIndicatorsData = {
    id: 'key-indicators-chart',
    title: 'Key Indicators',
    type: 'bar',
    data: {
      x: ['Revenue Concentration', 'Seasonal Patterns', 'Customer Segments'],
      y: [80, 75, 90],
      type: 'bar'
    },
    layout: {
      title: 'Key Indicators (%)',
      xaxis: { title: 'Indicator' },
      yaxis: { title: 'Score (%)' }
    }
  };

  const trendsData = {
    id: 'trends-chart',
    title: 'Trend Analysis',
    type: 'scatter',
    data: {
      x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
      y: [100, 120, 110, 130, 140, 160, 180, 170, 150, 140, 160, 200],
      type: 'scatter',
      mode: 'lines+markers'
    },
    layout: {
      title: 'Revenue Trend Over Time',
      xaxis: { title: 'Month' },
      yaxis: { title: 'Revenue' }
    }
  };

  const outliersData = {
    id: 'outliers-chart',
    title: 'Outlier Visualization',
    type: 'box',
    data: {
      y: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50], // 50 is an outlier
      type: 'box'
    },
    layout: {
      title: 'Revenue Distribution with Outliers',
      yaxis: { title: 'Revenue' }
    }
  };

  const useCasesData = {
    id: 'use-cases-chart',
    title: 'Use Cases Overview',
    type: 'pie',
    data: {
      labels: ['Inventory Planning', 'Pricing Strategy', 'Customer Targeting'],
      values: [40, 35, 25],
      type: 'pie'
    },
    layout: {
      title: 'Use Cases Distribution'
    }
  };

  return (
    <section id="visualizations-section" className="analysis-section">
      <div className="light-card rounded-3xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-semibold text-gray-900">Advanced Visuals</h2>
          <span className="badge badge-soft">EDA-driven</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {eda_summary.patterns_and_relationships.correlations && (
            <>
              <div className="light-card rounded-2xl p-4 flex flex-col">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-md font-semibold text-gray-900">Correlation Heatmap</h4>
                  <span className="badge badge-soft">Correlation</span>
                </div>
                <div className="chart-container flex-grow h-80">
                  <ChartRenderer chartData={correlationHeatmapData} />
                </div>
              </div>
              <div className="light-card rounded-2xl p-4 flex flex-col">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-md font-semibold text-gray-900">Key Indicators</h4>
                  <span className="badge badge-soft">Significance</span>
                </div>
                <div className="chart-container flex-grow h-80">
                  <ChartRenderer chartData={keyIndicatorsData} />
                </div>
              </div>
            </>
          )}
          
          {eda_summary.patterns_and_relationships.trends && (
            <div className="light-card rounded-2xl p-4 flex flex-col">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-md font-semibold text-gray-900">Trend Analysis</h4>
                <span className="badge badge-soft">Time Series</span>
              </div>
              <div className="chart-container flex-grow h-80">
                <ChartRenderer chartData={trendsData} />
              </div>
            </div>
          )}
          
          {eda_summary.patterns_and_relationships.outliers && (
            <div className="light-card rounded-2xl p-4 flex flex-col">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-md font-semibold text-gray-900">Outlier Visualization</h4>
                <span className="badge badge-soft">Anomaly</span>
              </div>
              <div className="chart-container flex-grow h-80">
                <ChartRenderer chartData={outliersData} />
              </div>
            </div>
          )}
          
          {eda_summary.use_cases && (
            <div className="light-card rounded-2xl p-4 flex flex-col">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-md font-semibold text-gray-900">Use Cases Overview</h4>
                <span className="badge badge-soft">Product</span>
              </div>
              <div className="chart-container flex-grow h-80">
                <ChartRenderer chartData={useCasesData} />
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};

export default VisualizationsTab;