import React from 'react';
import ChartRenderer from '../charts/ChartRenderer';
import KPICard from '../kpi/KPICard';

const OverviewTab = ({ data }) => {
  const { kpis, primary_chart, category_charts, all_charts } = data;

  return (
    <section id="overview-section" className="analysis-section">
      {/* KPIs Section */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <i className="fas fa-chart-pie text-blue-500 mr-2"></i> Key Performance Indicators
        </h2>
        {kpis && kpis.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {kpis.map((kpi, index) => (
              <div key={index} className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-4 border border-blue-100">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">{kpi.label}</p>
                    <p className="text-lg font-bold text-gray-900">{kpi.value}</p>
                  </div>
                  <div className="p-2 rounded-lg bg-blue-100 text-blue-600">
                    <i className="fas fa-chart-line"></i>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="bg-gray-50 rounded-xl p-8 text-center border-2 border-dashed border-gray-200">
            <i className="fas fa-info-circle text-gray-400 text-2xl mb-3"></i>
            <p className="text-gray-500">No meaningful KPIs could be generated for this dataset.</p>
          </div>
        )}
      </div>

      {/* Primary Chart */}
      {primary_chart && (
        <div className="mb-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <i className="fas fa-star text-yellow-500 mr-2"></i> Primary Chart
          </h2>
          <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold text-gray-900">
                {primary_chart.title || primary_chart.column ? `${primary_chart.column.charAt(0).toUpperCase() + primary_chart.column.slice(1)}` : 'Primary Chart'}
              </h3>
              <span className="badge bg-blue-100 text-blue-700 text-xs px-3 py-1 rounded-full">Featured</span>
            </div>
            <div className="chart-container h-96">
              <ChartRenderer chartData={primary_chart} />
            </div>
          </div>
        </div>
      )}

      {/* Category Charts */}
      {category_charts && Object.keys(category_charts).length > 0 && (
        <div className="mb-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <i className="fas fa-chart-bar text-purple-500 mr-2"></i> Category Charts
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.entries(category_charts).map(([column, chart_data]) => (
              <div key={column} className="bg-white rounded-xl p-5 shadow-sm border border-gray-100">
                <h4 className="text-lg font-medium text-gray-900 mb-3">
                  {chart_data.title || column.charAt(0).toUpperCase() + column.slice(1)}
                </h4>
                <div className="chart-container h-64">
                  <ChartRenderer chartData={chart_data} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* All Charts Gallery */}
      {all_charts && all_charts.length > 0 && (
        <div>
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <i className="fas fa-th-large text-green-500 mr-2"></i> All Generated Charts
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {all_charts.map((chart, index) => (
              <div key={index} className="bg-white rounded-xl p-5 shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
                <h4 className="text-md font-medium text-gray-900 mb-3 truncate">
                  {chart.title || chart.id || 'Chart'}
                </h4>
                <div className="chart-container h-64">
                  <ChartRenderer chartData={chart} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Fallback message when no charts are available */}
      {!primary_chart && (!category_charts || Object.keys(category_charts).length === 0) && (!all_charts || all_charts.length === 0) && (
        <div className="bg-gray-50 rounded-xl p-8 text-center border-2 border-dashed border-gray-200">
          <i className="fas fa-chart-bar text-gray-400 text-2xl mb-3"></i>
          <p className="text-gray-500">No charts could be generated for this dataset.</p>
        </div>
      )}
    </section>
  );
};

export default OverviewTab;