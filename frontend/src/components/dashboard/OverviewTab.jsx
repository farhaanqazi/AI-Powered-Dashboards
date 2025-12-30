import React from 'react';
import ChartRenderer from '../charts/ChartRenderer';
import KPICard from '../kpi/KPICard';

const OverviewTab = ({ data }) => {
  const { kpis, primary_chart, category_charts, all_charts } = data;

  return (
    <section id="overview-section" className="analysis-section">
      <div className="light-card rounded-3xl p-6 mb-6">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-[0.25em] text-gray-500 mb-2">KPIs</p>
            <div className="flex flex-wrap gap-2">
              {kpis && kpis.length > 0 ? (
                kpis.map((kpi, index) => (
                  <KPICard key={index} kpi={kpi} />
                ))
              ) : (
                <p className="text-gray-500 text-sm">No meaningful KPIs could be generated for this dataset.</p>
              )}
            </div>
          </div>
          <div className="text-right text-sm text-gray-600">
            <p className="font-semibold text-gray-900">Quick summary</p>
            <p>We prioritized key distributions and metrics to provide clear insights.</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {primary_chart || (category_charts && Object.keys(category_charts).length > 0) ? (
          <>
            {primary_chart && (
              <div className="lg:col-span-2 light-card rounded-3xl p-5">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-lg font-semibold text-gray-900">
                    {primary_chart.title || primary_chart.column ? `${primary_chart.column.charAt(0).toUpperCase() + primary_chart.column.slice(1)}` : 'Primary Chart'}
                  </h3>
                  <span className="badge badge-soft">Featured</span>
                </div>
                <div className="chart-container h-80">
                  <ChartRenderer chartData={primary_chart} />
                </div>
              </div>
            )}

            <div className="lg:col-span-1 space-y-4">
              {category_charts ? (
                Object.entries(category_charts).map(([column, chart_data]) => (
                  <div key={column} className="light-card rounded-2xl p-4">
                    <h4 className="text-md font-semibold text-gray-900 mb-2">
                      {chart_data.title || column.charAt(0).toUpperCase() + column.slice(1)}
                    </h4>
                    <div className="chart-container h-48">
                      <ChartRenderer chartData={chart_data} />
                    </div>
                  </div>
                ))
              ) : (
                <div className="light-card rounded-2xl p-4 text-center text-gray-500 italic">
                  No categorical charts generated.
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="col-span-full light-card rounded-2xl p-4 text-center text-gray-500 italic">
            No valid charts could be generated for the main categorical variables.
          </div>
        )}
      </div>

      <div className="light-card rounded-3xl p-6 mt-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-semibold text-gray-900">All Generated Charts</h2>
          <span className="badge badge-soft">Gallery</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {all_charts && all_charts.length > 0 ? (
            all_charts.map((chart, index) => (
              <div key={index} className="light-card rounded-2xl p-4">
                <h4 className="text-md font-semibold text-gray-900 mb-2">
                  {chart.title || chart.id || 'Chart'}
                </h4>
                <div className="chart-container h-64">
                  <ChartRenderer chartData={chart} />
                </div>
              </div>
            ))
          ) : (
            <div className="col-span-full light-card rounded-2xl p-4 text-center text-gray-500 italic">
              No additional charts generated.
            </div>
          )}
        </div>
      </div>
    </section>
  );
};

export default OverviewTab;