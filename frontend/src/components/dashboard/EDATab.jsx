import React from 'react';

const EDATab = ({ data }) => {
  console.log("EDATab received data:", data);
  const { eda_summary } = data;
  console.log("EDATab eda_summary:", eda_summary);

  if (!eda_summary) {
    return (
      <div className="bg-gray-50 rounded-xl p-8 text-center border-2 border-dashed border-gray-200">
        <i className="fas fa-search text-gray-400 text-2xl mb-3"></i>
        <p className="text-gray-500">No EDA summary available for this dataset.</p>
      </div>
    );
  }

  return (
    <section id="eda-section" className="analysis-section">
      <div className="space-y-8">
        {/* Key Indicators */}
        {eda_summary.key_indicators && eda_summary.key_indicators.length > 0 && (
          <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <i className="fas fa-lightbulb text-yellow-500 mr-2"></i> Key Indicators
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {eda_summary.key_indicators.slice(0, 5).map((indicator, index) => (
                <div key={index} className="bg-gradient-to-br from-yellow-50 to-yellow-100 rounded-lg p-4 border border-yellow-100">
                  <p className="font-medium text-gray-900">{indicator.indicator}</p>
                  <p className="text-sm text-gray-600">{indicator.description}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Use Cases */}
        {eda_summary.use_cases && eda_summary.use_cases.length > 0 && (
          <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <i className="fas fa-briefcase text-blue-500 mr-2"></i> Potential Use Cases
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {eda_summary.use_cases.slice(0, 3).map((useCase, index) => (
                <div key={index} className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-4 border border-blue-100">
                  <p className="font-medium text-gray-900">{useCase.use_case}</p>
                  <p className="text-sm text-gray-600">{useCase.description}</p>
                  <div className="mt-2">
                    <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                      <i className="fas fa-key mr-1"></i> Key Inputs
                    </span>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {useCase.key_inputs && useCase.key_inputs.slice(0, 3).map((input, idx) => (
                        <span key={idx} className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                          {input}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Patterns & Relationships */}
        {eda_summary.patterns_and_relationships && (
          <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <i className="fas fa-project-diagram text-purple-500 mr-2"></i> Patterns & Relationships
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Correlations */}
              {eda_summary.patterns_and_relationships.correlations && eda_summary.patterns_and_relationships.correlations.length > 0 && (
                <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-4 border border-purple-100">
                  <h3 className="font-medium text-gray-900 mb-3 flex items-center">
                    <i className="fas fa-link mr-2 text-purple-600"></i> Correlations
                  </h3>
                  <div className="space-y-2">
                    {eda_summary.patterns_and_relationships.correlations.slice(0, 3).map((corr, index) => (
                      <div key={index} className="bg-white rounded p-3 shadow-sm">
                        <div className="flex justify-between items-center">
                          <span className="text-sm font-medium">{corr.variable1} ↔ {corr.variable2}</span>
                          <span className={`px-2 py-1 rounded text-xs font-bold ${
                            Math.abs(corr.correlation) > 0.7
                              ? 'bg-red-100 text-red-800'
                              : Math.abs(corr.correlation) > 0.5
                                ? 'bg-orange-100 text-orange-800'
                                : 'bg-green-100 text-green-800'
                          }`}>
                            {corr.correlation.toFixed(3)}
                          </span>
                        </div>
                        <div className="mt-1">
                          <span className={`inline-block w-full h-2 rounded-full ${
                            Math.abs(corr.correlation) > 0.7
                              ? 'bg-red-400'
                              : Math.abs(corr.correlation) > 0.5
                                ? 'bg-orange-400'
                                : 'bg-green-400'
                          }`} style={{ width: `${Math.abs(corr.correlation) * 100}%` }}></span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Outliers */}
              {eda_summary.patterns_and_relationships.outliers && eda_summary.patterns_and_relationships.outliers.length > 0 && (
                <div className="bg-gradient-to-br from-red-50 to-red-100 rounded-lg p-4 border border-red-100">
                  <h3 className="font-medium text-gray-900 mb-3 flex items-center">
                    <i className="fas fa-exclamation-triangle mr-2 text-red-600"></i> Outliers
                  </h3>
                  <div className="space-y-2">
                    {eda_summary.patterns_and_relationships.outliers.slice(0, 3).map((outlier, index) => (
                      <div key={index} className="bg-white rounded p-3 shadow-sm">
                        <div className="flex justify-between items-center">
                          <span className="text-sm font-medium">{outlier.column}</span>
                          <span className="px-2 py-1 rounded text-xs font-bold bg-red-100 text-red-800">
                            {outlier.outlier_count} outliers
                          </span>
                        </div>
                        <div className="mt-1 text-xs text-gray-600">
                          {outlier.outlier_percentage ? `${outlier.outlier_percentage.toFixed(2)}% of data` : ''}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Trends */}
            {eda_summary.patterns_and_relationships.trends && eda_summary.patterns_and_relationships.trends.length > 0 && (
              <div className="mt-6 bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-4 border border-green-100">
                <h3 className="font-medium text-gray-900 mb-3 flex items-center">
                  <i className="fas fa-chart-line mr-2 text-green-600"></i> Trends
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {eda_summary.patterns_and_relationships.trends.slice(0, 3).map((trend, index) => (
                    <div key={index} className="bg-white rounded p-3 shadow-sm">
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium">{trend.datetime_column} vs {trend.numeric_column}</span>
                        <span className="px-2 py-1 rounded text-xs font-medium bg-green-100 text-green-800">
                          {trend.trend_type}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Recommendations */}
        {eda_summary.recommendations && eda_summary.recommendations.length > 0 && (
          <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <i className="fas fa-rocket text-indigo-500 mr-2"></i> Recommendations
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {eda_summary.recommendations.map((rec, index) => (
                <div key={index} className="bg-gradient-to-br from-indigo-50 to-indigo-100 rounded-lg p-4 border border-indigo-100">
                  <div className="flex items-start">
                    <div className="mr-3">
                      <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                        rec.priority === 'high'
                          ? 'bg-red-100 text-red-800'
                          : rec.priority === 'medium'
                            ? 'bg-yellow-100 text-yellow-800'
                            : 'bg-green-100 text-green-800'
                      }`}>
                        {rec.priority}
                      </span>
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">{rec.title}</p>
                      <p className="text-sm text-gray-600">{rec.description}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Fallback message when no EDA data is available */}
        {!eda_summary.key_indicators && !eda_summary.use_cases && !eda_summary.patterns_and_relationships && !eda_summary.recommendations && (
          <div className="bg-gray-50 rounded-xl p-8 text-center border-2 border-dashed border-gray-200">
            <i className="fas fa-info-circle text-gray-400 text-2xl mb-3"></i>
            <p className="text-gray-500">No EDA insights available for this dataset.</p>
          </div>
        )}
      </div>
    </section>
  );
};

export default EDATab;