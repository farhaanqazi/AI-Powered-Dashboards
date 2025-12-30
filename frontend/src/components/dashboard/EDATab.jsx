import React from 'react';

const EDATab = ({ data }) => {
  const { eda_summary } = data;

  if (!eda_summary) {
    return (
      <div className="light-card rounded-3xl p-6">
        <p className="text-gray-500">No EDA summary available for this dataset.</p>
      </div>
    );
  }

  return (
    <section id="eda-section" className="analysis-section">
      <div className="light-card rounded-3xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-semibold text-gray-900">EDA Insights</h2>
          <span className="badge badge-soft">Analysis</span>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <h3 className="text-xl font-semibold text-gray-900">Key Indicators</h3>
            {eda_summary.key_indicators ? (
              <ul className="list-disc pl-5 space-y-1 text-gray-700">
                {eda_summary.key_indicators.slice(0, 5).map((indicator, index) => (
                  <li key={index}>{indicator.indicator}: {indicator.description}</li>
                ))}
              </ul>
            ) : (
              <p className="text-gray-500">No key indicators identified.</p>
            )}
          </div>
          
          <div className="space-y-4">
            <h3 className="text-xl font-semibold text-gray-900">Potential Use Cases</h3>
            {eda_summary.use_cases ? (
              <ul className="list-disc pl-5 space-y-1 text-gray-700">
                {eda_summary.use_cases.slice(0, 3).map((useCase, index) => (
                  <li key={index}>{useCase.use_case}: {useCase.description}</li>
                ))}
              </ul>
            ) : (
              <p className="text-gray-500">No specific use cases suggested.</p>
            )}
          </div>
        </div>

        <div className="mt-6 space-y-4">
          <h3 className="text-xl font-semibold text-gray-900">Patterns & Relationships</h3>
          {eda_summary.patterns_and_relationships ? (
            <>
              {eda_summary.patterns_and_relationships.correlations && (
                <div>
                  <h4 className="font-medium text-gray-800">Correlations:</h4>
                  <ul className="list-disc pl-5 text-gray-700">
                    {eda_summary.patterns_and_relationships.correlations.slice(0, 3).map((corr, index) => (
                      <li key={index}>
                        {corr.variable1} ↔ {corr.variable2} ({corr.correlation.toFixed(3)})
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              
              {eda_summary.patterns_and_relationships.trends && (
                <div>
                  <h4 className="font-medium text-gray-800">Trends:</h4>
                  <ul className="list-disc pl-5 text-gray-700">
                    {eda_summary.patterns_and_relationships.trends.slice(0, 3).map((trend, index) => (
                      <li key={index}>
                        {trend.datetime_column} vs {trend.numeric_column} ({trend.trend_type})
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              
              {eda_summary.patterns_and_relationships.outliers && (
                <div>
                  <h4 className="font-medium text-gray-800">Outliers:</h4>
                  <ul className="list-disc pl-5 text-gray-700">
                    {eda_summary.patterns_and_relationships.outliers.slice(0, 3).map((outlier, index) => (
                      <li key={index}>
                        {outlier.column} ({outlier.outlier_count} outliers)
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              
              {eda_summary.patterns_and_relationships.anomalies && (
                <div>
                  <h4 className="font-medium text-gray-800">Anomalies:</h4>
                  <ul className="list-disc pl-5 text-gray-700">
                    {eda_summary.patterns_and_relationships.anomalies.slice(0, 3).map((anomaly, index) => (
                      <li key={index}>
                        {anomaly.column} ({anomaly.anomaly_type})
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          ) : (
            <p className="text-gray-500">No patterns detected.</p>
          )}
        </div>

        <div className="mt-6 space-y-4">
          <h3 className="text-xl font-semibold text-gray-900">Recommendations</h3>
          {eda_summary.recommendations ? (
            <ul className="list-disc pl-5 space-y-1 text-gray-700">
              {eda_summary.recommendations.map((rec, index) => (
                <li key={index}>
                  {rec.title}: {rec.description}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-gray-500">No specific recommendations provided.</p>
          )}
        </div>
      </div>
    </section>
  );
};

export default EDATab;