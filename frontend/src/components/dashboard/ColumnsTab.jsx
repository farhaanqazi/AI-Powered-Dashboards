import React from 'react';

const ColumnsTab = ({ data }) => {
  const { dataset_profile } = data;

  if (!dataset_profile || !dataset_profile.columns) {
    return (
      <section id="column_profiling-section" className="analysis-section">
        <div className="light-card rounded-3xl p-6">
          <p className="text-gray-500">No column profiling data available.</p>
        </div>
      </section>
    );
  }

  return (
    <section id="column_profiling-section" className="analysis-section">
      <div className="light-card rounded-3xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-semibold text-gray-900">Column Profiling</h2>
          <span className="badge badge-soft">Schema</span>
        </div>
        <div className="overflow-x-auto">
          <table className="table table-zebra text-sm w-full">
            <thead className="text-gray-500 bg-gray-50">
              <tr>
                <th>Column Name</th>
                <th>Data Type</th>
                <th>Missing Values</th>
                <th>Unique Values</th>
                <th>Role</th>
                <th>Min</th>
                <th>Max</th>
                <th>Mean</th>
                <th>Top Categories</th>
              </tr>
            </thead>
            <tbody className="text-gray-700">
              {dataset_profile.columns.map((col, index) => (
                <tr key={index}>
                  <td className="font-medium">{col.name}</td>
                  <td>{col.dtype}</td>
                  <td>{col.missing_count}</td>
                  <td>{col.unique_count}</td>
                  <td>
                    <span className="badge badge-xs badge-soft text-xs">{col.role}</span>
                  </td>
                  <td>{col.stats && col.stats.min !== undefined ? col.stats.min : 'N/A'}</td>
                  <td>{col.stats && col.stats.max !== undefined ? col.stats.max : 'N/A'}</td>
                  <td>{col.stats && col.stats.mean !== undefined ? col.stats.mean.toFixed(2) : 'N/A'}</td>
                  <td>
                    {col.top_categories && col.top_categories.length > 0 ? (
                      col.top_categories.slice(0, 3).map((cat, catIndex) => (
                        <span key={catIndex} className="badge badge-xs badge-soft mr-1">
                          {cat.value?.toString().substring(0, 10)}... ({cat.count})
                        </span>
                      ))
                    ) : (
                      'N/A'
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
};

export default ColumnsTab;