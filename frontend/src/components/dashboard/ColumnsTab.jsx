import React, { useState } from 'react';

const ColumnsTab = ({ data }) => {
  const { dataset_profile } = data;
  const [searchTerm, setSearchTerm] = useState('');
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });

  if (!dataset_profile || !dataset_profile.columns) {
    return (
      <section id="column_profiling-section" className="analysis-section">
        <div className="bg-gray-50 rounded-xl p-8 text-center border-2 border-dashed border-gray-200">
          <i className="fas fa-table text-gray-400 text-2xl mb-3"></i>
          <p className="text-gray-500">No column profiling data available.</p>
        </div>
      </section>
    );
  }

  // Filter columns based on search term
  const filteredColumns = dataset_profile.columns.filter(col =>
    col.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    col.role.toLowerCase().includes(searchTerm.toLowerCase()) ||
    col.dtype.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Sort columns based on sort configuration
  const sortedColumns = [...filteredColumns].sort((a, b) => {
    if (sortConfig.key) {
      const aValue = a[sortConfig.key];
      const bValue = b[sortConfig.key];

      if (aValue < bValue) {
        return sortConfig.direction === 'asc' ? -1 : 1;
      }
      if (aValue > bValue) {
        return sortConfig.direction === 'asc' ? 1 : -1;
      }
    }
    return 0;
  });

  const handleSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  const getRoleColor = (role) => {
    switch (role) {
      case 'numeric':
        return 'bg-blue-100 text-blue-800';
      case 'categorical':
        return 'bg-purple-100 text-purple-800';
      case 'datetime':
        return 'bg-green-100 text-green-800';
      case 'identifier':
        return 'bg-red-100 text-red-800';
      case 'text':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <section id="column_profiling-section" className="analysis-section">
      <div className="space-y-6">
        {/* Search and Filter */}
        <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-100">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <h2 className="text-lg font-semibold text-gray-900 flex items-center">
              <i className="fas fa-table text-blue-500 mr-2"></i> Column Profiling
            </h2>
            <div className="relative">
              <input
                type="text"
                placeholder="Search columns..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full md:w-64 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
              <i className="fas fa-search absolute right-3 top-3 text-gray-400"></i>
            </div>
          </div>
        </div>

        {/* Stats Summary */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-100">
            <div className="flex items-center">
              <div className="p-2 rounded-lg bg-blue-100 text-blue-600 mr-3">
                <i className="fas fa-table"></i>
              </div>
              <div>
                <p className="text-sm text-gray-500">Total Columns</p>
                <p className="text-xl font-bold text-gray-900">{dataset_profile.columns.length}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-100">
            <div className="flex items-center">
              <div className="p-2 rounded-lg bg-green-100 text-green-600 mr-3">
                <i className="fas fa-calculator"></i>
              </div>
              <div>
                <p className="text-sm text-gray-500">Numeric Fields</p>
                <p className="text-xl font-bold text-gray-900">
                  {dataset_profile.columns.filter(col => col.role === 'numeric').length}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-100">
            <div className="flex items-center">
              <div className="p-2 rounded-lg bg-purple-100 text-purple-600 mr-3">
                <i className="fas fa-tags"></i>
              </div>
              <div>
                <p className="text-sm text-gray-500">Categorical Fields</p>
                <p className="text-xl font-bold text-gray-900">
                  {dataset_profile.columns.filter(col => col.role === 'categorical').length}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-100">
            <div className="flex items-center">
              <div className="p-2 rounded-lg bg-red-100 text-red-600 mr-3">
                <i className="fas fa-fingerprint"></i>
              </div>
              <div>
                <p className="text-sm text-gray-500">Identifiers</p>
                <p className="text-xl font-bold text-gray-900">
                  {dataset_profile.columns.filter(col => col.role === 'identifier').length}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Columns Table */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSort('name')}
                  >
                    <div className="flex items-center">
                      Column Name
                      {sortConfig.key === 'name' && (
                        <i className={`ml-1 fas fa-${sortConfig.direction === 'asc' ? 'arrow-up' : 'arrow-down'}`}></i>
                      )}
                    </div>
                  </th>
                  <th
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSort('dtype')}
                  >
                    <div className="flex items-center">
                      Data Type
                      {sortConfig.key === 'dtype' && (
                        <i className={`ml-1 fas fa-${sortConfig.direction === 'asc' ? 'arrow-up' : 'arrow-down'}`}></i>
                      )}
                    </div>
                  </th>
                  <th
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSort('missing_count')}
                  >
                    <div className="flex items-center">
                      Missing Values
                      {sortConfig.key === 'missing_count' && (
                        <i className={`ml-1 fas fa-${sortConfig.direction === 'asc' ? 'arrow-up' : 'arrow-down'}`}></i>
                      )}
                    </div>
                  </th>
                  <th
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSort('unique_count')}
                  >
                    <div className="flex items-center">
                      Unique Values
                      {sortConfig.key === 'unique_count' && (
                        <i className={`ml-1 fas fa-${sortConfig.direction === 'asc' ? 'arrow-up' : 'arrow-down'}`}></i>
                      )}
                    </div>
                  </th>
                  <th
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSort('role')}
                  >
                    <div className="flex items-center">
                      Role
                      {sortConfig.key === 'role' && (
                        <i className={`ml-1 fas fa-${sortConfig.direction === 'asc' ? 'arrow-up' : 'arrow-down'}`}></i>
                      )}
                    </div>
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Min
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Max
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Mean
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Top Categories
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {sortedColumns.map((col, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">{col.name}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900 font-mono bg-gray-100 px-2 py-1 rounded">{col.dtype}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900">{col.missing_count}</div>
                      <div className="text-xs text-gray-500">
                        {col.missing_count > 0 ? `${((col.missing_count / dataset_profile.n_rows) * 100).toFixed(1)}%` : ''}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900">{col.unique_count}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getRoleColor(col.role)}`}>
                        {col.role}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {col.stats && col.stats.min !== undefined ? col.stats.min : 'N/A'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {col.stats && col.stats.max !== undefined ? col.stats.max : 'N/A'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {col.stats && col.stats.mean !== undefined ? col.stats.mean.toFixed(2) : 'N/A'}
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex flex-wrap gap-1">
                        {col.top_categories && col.top_categories.length > 0 ? (
                          col.top_categories.slice(0, 3).map((cat, catIndex) => (
                            <span key={catIndex} className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800 truncate max-w-xs">
                              {cat.value?.toString().substring(0, 15)}... ({cat.count})
                            </span>
                          ))
                        ) : (
                          <span className="text-xs text-gray-500">N/A</span>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {filteredColumns.length === 0 && (
          <div className="bg-gray-50 rounded-xl p-8 text-center border-2 border-dashed border-gray-200">
            <i className="fas fa-search text-gray-400 text-2xl mb-3"></i>
            <p className="text-gray-500">No columns match your search criteria.</p>
          </div>
        )}
      </div>
    </section>
  );
};

export default ColumnsTab;