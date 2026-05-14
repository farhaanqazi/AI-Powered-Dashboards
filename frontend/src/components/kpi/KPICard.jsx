import React from 'react';

const KPICard = ({ kpi }) => {
  return (
    <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-3 border border-blue-100">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-medium text-gray-600 truncate max-w-xs">{kpi.label}</p>
          <p className="text-sm font-bold text-gray-900">{kpi.value}</p>
        </div>
        <div className="p-1 rounded bg-blue-100 text-blue-600">
          <i className="fas fa-chart-line text-xs"></i>
        </div>
      </div>
    </div>
  );
};

export default KPICard;