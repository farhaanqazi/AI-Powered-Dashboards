import React from 'react';

const KPICard = ({ kpi }) => {
  return (
    <div 
      className="badge badge-soft kpi-badge text-gray-700 bg-gray-100 border border-gray-200"
      data-kpi-column={kpi.label}
    >
      <span className="font-semibold">{kpi.label}:</span> {kpi.value}
    </div>
  );
};

export default KPICard;