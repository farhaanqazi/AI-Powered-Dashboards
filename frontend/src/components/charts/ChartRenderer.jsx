import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-basic-dist';
import createPlotlyComponent from 'react-plotly.js/factory';

const Plot = createPlotlyComponent(Plotly);

const ChartRenderer = ({ chartData }) => {
  if (!chartData || !chartData.data) {
    return <div className="text-center py-10 text-gray-500">No chart data available</div>;
  }

  // If using the Plotly React component
  return (
    <Plot
      data={[chartData.data]}
      layout={chartData.layout || {}}
      config={{ displayModeBar: true, responsive: true }}
      style={{ width: '100%', height: '100%' }}
    />
  );
};

export default ChartRenderer;