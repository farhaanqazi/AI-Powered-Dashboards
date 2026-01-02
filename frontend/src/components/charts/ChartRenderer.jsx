import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-basic-dist';
import createPlotlyComponent from 'react-plotly.js/factory';

const Plot = createPlotlyComponent(Plotly);

const ChartRenderer = ({ chartData }) => {
  if (!chartData || !chartData.data) {
    return <div className="text-center py-10 text-gray-500">No chart data available</div>;
  }

  // Convert the backend data format to Plotly-compatible format
  let plotlyData = [];
  let layout = chartData.layout || {};

  // Determine chart type and convert data accordingly
  const chartType = chartData.type || chartData.chart_type;
  const title = chartData.title || chartData.column;

  if (chartData.data && Array.isArray(chartData.data) && chartData.data.length > 0) {
    if (chartType === 'category_count' || chartType === 'bar') {
      // For bar charts (categories and counts)
      const xValues = chartData.data.map(item => item.category || item.bin_range || item.x);
      const yValues = chartData.data.map(item => item.count || item.value || item.y);

      plotlyData = [{
        x: xValues,
        y: yValues,
        type: 'bar',
        marker: { color: '#3b82f6' }
      }];

      layout = {
        ...layout,
        title: title,
        xaxis: { title: chartData.x_column || 'Category' },
        yaxis: { title: 'Count' },
        height: 400
      };
    }
    else if (chartType === 'histogram' || chartType === 'distribution') {
      // For histograms (ranges and counts)
      const xValues = chartData.data.map(item => item.bin_range || item.category || item.x);
      const yValues = chartData.data.map(item => item.count || item.value || item.y);

      plotlyData = [{
        x: xValues,
        y: yValues,
        type: 'bar',
        marker: { color: '#8b5cf6' }
      }];

      layout = {
        ...layout,
        title: title,
        xaxis: { title: chartData.x_column || 'Range' },
        yaxis: { title: 'Frequency' },
        height: 400
      };
    }
    else if (chartType === 'time_series' || chartType === 'line') {
      // For time series (dates and values)
      const xValues = chartData.data.map(item => item.date || item.x);
      const yValues = chartData.data.map(item => item.value || item.y);

      plotlyData = [{
        x: xValues,
        y: yValues,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#10b981' }
      }];

      layout = {
        ...layout,
        title: title,
        xaxis: { title: chartData.x_column || 'Date' },
        yaxis: { title: chartData.y_column || 'Value' },
        height: 400
      };
    }
    else if (chartType === 'scatter') {
      // For scatter plots (x and y values)
      const xValues = chartData.data.map(item => item.x);
      const yValues = chartData.data.map(item => item.y);

      plotlyData = [{
        x: xValues,
        y: yValues,
        type: 'scatter',
        mode: 'markers',
        marker: { color: '#f59e0b' }
      }];

      layout = {
        ...layout,
        title: title,
        xaxis: { title: chartData.x_column || 'X Value' },
        yaxis: { title: chartData.y_column || 'Y Value' },
        height: 400
      };
    }
    else {
      // Default case - try to render as bar chart
      const xValues = chartData.data.map(item => item.category || item.bin_range || item.date || item.x || '');
      const yValues = chartData.data.map(item => item.count || item.value || item.y || 0);

      plotlyData = [{
        x: xValues,
        y: yValues,
        type: 'bar',
        marker: { color: '#6b7280' }
      }];

      layout = {
        ...layout,
        title: title || 'Chart',
        xaxis: { title: 'X' },
        yaxis: { title: 'Y' },
        height: 400
      };
    }
  }

  // If using the Plotly React component
  return (
    <Plot
      data={plotlyData}
      layout={layout}
      config={{
        displayModeBar: true,
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d', 'resetScale2d'],
        toImageButtonOptions: {
          format: 'png',
          filename: 'chart',
          height: 600,
          width: 800,
          scale: 2
        }
      }}
      style={{ width: '100%', height: '100%' }}
    />
  );
};

export default ChartRenderer;