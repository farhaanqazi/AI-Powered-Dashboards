import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-basic-dist';
import createPlotlyComponent from 'react-plotly.js/factory';

const Plot = createPlotlyComponent(Plotly);

const ChartRenderer = ({ chartData }) => {
  if (!chartData || !chartData.data) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center p-8">
        <div className="text-gray-400 mb-4">
          <i className="fas fa-chart-bar text-4xl"></i>
        </div>
        <p className="text-gray-500">No chart data available</p>
      </div>
    );
  }

  // Convert the backend data format to Plotly-compatible format
  let plotlyData = [];
  let layout = chartData.layout || {};

  // Determine chart type and convert data accordingly
  const chartType = chartData.type || chartData.chart_type;
  const title = chartData.title || chartData.column;

  // Check if chartData has the structure we're sending from VisualizationsTab
  if (chartType === 'heatmap') {
    // For heatmap charts
    plotlyData = [{
      z: chartData.data.z || [],
      x: chartData.data.x || [],
      y: chartData.data.y || [],
      type: 'heatmap',
      colorscale: chartData.data.colorscale || 'Viridis'
    }];

    layout = {
      ...layout,
      title: title,
      xaxis: { title: chartData.data.xaxis?.title || 'Variables' },
      yaxis: { title: chartData.data.yaxis?.title || 'Variables' },
      height: 400,
      margin: { t: 40, b: 60, l: 60, r: 40 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent'
    };
  }
  else if (chartType === 'bar') {
    // For bar charts
    plotlyData = [{
      x: chartData.data.x || [],
      y: chartData.data.y || [],
      type: 'bar',
      marker: { color: '#3b82f6' }
    }];

    layout = {
      ...layout,
      title: title,
      xaxis: { title: chartData.data.xaxis?.title || 'Categories' },
      yaxis: { title: chartData.data.yaxis?.title || 'Values' },
      height: 400,
      margin: { t: 40, b: 60, l: 60, r: 40 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent'
    };
  }
  else if (chartType === 'scatter') {
    // For scatter plots
    plotlyData = [{
      x: chartData.data.x || [],
      y: chartData.data.y || [],
      type: 'scatter',
      mode: chartData.data.mode || 'markers',
      marker: { color: '#f59e0b' }
    }];

    layout = {
      ...layout,
      title: title,
      xaxis: { title: chartData.data.xaxis?.title || 'X Values' },
      yaxis: { title: chartData.data.yaxis?.title || 'Y Values' },
      height: 400,
      margin: { t: 40, b: 60, l: 60, r: 40 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent'
    };
  }
  else if (chartType === 'box') {
    // For box plots
    plotlyData = [{
      y: chartData.data.y || [],
      type: 'box'
    }];

    layout = {
      ...layout,
      title: title,
      yaxis: { title: chartData.data.yaxis?.title || 'Values' },
      height: 400,
      margin: { t: 40, b: 60, l: 60, r: 40 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent'
    };
  }
  else if (chartType === 'pie') {
    // For pie charts
    plotlyData = [{
      labels: chartData.data.labels || [],
      values: chartData.data.values || [],
      type: 'pie'
    }];

    layout = {
      ...layout,
      title: title,
      height: 400,
      margin: { t: 40, b: 40, l: 40, r: 40 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent'
    };
  }
  else if (chartData.data && Array.isArray(chartData.data) && chartData.data.length > 0) {
    // Original logic for backward compatibility
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
        height: 400,
        margin: { t: 40, b: 60, l: 60, r: 40 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent'
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
        height: 400,
        margin: { t: 40, b: 60, l: 60, r: 40 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent'
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
        height: 400,
        margin: { t: 40, b: 60, l: 60, r: 40 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent'
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
        height: 400,
        margin: { t: 40, b: 60, l: 60, r: 40 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent'
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
        height: 400,
        margin: { t: 40, b: 60, l: 60, r: 40 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent'
      };
    }
  }
  else {
    // If no data array but we have x, y, z properties directly in chartData.data
    if (chartData.data && typeof chartData.data === 'object') {
      if (chartData.data.type === 'heatmap') {
        plotlyData = [{
          z: chartData.data.z || [],
          x: chartData.data.x || [],
          y: chartData.data.y || [],
          type: 'heatmap',
          colorscale: chartData.data.colorscale || 'Viridis'
        }];

        layout = {
          ...layout,
          title: title,
          xaxis: { title: chartData.data.xaxis?.title || 'Variables' },
          yaxis: { title: chartData.data.yaxis?.title || 'Variables' },
          height: 400,
          margin: { t: 40, b: 60, l: 60, r: 40 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent'
        };
      }
      else if (chartData.data.type === 'bar') {
        plotlyData = [{
          x: chartData.data.x || [],
          y: chartData.data.y || [],
          type: 'bar',
          marker: { color: '#3b82f6' }
        }];

        layout = {
          ...layout,
          title: title,
          xaxis: { title: chartData.data.xaxis?.title || 'Categories' },
          yaxis: { title: chartData.data.yaxis?.title || 'Values' },
          height: 400,
          margin: { t: 40, b: 60, l: 60, r: 40 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent'
        };
      }
      else if (chartData.data.type === 'scatter') {
        plotlyData = [{
          x: chartData.data.x || [],
          y: chartData.data.y || [],
          type: 'scatter',
          mode: chartData.data.mode || 'markers',
          marker: { color: '#f59e0b' }
        }];

        layout = {
          ...layout,
          title: title,
          xaxis: { title: chartData.data.xaxis?.title || 'X Values' },
          yaxis: { title: chartData.data.yaxis?.title || 'Y Values' },
          height: 400,
          margin: { t: 40, b: 60, l: 60, r: 40 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent'
        };
      }
      else if (chartData.data.type === 'box') {
        plotlyData = [{
          y: chartData.data.y || [],
          type: 'box'
        }];

        layout = {
          ...layout,
          title: title,
          yaxis: { title: chartData.data.yaxis?.title || 'Values' },
          height: 400,
          margin: { t: 40, b: 60, l: 60, r: 40 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent'
        };
      }
      else if (chartData.data.type === 'pie') {
        plotlyData = [{
          labels: chartData.data.labels || [],
          values: chartData.data.values || [],
          type: 'pie'
        }];

        layout = {
          ...layout,
          title: title,
          height: 400,
          margin: { t: 40, b: 40, l: 40, r: 40 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent'
        };
      }
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