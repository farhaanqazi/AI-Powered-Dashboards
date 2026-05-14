import React from 'react';
import Plotly from 'plotly.js-basic-dist';
import createPlotlyComponent from 'react-plotly.js/factory';

const Plot = createPlotlyComponent(Plotly);

const TRUNCATE_LEN = 18;

// Dark-theme palette aligned with .dash-shell futuristic theme
const DARK_THEME = {
  font: '#cbd5e1',
  tick: '#94a3b8',
  axisLine: 'rgba(148,163,184,0.20)',
  grid: 'rgba(148,163,184,0.10)',
  zero: 'rgba(148,163,184,0.25)',
  title: '#f1f5f9',
};

// Neon-friendly categorical palette
const CAT_PALETTE = ['#60a5fa', '#a78bfa', '#34d399', '#fbbf24', '#f472b6', '#22d3ee', '#fb923c', '#c4b5fd', '#5eead4', '#fda4af'];

const truncateLabel = (label, maxLen = 14) => {
  const str = String(label ?? '');
  if (str.length <= maxLen) return str;
  return `${str.slice(0, maxLen - 3)}...`;
};

const buildCategoryAxis = (labels) => {
  const full = labels.map(l => String(l ?? ''));
  const truncated = full.map(l => truncateLabel(l, TRUNCATE_LEN));
  const count = full.length;

  let angle = 0;
  let fontSize = 12;
  if (count > 12) {
    angle = -90;
    fontSize = 10;
  } else if (count > 6 || full.some(l => l.length > 10)) {
    angle = -45;
    fontSize = 11;
  }

  return {
    tickmode: 'array',
    tickvals: full,
    ticktext: truncated,
    tickangle: angle,
    tickfont: { size: fontSize, color: DARK_THEME.tick },
    automargin: true,
    linecolor: DARK_THEME.axisLine,
    gridcolor: DARK_THEME.grid,
    zerolinecolor: DARK_THEME.zero,
  };
};

const horizontalCategoryAxis = (labels) => {
  const full = labels.map(l => String(l ?? ''));
  const truncated = full.map(l => truncateLabel(l, TRUNCATE_LEN));
  return {
    tickmode: 'array',
    tickvals: full,
    ticktext: truncated,
    tickangle: 0,
    tickfont: { size: 11, color: DARK_THEME.tick },
    automargin: true,
    linecolor: DARK_THEME.axisLine,
    gridcolor: DARK_THEME.grid,
    zerolinecolor: DARK_THEME.zero,
  };
};

const shouldUseHorizontalBars = (labels) => {
  const count = labels.length;
  const hasLongLabels = labels.some(l => String(l ?? '').length > 15);
  return (count > 10 && hasLongLabels) || count > 20;
};

const shouldUseLogScale = (values) => {
  const nums = (values || []).map(Number).filter(v => Number.isFinite(v) && v > 0);
  if (nums.length < 20) return false;
  const sorted = [...nums].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)] || 1;
  const max = sorted[sorted.length - 1];
  if (median <= 0) return false;
  return max / median > 100;
};

const themedAxis = (extra = {}) => ({
  tickfont: { size: 11, color: DARK_THEME.tick },
  titlefont: { color: DARK_THEME.tick, size: 12 },
  linecolor: DARK_THEME.axisLine,
  gridcolor: DARK_THEME.grid,
  zerolinecolor: DARK_THEME.zero,
  automargin: true,
  ...extra,
});

const baseLayout = (title, extras = {}) => {
  const titleObj = typeof title === 'string'
    ? { text: title, font: { color: DARK_THEME.title, size: 14 } }
    : (title || undefined);

  // Merge axis defaults if caller supplied them
  const mergeAxis = (override) => {
    if (!override) return themedAxis();
    const { title: t, ...rest } = override;
    const merged = themedAxis(rest);
    if (t) {
      merged.title = typeof t === 'string' ? { text: t, font: { color: DARK_THEME.tick, size: 12 } } : t;
    }
    return merged;
  };

  const { xaxis, yaxis, ...restExtras } = extras;

  return {
    title: titleObj,
    height: 400,
    margin: { t: 50, b: 60, l: 60, r: 40 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: DARK_THEME.font, family: 'Inter, ui-sans-serif, system-ui, sans-serif' },
    colorway: CAT_PALETTE,
    hoverlabel: {
      bgcolor: 'rgba(15,23,42,0.92)',
      bordercolor: 'rgba(96,165,250,0.45)',
      font: { color: '#f1f5f9', size: 12 },
    },
    legend: {
      font: { color: DARK_THEME.font, size: 11 },
      bgcolor: 'rgba(15,23,42,0.0)',
      bordercolor: 'rgba(148,163,184,0.15)',
    },
    xaxis: mergeAxis(xaxis),
    yaxis: mergeAxis(yaxis),
    ...restExtras,
  };
};

const pickX = (item) => item.x ?? item.category ?? item.bin_range ?? item.date ?? '';
const pickY = (item) => item.y ?? item.count ?? item.value ?? item.agg_value ?? 0;

const renderCategoricalSeries = (xValues, yValues, { title, color, layoutExtras, xTitle, yTitle }) => {
  const useHorizontal = shouldUseHorizontalBars(xValues);
  const catAxisVertical = buildCategoryAxis(xValues);
  const catAxisHorizontal = horizontalCategoryAxis(xValues);
  const yIsLog = shouldUseLogScale(yValues);

  const data = [{
    x: useHorizontal ? yValues : xValues,
    y: useHorizontal ? xValues : yValues,
    type: 'bar',
    orientation: useHorizontal ? 'h' : 'v',
    marker: {
      color,
      line: { color: 'rgba(255,255,255,0.10)', width: 0.5 },
    },
    hovertext: xValues,
    hovertemplate: useHorizontal
      ? '%{y}<br>%{x:,}<extra></extra>'
      : '%{hovertext}<br>%{y:,}<extra></extra>',
  }];

  const layout = baseLayout(title, {
    ...layoutExtras,
    xaxis: useHorizontal
      ? { title: yTitle, type: yIsLog ? 'log' : 'linear' }
      : { title: xTitle, ...catAxisVertical },
    yaxis: useHorizontal
      ? { title: xTitle, ...catAxisHorizontal }
      : { title: yTitle, type: yIsLog ? 'log' : 'linear' },
  });

  return { data, layout };
};

const ChartRenderer = ({ chartData }) => {
  if (!chartData || chartData.data == null) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center p-8">
        <div className="text-slate-500 mb-4">
          <i className="fas fa-chart-bar text-4xl"></i>
        </div>
        <p className="text-slate-400">No chart data available</p>
      </div>
    );
  }

  const chartType = chartData.type || chartData.chart_type;
  const title = chartData.title || chartData.column;
  const dataIsArray = Array.isArray(chartData.data);
  const dataObj = !dataIsArray && typeof chartData.data === 'object' ? chartData.data : null;

  let plotlyData = [];
  let layout = chartData.layout || {};

  if (chartType === 'heatmap') {
    const src = dataObj || {};
    // Cool neon diverging scale for dark theme (rose -> deep -> emerald)
    const neonDiverging = [
      [0.0, '#f472b6'],
      [0.25, '#a78bfa'],
      [0.5, '#1e293b'],
      [0.75, '#22d3ee'],
      [1.0, '#34d399'],
    ];
    plotlyData = [{
      z: src.z || [],
      x: src.x || [],
      y: src.y || [],
      type: 'heatmap',
      colorscale: src.colorscale && src.colorscale !== 'RdBu' ? src.colorscale : neonDiverging,
      zmin: src.zmin,
      zmax: src.zmax,
      zmid: src.zmid,
      colorbar: {
        tickfont: { color: DARK_THEME.tick, size: 10 },
        outlinewidth: 0,
        thickness: 14,
      },
    }];
    layout = baseLayout(title, {
      ...layout,
      xaxis: { title: src.xaxis?.title || 'Variables', automargin: true },
      yaxis: { title: src.yaxis?.title || 'Variables', automargin: true },
    });
  }
  else if (chartType === 'bar' || chartType === 'category_count' || chartType === 'histogram'
           || chartType === 'distribution' || chartType === 'category_summary' || chartType === 'group_comparison') {
    let xValues = [];
    let yValues = [];
    if (dataIsArray) {
      xValues = chartData.data.map(pickX);
      yValues = chartData.data.map(pickY);
    } else if (dataObj) {
      xValues = dataObj.x || [];
      yValues = dataObj.y || [];
    }
    const xTitle = chartData.x_column || dataObj?.xaxis?.title || 'Category';
    const yTitle = chartData.y_column || dataObj?.yaxis?.title || (chartType === 'histogram' || chartType === 'distribution' ? 'Frequency' : 'Value');
    const color = chartType === 'histogram' || chartType === 'distribution' ? '#a78bfa' : '#60a5fa';
    ({ data: plotlyData, layout } = renderCategoricalSeries(xValues, yValues, {
      title, color, layoutExtras: layout, xTitle, yTitle,
    }));
  }
  else if (chartType === 'scatter') {
    let xValues = [];
    let yValues = [];
    if (dataIsArray) {
      xValues = chartData.data.map(item => item.x);
      yValues = chartData.data.map(item => item.y);
    } else if (dataObj) {
      xValues = dataObj.x || [];
      yValues = dataObj.y || [];
    }
    const yIsLog = shouldUseLogScale(yValues);
    plotlyData = [{
      x: xValues,
      y: yValues,
      type: 'scatter',
      mode: dataObj?.mode || 'markers',
      marker: {
        color: '#fbbf24',
        size: 7,
        opacity: 0.8,
        line: { color: 'rgba(15,23,42,0.8)', width: 1 },
      },
      hovertemplate: '%{x}<br>%{y}<extra></extra>',
    }];
    layout = baseLayout(title, {
      ...layout,
      xaxis: { title: chartData.x_column || dataObj?.xaxis?.title || 'X', automargin: true },
      yaxis: { title: chartData.y_column || dataObj?.yaxis?.title || 'Y', type: yIsLog ? 'log' : 'linear', automargin: true },
    });
  }
  else if (chartType === 'time_series' || chartType === 'line') {
    let xValues = [];
    let yValues = [];
    if (dataIsArray) {
      xValues = chartData.data.map(item => item.date ?? item.x);
      yValues = chartData.data.map(item => item.value ?? item.y);
    } else if (dataObj) {
      xValues = dataObj.x || [];
      yValues = dataObj.y || [];
    }
    const yIsLog = shouldUseLogScale(yValues);
    plotlyData = [{
      x: xValues,
      y: yValues,
      type: 'scatter',
      mode: xValues.length > 2000 ? 'lines' : 'lines+markers',
      line: { color: '#34d399', width: 2.5, shape: 'spline', smoothing: 0.6 },
      marker: { color: '#34d399', size: 6, line: { color: 'rgba(15,23,42,0.8)', width: 1 } },
      fill: 'tozeroy',
      fillcolor: 'rgba(52,211,153,0.10)',
      hovertemplate: '%{x}<br>%{y}<extra></extra>',
    }];
    layout = baseLayout(title, {
      ...layout,
      xaxis: { title: chartData.x_column || 'Date', automargin: true },
      yaxis: { title: chartData.y_column || 'Value', type: yIsLog ? 'log' : 'linear', automargin: true },
    });
  }
  else if (chartType === 'box' || chartType === 'box_plot') {
    const boxStyle = {
      marker: { color: '#fb7185', outliercolor: '#f43f5e', line: { width: 1, color: 'rgba(15,23,42,0.6)' } },
      line: { color: '#f87171' },
      fillcolor: 'rgba(248,113,113,0.18)',
      boxpoints: 'outliers',
    };
    if (dataIsArray && chartData.data.length > 0 && Array.isArray(chartData.data[0]?.values)) {
      plotlyData = chartData.data.map((group, idx) => ({
        y: group.values,
        type: 'box',
        name: String(group.category ?? ''),
        ...boxStyle,
        marker: { ...boxStyle.marker, color: CAT_PALETTE[idx % CAT_PALETTE.length] },
      }));
    } else if (dataIsArray) {
      const yVals = chartData.data.map(pickY);
      plotlyData = [{ y: yVals, type: 'box', ...boxStyle }];
    } else if (dataObj) {
      plotlyData = [{ y: dataObj.y || [], type: 'box', ...boxStyle }];
    }
    layout = baseLayout(title, {
      ...layout,
      xaxis: { title: chartData.x_column || dataObj?.xaxis?.title || '', automargin: true },
      yaxis: { title: chartData.y_column || dataObj?.yaxis?.title || 'Values', automargin: true },
      showlegend: plotlyData.length > 1,
    });
  }
  else if (chartType === 'pie') {
    let labels = [];
    let values = [];
    if (dataIsArray) {
      labels = chartData.data.map(item => item.category ?? item.label ?? item.name ?? '');
      values = chartData.data.map(item => item.value ?? item.count ?? 0);
    } else if (dataObj) {
      labels = dataObj.labels || [];
      values = dataObj.values || [];
    }
    plotlyData = [{
      labels,
      values,
      type: 'pie',
      hole: 0.55,
      textinfo: 'label+percent',
      textfont: { color: '#f1f5f9', size: 11 },
      marker: {
        colors: CAT_PALETTE,
        line: { color: 'rgba(2,6,23,0.85)', width: 2 },
      },
      hovertemplate: '%{label}<br>%{value} (%{percent})<extra></extra>',
    }];
    layout = baseLayout(title, { ...layout, margin: { t: 40, b: 40, l: 40, r: 40 } });
  }
  else {
    // Fallback: treat array-of-records as a bar chart
    if (dataIsArray && chartData.data.length > 0) {
      const xValues = chartData.data.map(pickX);
      const yValues = chartData.data.map(pickY);
      ({ data: plotlyData, layout } = renderCategoricalSeries(xValues, yValues, {
        title: title || 'Chart',
        color: '#6b7280',
        layoutExtras: layout,
        xTitle: 'X',
        yTitle: 'Y',
      }));
    } else {
      return (
        <div className="flex flex-col items-center justify-center h-full text-center p-8">
          <div className="text-slate-500 mb-4">
            <i className="fas fa-chart-bar text-4xl"></i>
          </div>
          <p className="text-slate-400">Unsupported chart type: {String(chartType)}</p>
        </div>
      );
    }
  }

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
          scale: 2,
        },
      }}
      style={{ width: '100%', height: '100%', overflow: 'hidden' }}
    />
  );
};

export default ChartRenderer;
