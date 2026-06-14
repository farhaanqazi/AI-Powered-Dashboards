import React from 'react';
import Plotly from 'plotly.js-basic-dist';
import createPlotlyComponent from 'react-plotly.js/factory';
import { useDashboardStore } from '../../dashboardStore';
import { highlightOpacity } from '../../lib/clientFilter';

const Plot = createPlotlyComponent(Plotly);

// Categorical chart types that participate in cross-highlight (clicking a
// mark focuses that category across charts on the same dimension).
const HIGHLIGHTABLE = new Set([
  'bar', 'category_count', 'category_summary', 'group_comparison', 'pie',
]);

const TRUNCATE_LEN = 18;

// Plotly palettes that mirror the two .dash-shell themes. The dark variant is
// aligned with the futuristic dark theme; the light variant is its
// contrast-matched counterpart for the white theme.
const THEMES = {
  dark: {
    font: '#cbd5e1',
    tick: '#94a3b8',
    axisLine: 'rgba(148,163,184,0.20)',
    grid: 'rgba(148,163,184,0.10)',
    zero: 'rgba(148,163,184,0.25)',
    title: '#f1f5f9',
    markerLine: 'rgba(255,255,255,0.10)',
    pointLine: 'rgba(15,23,42,0.8)',
    pieLine: 'rgba(2,6,23,0.85)',
    pieText: '#f1f5f9',
    hoverBg: 'rgba(15,23,42,0.92)',
    hoverBorder: 'rgba(96,165,250,0.45)',
    hoverFont: '#f1f5f9',
    legendBorder: 'rgba(148,163,184,0.15)',
    heatMid: '#1e293b',
  },
  light: {
    font: '#334155',
    tick: '#475569',
    axisLine: 'rgba(71,85,105,0.30)',
    grid: 'rgba(71,85,105,0.12)',
    zero: 'rgba(71,85,105,0.30)',
    title: '#0f172a',
    markerLine: 'rgba(15,23,42,0.10)',
    pointLine: 'rgba(255,255,255,0.85)',
    pieLine: 'rgba(255,255,255,0.9)',
    pieText: '#0f172a',
    hoverBg: 'rgba(255,255,255,0.96)',
    hoverBorder: 'rgba(37,99,235,0.45)',
    hoverFont: '#0f172a',
    legendBorder: 'rgba(71,85,105,0.20)',
    heatMid: '#e2e8f0',
  },
};

// Neon-friendly categorical palette
const CAT_PALETTE = ['#60a5fa', '#a78bfa', '#34d399', '#fbbf24', '#f472b6', '#22d3ee', '#fb923c', '#c4b5fd', '#5eead4', '#fda4af'];

// Phase 16 S16.1 — per-intent smart colour defaults. A chart's section/intent
// drives its accent so the dashboard reads as a coherent system (Breakdowns
// blue, Distributions violet, Trends cyan, Relationships amber, Predictions
// emerald) instead of every chart defaulting to the same hue.
const INTENT_COLOR = {
  category_count: '#60a5fa',
  category_summary: '#60a5fa',
  group_comparison: '#38bdf8',
  histogram: '#a78bfa',
  distribution: '#a78bfa',
  time_series: '#22d3ee',
  scatter: '#fbbf24',
  feature_importance: '#34d399',
};

const intentColor = (chartData, fallback = '#60a5fa') =>
  INTENT_COLOR[chartData?.intent || chartData?.type] || fallback;

// Compact, locale-aware number formatting for on-chart value labels
// (1.2k / 3.4M) so labels stay short and the data-ink ratio stays high.
const fmtCompact = (v) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return '';
  const abs = Math.abs(n);
  if (abs >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  if (Number.isInteger(n)) return n.toLocaleString();
  return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
};

// Direct value labels stay on only when the axis isn't crowded — past this the
// labels overlap and hurt readability (data-ink discipline, S16.3).
const VALUE_LABEL_MAX_BARS = 12;

const truncateLabel = (label, maxLen = 14) => {
  const str = String(label ?? '');
  if (str.length <= maxLen) return str;
  return `${str.slice(0, maxLen - 3)}...`;
};

const buildCategoryAxis = (labels, T) => {
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
    tickfont: { size: fontSize, color: T.tick },
    automargin: true,
    linecolor: T.axisLine,
    gridcolor: T.grid,
    zerolinecolor: T.zero,
  };
};

const horizontalCategoryAxis = (labels, T) => {
  const full = labels.map(l => String(l ?? ''));
  const truncated = full.map(l => truncateLabel(l, TRUNCATE_LEN));
  return {
    tickmode: 'array',
    tickvals: full,
    ticktext: truncated,
    tickangle: 0,
    tickfont: { size: 11, color: T.tick },
    automargin: true,
    linecolor: T.axisLine,
    gridcolor: T.grid,
    zerolinecolor: T.zero,
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

const themedAxis = (extra = {}, T) => ({
  tickfont: { size: 11, color: T.tick },
  titlefont: { color: T.tick, size: 12 },
  linecolor: T.axisLine,
  gridcolor: T.grid,
  zerolinecolor: T.zero,
  automargin: true,
  ...extra,
});

// Wrap a long axis title onto multiple lines so the rotated y-axis label
// isn't clipped at the container edge.
const wrapAxisTitle = (text, maxLen = 18) => {
  const str = String(text ?? '');
  if (str.length <= maxLen) return str;
  const words = str.split(/[\s_]+/);
  const lines = [];
  let line = '';
  for (const w of words) {
    if (line && (line.length + w.length + 1) > maxLen) {
      lines.push(line);
      line = w;
    } else {
      line = line ? `${line} ${w}` : w;
    }
  }
  if (line) lines.push(line);
  return lines.join('<br>');
};

const baseLayout = (_title, extras = {}, T) => {
  // The chart title is intentionally NOT rendered inside the Plotly figure —
  // every chart is wrapped in a card that already shows the title as a heading.
  // Rendering it here too caused a duplicated title.

  // Merge axis defaults if caller supplied them
  const mergeAxis = (override) => {
    if (!override) return themedAxis({}, T);
    const { title: t, ...rest } = override;
    const merged = themedAxis(rest, T);
    if (t) {
      const titleText = typeof t === 'string' ? t : t.text;
      merged.title = {
        text: wrapAxisTitle(titleText),
        font: { color: T.tick, size: 11 },
        standoff: 12,
        ...(typeof t === 'object' ? { ...t, text: wrapAxisTitle(t.text) } : {}),
      };
    }
    return merged;
  };

  const { xaxis, yaxis, ...restExtras } = extras;

  return {
    title: undefined,
    height: 400,
    margin: { t: 24, b: 64, l: 72, r: 36 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: T.font, family: 'Inter, ui-sans-serif, system-ui, sans-serif' },
    colorway: CAT_PALETTE,
    hoverlabel: {
      bgcolor: T.hoverBg,
      bordercolor: T.hoverBorder,
      font: { color: T.hoverFont, size: 12 },
    },
    legend: {
      font: { color: T.font, size: 11 },
      bgcolor: 'rgba(0,0,0,0)',
      bordercolor: T.legendBorder,
    },
    xaxis: mergeAxis(xaxis),
    yaxis: mergeAxis(yaxis),
    ...restExtras,
  };
};

const pickX = (item) => item.x ?? item.category ?? item.bin_range ?? item.date ?? '';
const pickY = (item) => item.y ?? item.count ?? item.value ?? item.agg_value ?? 0;

const mean = (vals) => {
  const nums = (vals || []).map(Number).filter(Number.isFinite);
  return nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : null;
};

// S16.2 — ordinary least-squares fit of y over the point index (0..n-1) for a
// deterministic trendline overlay. Returns null when there's too little signal.
const linearFit = (yValues) => {
  const ys = (yValues || []).map(Number);
  const idx = ys.map((y, i) => (Number.isFinite(y) ? i : null)).filter((v) => v != null);
  if (idx.length < 5) return null;
  const xs = idx;
  const ysv = idx.map((i) => ys[i]);
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ysv.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (xs[i] - mx) * (ysv[i] - my);
    den += (xs[i] - mx) ** 2;
  }
  if (den === 0) return null;
  const slope = num / den;
  return { slope, intercept: my - slope * mx };
};

// S16.2 — a dashed "average" reference line so every bar is read against the
// mean. Built as a layout shape + annotation (works in plotly basic-dist).
const meanReferenceShapes = (yValues, { useHorizontal, yIsLog, T }) => {
  const m = mean(yValues);
  if (m == null || yIsLog || yValues.length < 3) return { shapes: [], annotations: [] };
  const axisProps = useHorizontal
    ? { xref: 'x', x0: m, x1: m, yref: 'paper', y0: 0, y1: 1 }
    : { yref: 'y', y0: m, y1: m, xref: 'paper', x0: 0, x1: 1 };
  const annPos = useHorizontal
    ? { xref: 'x', x: m, yref: 'paper', y: 1, xanchor: 'left', yanchor: 'bottom' }
    : { xref: 'paper', x: 1, yref: 'y', y: m, xanchor: 'right', yanchor: 'bottom' };
  return {
    shapes: [{ type: 'line', ...axisProps,
      line: { color: T.tick, width: 1.25, dash: 'dot' }, layer: 'below' }],
    annotations: [{ ...annPos, text: `avg ${fmtCompact(m)}`, showarrow: false,
      font: { color: T.tick, size: 10 } }],
  };
};

const renderCategoricalSeries = (xValues, yValues, { title, color, layoutExtras, xTitle, yTitle, T, markerOpacity, showLabels = true, meanReference = false }) => {
  const useHorizontal = shouldUseHorizontalBars(xValues);
  const catAxisVertical = buildCategoryAxis(xValues, T);
  const catAxisHorizontal = horizontalCategoryAxis(xValues, T);
  const yIsLog = shouldUseLogScale(yValues);

  // S16.1 — direct value labels when the axis isn't crowded (and not log-scaled,
  // where an absolute label on a compressed axis misleads).
  const labelled = showLabels && !yIsLog && xValues.length <= VALUE_LABEL_MAX_BARS;
  const labels = labelled ? yValues.map(fmtCompact) : undefined;

  const data = [{
    x: useHorizontal ? yValues : xValues,
    y: useHorizontal ? xValues : yValues,
    type: 'bar',
    orientation: useHorizontal ? 'h' : 'v',
    marker: {
      color,
      // Per-bar opacity drives cross-highlight dimming. `undefined` (no active
      // highlight on this dimension) leaves the prior full-opacity render intact.
      opacity: markerOpacity || undefined,
      line: { color: T.markerLine, width: 0.5 },
      cornerradius: 5,  // S16.1 — soft rounded bars
    },
    text: labels,
    texttemplate: labelled ? '%{text}' : undefined,
    textposition: labelled ? 'outside' : undefined,
    textfont: labelled ? { color: T.tick, size: 11, family: 'Inter, ui-sans-serif, system-ui, sans-serif' } : undefined,
    cliponaxis: false,
    hovertext: xValues,
    hovertemplate: useHorizontal
      ? '%{y}<br>%{x:,}<extra></extra>'
      : '%{hovertext}<br>%{y:,}<extra></extra>',
  }];

  const ref = meanReference
    ? meanReferenceShapes(yValues, { useHorizontal, yIsLog, T })
    : { shapes: [], annotations: [] };

  const layout = baseLayout(title, {
    ...layoutExtras,
    bargap: 0.28,  // S16.1 — airier spacing
    shapes: [...(layoutExtras?.shapes || []), ...ref.shapes],
    annotations: [...(layoutExtras?.annotations || []), ...ref.annotations],
    xaxis: useHorizontal
      ? { title: yTitle, type: yIsLog ? 'log' : 'linear' }
      : { title: xTitle, ...catAxisVertical },
    yaxis: useHorizontal
      ? { title: xTitle, ...catAxisHorizontal }
      : { title: yTitle, type: yIsLog ? 'log' : 'linear' },
  }, T);

  return { data, layout };
};

const ChartRenderer = ({ chartData, interactive = false }) => {
  const themeMode = useDashboardStore((s) => s.theme);
  const highlight = useDashboardStore((s) => s.highlight);
  const setHighlight = useDashboardStore((s) => s.setHighlight);
  const T = THEMES[themeMode] || THEMES.dark;

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
      [0.5, T.heatMid],
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
        tickfont: { color: T.tick, size: 10 },
        outlinewidth: 0,
        thickness: 14,
      },
    }];
    layout = baseLayout(title, {
      ...layout,
      xaxis: { title: src.xaxis?.title || 'Variables', automargin: true },
      yaxis: { title: src.yaxis?.title || 'Variables', automargin: true },
    }, T);
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
    const xTitle = chartData.x_title || chartData.x_column || dataObj?.xaxis?.title || 'Category';
    const yTitle = chartData.y_title || chartData.y_column || dataObj?.yaxis?.title || (chartType === 'histogram' || chartType === 'distribution' ? 'Frequency' : 'Value');
    // S16.1 — colour by the chart's intent (falls back to type) so the section
    // it belongs to reads as a coherent colour family.
    const color = intentColor(chartData, chartType === 'histogram' || chartType === 'distribution' ? '#a78bfa' : '#60a5fa');
    // Cross-highlight dimming applies only to true category axes — never to
    // histogram/distribution bins (their x is a numeric range, not a key).
    const isCategorical = chartType !== 'histogram' && chartType !== 'distribution';
    const markerOpacity = isCategorical
      ? highlightOpacity(xValues, chartData.x_column || chartData.column, highlight)
      : null;
    ({ data: plotlyData, layout } = renderCategoricalSeries(xValues, yValues, {
      title, color, layoutExtras: layout, xTitle, yTitle, T, markerOpacity,
      // Bins are dense and numeric — direct labels there clutter; keep them for
      // true categories only (S16.3 data-ink).
      showLabels: isCategorical,
      // S16.2 — average reference line on true category comparisons only.
      meanReference: isCategorical,
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
    // A `group` key on each point (e.g. cluster segments) splits the cloud into
    // one colour-coded trace per group; otherwise it's a single gold series.
    const groups = dataIsArray ? chartData.data.map(it => it.group).filter(g => g != null) : [];
    if (groups.length) {
      const palette = ['#60a5fa', '#f472b6', '#34d399', '#fbbf24', '#a78bfa', '#22d3ee', '#fb7185'];
      const uniq = Array.from(new Set(chartData.data.map(it => it.group)));
      plotlyData = uniq.map((g, i) => {
        const pts = chartData.data.filter(it => it.group === g);
        return {
          x: pts.map(p => p.x), y: pts.map(p => p.y), name: String(g),
          type: 'scatter', mode: 'markers',
          marker: { color: palette[i % palette.length], size: 7, opacity: 0.78,
            line: { color: T.pointLine, width: 1 } },
          hovertemplate: `${g}<br>%{x}, %{y}<extra></extra>`,
        };
      });
    } else {
      plotlyData = [{
        x: xValues,
        y: yValues,
        type: 'scatter',
        mode: dataObj?.mode || 'markers',
        marker: {
          color: '#fbbf24',
          size: 7,
          opacity: 0.8,
          line: { color: T.pointLine, width: 1 },
        },
        hovertemplate: '%{x}<br>%{y}<extra></extra>',
      }];
    }
    layout = baseLayout(title, {
      ...layout,
      xaxis: { title: chartData.x_title || chartData.x_column || dataObj?.xaxis?.title || 'X', automargin: true },
      yaxis: { title: chartData.y_title || chartData.y_column || dataObj?.yaxis?.title || 'Y', type: yIsLog ? 'log' : 'linear', automargin: true },
    }, T);
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
    const tsColor = intentColor(chartData, '#34d399');
    plotlyData = [{
      x: xValues,
      y: yValues,
      type: 'scatter',
      mode: xValues.length > 2000 ? 'lines' : 'lines+markers',
      name: chartData.y_column || 'Value',
      line: { color: tsColor, width: 2.5, shape: 'spline', smoothing: 0.6 },
      marker: { color: tsColor, size: 6, line: { color: T.pointLine, width: 1 } },
      fill: 'tozeroy',
      fillcolor: 'rgba(52,211,153,0.10)',
      hovertemplate: '%{x}<br>%{y}<extra></extra>',
    }];
    // S16.2 — least-squares trendline overlay so the underlying direction reads
    // through noisy series (annotated trend lines). Skip for forecasts (the
    // forecast already carries its own projection) and very short series.
    const fit = chartData.intent === 'forecast' ? null : linearFit(yValues);
    if (fit) {
      plotlyData.push({
        x: xValues,
        y: xValues.map((_, i) => fit.intercept + fit.slope * i),
        type: 'scatter', mode: 'lines', name: 'Trend',
        line: { color: T.tick, width: 1.5, dash: 'dash' },
        hoverinfo: 'skip',
      });
    }
    layout = baseLayout(title, {
      ...layout,
      showlegend: !!fit,
      legend: { orientation: 'h', y: 1.08, x: 1, xanchor: 'right', font: { color: T.font, size: 10 } },
      xaxis: { title: chartData.x_title || chartData.x_column || 'Date', automargin: true },
      yaxis: { title: chartData.y_title || chartData.y_column || 'Value', type: yIsLog ? 'log' : 'linear', automargin: true },
    }, T);
  }
  else if (chartType === 'box' || chartType === 'box_plot') {
    const boxStyle = {
      marker: { color: '#fb7185', outliercolor: '#f43f5e', line: { width: 1, color: T.pointLine } },
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
      xaxis: { title: chartData.x_title || chartData.x_column || dataObj?.xaxis?.title || '', automargin: true },
      yaxis: { title: chartData.y_title || chartData.y_column || dataObj?.yaxis?.title || 'Values', automargin: true },
      showlegend: plotlyData.length > 1,
    }, T);
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
      textfont: { color: T.pieText, size: 11 },
      marker: {
        colors: CAT_PALETTE,
        line: { color: T.pieLine, width: 2 },
      },
      hovertemplate: '%{label}<br>%{value} (%{percent})<extra></extra>',
    }];
    layout = baseLayout(title, { ...layout, margin: { t: 40, b: 40, l: 40, r: 40 } }, T);
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
        T,
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
      onClick={interactive && HIGHLIGHTABLE.has(chartType) ? (e) => {
        // Click a mark → focus that category across charts on the same
        // dimension (toggles off when the same mark is clicked again).
        // Client-side only: it restyles already-shipped data, no server call.
        const pt = e?.points?.[0];
        const col = chartData.x_column || chartData.column;
        if (!pt || !col) return;
        const value = pt.label ?? pt.hovertext ?? pt.x ?? pt.y;
        if (value == null) return;
        const same = highlight
          && String(highlight.column) === String(col)
          && String(highlight.value) === String(value);
        setHighlight(same ? null : { column: col, value });
      } : undefined}
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

export default React.memo(ChartRenderer);
