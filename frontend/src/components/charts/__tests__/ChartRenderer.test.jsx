import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render } from '@testing-library/react';

// Phase 16 — exercise the Plotly figure ChartRenderer builds without loading
// the real (canvas-dependent) Plotly bundle. The factory mock captures the
// {data, layout} props handed to the Plot component so we can assert on them.
const cap = vi.hoisted(() => ({ props: null }));
vi.mock('plotly.js-basic-dist', () => ({ default: {} }));
vi.mock('react-plotly.js/factory', () => ({
  default: () => function MockPlot(props) { cap.props = props; return null; },
}));

import ChartRenderer from '../ChartRenderer';

beforeEach(() => { cap.props = null; });

const catData = {
  type: 'category_count',
  intent: 'category_count',
  column: 'region',
  data: [
    { category: 'North', count: 1200 },
    { category: 'South', count: 800 },
    { category: 'East', count: 400 },
  ],
};

describe('ChartRenderer S16 visual language', () => {
  it('rounds bar corners and colours by intent', () => {
    render(<ChartRenderer chartData={catData} />);
    const bar = cap.props.data[0];
    expect(bar.type).toBe('bar');
    expect(bar.marker.cornerradius).toBe(5);
    expect(bar.marker.color).toBe('#60a5fa');      // category intent → blue
    expect(cap.props.layout.bargap).toBeCloseTo(0.28);
  });

  it('adds compact value labels when the axis is not crowded', () => {
    render(<ChartRenderer chartData={catData} />);
    const bar = cap.props.data[0];
    expect(bar.text).toEqual(['1.2k', '800', '400']);
    expect(bar.textposition).toBe('outside');
  });

  it('draws an average reference line on category comparisons', () => {
    render(<ChartRenderer chartData={catData} />);
    const shapes = cap.props.layout.shapes || [];
    expect(shapes.length).toBeGreaterThanOrEqual(1);
    expect(shapes[0].line.dash).toBe('dot');
    const anns = cap.props.layout.annotations || [];
    expect(anns.some((a) => /avg/.test(a.text))).toBe(true);
  });

  it('suppresses value labels and mean line for histograms (dense bins)', () => {
    render(<ChartRenderer chartData={{
      type: 'histogram', intent: 'histogram', column: 'amount',
      data: Array.from({ length: 15 }, (_, i) => ({ bin_range: `${i}-${i + 1}`, count: i })),
    }} />);
    const bar = cap.props.data[0];
    expect(bar.text).toBeUndefined();
    expect(bar.marker.color).toBe('#a78bfa');       // distribution intent → violet
    expect((cap.props.layout.shapes || []).length).toBe(0);
  });

  it('overlays a least-squares trendline on a time series', () => {
    render(<ChartRenderer chartData={{
      type: 'time_series', intent: 'time_series', x_column: 'day', y_column: 'revenue',
      data: Array.from({ length: 12 }, (_, i) => ({ date: `2022-01-${i + 1}`, value: 100 + i * 5 })),
    }} />);
    expect(cap.props.data).toHaveLength(2);
    expect(cap.props.data[1].name).toBe('Trend');
    expect(cap.props.data[1].line.dash).toBe('dash');
  });

  it('does not add a trendline to a forecast series', () => {
    render(<ChartRenderer chartData={{
      type: 'time_series', intent: 'forecast', y_column: 'revenue',
      data: Array.from({ length: 12 }, (_, i) => ({ date: `2022-01-${i + 1}`, value: 100 + i * 5 })),
    }} />);
    expect(cap.props.data).toHaveLength(1);
  });
});
