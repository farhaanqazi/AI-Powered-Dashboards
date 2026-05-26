// Phase 14 S14.3 — Stage 1 client-side interaction logic.
import { describe, it, expect } from 'vitest';
import {
  rowMatches,
  applyClientFilters,
  isHighlighted,
  withExclusions,
  affectedCharts,
  chartDeps,
  highlightOpacity,
} from './clientFilter';

const rows = [
  { region: 'EMEA', units: 12, product: 'A' },
  { region: 'EMEA', units: 3, product: 'B' },
  { region: 'APAC', units: 20, product: 'A' },
  { region: 'AMER', units: 7, product: 'C' },
];

describe('rowMatches', () => {
  it('eq / neq on strings', () => {
    expect(rowMatches(rows[0], { column: 'region', op: 'eq', value: 'EMEA' })).toBe(true);
    expect(rowMatches(rows[0], { column: 'region', op: 'neq', value: 'EMEA' })).toBe(false);
  });
  it('numeric comparisons and between', () => {
    expect(rowMatches(rows[0], { column: 'units', op: 'gte', value: 12 })).toBe(true);
    expect(rowMatches(rows[1], { column: 'units', op: 'gt', value: 5 })).toBe(false);
    expect(rowMatches(rows[3], { column: 'units', op: 'between', value: [5, 10] })).toBe(true);
  });
  it('in / nin', () => {
    expect(rowMatches(rows[2], { column: 'region', op: 'in', value: ['APAC', 'AMER'] })).toBe(true);
    expect(rowMatches(rows[0], { column: 'region', op: 'nin', value: ['APAC'] })).toBe(true);
  });
  it('no filter matches everything', () => {
    expect(rowMatches(rows[0], null)).toBe(true);
  });
});

describe('applyClientFilters', () => {
  it('ANDs filters and never mutates input', () => {
    const out = applyClientFilters(rows, [
      { column: 'region', op: 'eq', value: 'EMEA' },
      { column: 'units', op: 'gte', value: 10 },
    ]);
    expect(out).toEqual([{ region: 'EMEA', units: 12, product: 'A' }]);
    expect(rows).toHaveLength(4);
  });
  it('empty filters returns all rows', () => {
    expect(applyClientFilters(rows, [])).toHaveLength(4);
  });
  it('non-array input is safe', () => {
    expect(applyClientFilters(null, [{ column: 'x', value: 1 }])).toEqual([]);
  });
});

describe('isHighlighted', () => {
  it('lights only the matching dimension key; null lights all', () => {
    expect(isHighlighted(rows[0], { column: 'product', value: 'A' })).toBe(true);
    expect(isHighlighted(rows[1], { column: 'product', value: 'A' })).toBe(false);
    expect(isHighlighted(rows[1], null)).toBe(true);
  });
});

describe('withExclusions', () => {
  it('drops flagged keys', () => {
    const out = withExclusions(rows, ['B'], (r) => r.product);
    expect(out.map((r) => r.product)).toEqual(['A', 'A', 'C']);
  });
});

describe('affectedCharts', () => {
  it('only returns charts whose dependsOn intersects the change', () => {
    const charts = [
      { id: 1, dependsOn: ['region'] },
      { id: 2, dependsOn: ['price'] },
      { id: 3, dependsOn: ['region', 'units'] },
    ];
    expect(affectedCharts(charts, ['region']).map((c) => c.id)).toEqual([1, 3]);
  });
  it('falls back to derived deps when dependsOn is absent', () => {
    const charts = [
      { id: 1, x_column: 'region', y_column: 'units' },
      { id: 2, column: 'price' },
    ];
    expect(affectedCharts(charts, ['region']).map((c) => c.id)).toEqual([1]);
    expect(affectedCharts(charts, ['price']).map((c) => c.id)).toEqual([2]);
  });
});

describe('chartDeps', () => {
  it('prefers an explicit dependsOn', () => {
    expect(chartDeps({ dependsOn: ['a', 'b'], x_column: 'z' })).toEqual(['a', 'b']);
  });
  it('derives from x_column / y_column / column and de-dupes', () => {
    expect(chartDeps({ x_column: 'region', y_column: 'units' })).toEqual(['region', 'units']);
    expect(chartDeps({ x_column: 'region', column: 'region' })).toEqual(['region']);
    expect(chartDeps({ column: 'price' })).toEqual(['price']);
  });
  it('is empty for a chart with no dimension fields', () => {
    expect(chartDeps({ title: 'x' })).toEqual([]);
    expect(chartDeps(null)).toEqual([]);
  });
});

describe('highlightOpacity', () => {
  const labels = ['EMEA', 'APAC', 'AMER'];
  it('returns null when the highlight is not on this chart dimension', () => {
    expect(highlightOpacity(labels, 'region', null)).toBeNull();
    expect(highlightOpacity(labels, 'region', { column: 'product', value: 'A' })).toBeNull();
    expect(highlightOpacity(labels, null, { column: 'region', value: 'APAC' })).toBeNull();
  });
  it('lights the matching label and dims the rest when the column matches', () => {
    expect(highlightOpacity(labels, 'region', { column: 'region', value: 'APAC' }))
      .toEqual([0.22, 1, 0.22]);
  });
  it('honours custom lit/dim and handles empty labels', () => {
    expect(highlightOpacity(['X'], 'c', { column: 'c', value: 'X' }, { dim: 0.1, lit: 0.9 }))
      .toEqual([0.9]);
    expect(highlightOpacity([], 'c', { column: 'c', value: 'X' })).toBeNull();
  });
});
