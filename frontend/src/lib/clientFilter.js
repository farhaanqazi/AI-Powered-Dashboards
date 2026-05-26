// Phase 14 S14.3 — Stage 1 client-side interaction logic.
//
// Pure, dependency-free, deterministic filtering/highlighting over data that
// is ALREADY on the page (the points inside the shipped chart specs). This is
// NOT WASM and NOT a query engine: it is plain array work, the cheap tier
// that handles cross-highlight + simple narrowing with zero server calls.
// Anything needing fresh math goes to the server via services.runInteraction.
//
// Op vocabulary is kept identical to the backend interaction engine so a
// client view and a later server "verify" never diverge in meaning.

const NUMERIC_OPS = new Set(['gt', 'gte', 'lt', 'lte', 'between']);

export function rowMatches(row, filter) {
  if (!filter || !filter.column) return true;
  const { column, op = 'eq', value } = filter;
  const raw = row?.[column];
  if (NUMERIC_OPS.has(op)) {
    const n = Number(raw);
    if (Number.isNaN(n)) return false;
    if (op === 'between') {
      if (!Array.isArray(value) || value.length !== 2) return false;
      const lo = Number(value[0]);
      const hi = Number(value[1]);
      return n >= lo && n <= hi;
    }
    const v = Number(value);
    if (op === 'gt') return n > v;
    if (op === 'gte') return n >= v;
    if (op === 'lt') return n < v;
    return n <= v; // lte
  }
  const s = String(raw);
  if (op === 'in' || op === 'nin') {
    const set = (Array.isArray(value) ? value : [value]).map(String);
    const isin = set.includes(s);
    return op === 'in' ? isin : !isin;
  }
  if (op === 'neq') return s !== String(value);
  return s === String(value); // eq (default)
}

// AND across all filters (matches the backend's sequential narrowing).
export function applyClientFilters(rows, filters) {
  if (!Array.isArray(rows)) return [];
  if (!filters || filters.length === 0) return rows;
  return rows.filter((r) => filters.every((f) => rowMatches(r, f)));
}

// Cross-highlight: a row is "lit" when it shares the selected dimension key.
// `highlight` = { column, value } | null. No data removal — caller dims the
// non-matching marks visually.
export function isHighlighted(row, highlight) {
  if (!highlight || !highlight.column) return true;
  return String(row?.[highlight.column]) === String(highlight.value);
}

// Exclude-outliers (client tier): drop rows whose key is user-flagged.
export function withExclusions(rows, excludedKeys, keyFn) {
  if (!Array.isArray(rows) || !excludedKeys || excludedKeys.length === 0) {
    return rows || [];
  }
  const set = new Set(excludedKeys.map(String));
  const kf = typeof keyFn === 'function' ? keyFn : (r) => r;
  return rows.filter((r) => !set.has(String(kf(r))));
}

// Derive the dataset columns a chart is built on from the fields the backend
// ALREADY ships (x_column / y_column / column) — the payload carries no
// explicit `dependsOn`, and adding one would fatten every chart spec. An
// explicit `dependsOn` (if a future payload sets it) wins. This is the
// dependency-gating primitive: a chart only reacts to a highlight whose
// column is one of these.
export function chartDeps(chart) {
  if (!chart) return [];
  if (Array.isArray(chart.dependsOn) && chart.dependsOn.length) {
    return chart.dependsOn.map(String);
  }
  const cols = [chart.x_column, chart.y_column, chart.column].filter(Boolean);
  return Array.from(new Set(cols.map(String)));
}

// Which charts must update for a filter/highlight change: only those whose
// dimensions intersect the changed columns. Visibility gating is applied by
// the caller (IntersectionObserver) so off-screen charts defer.
export function affectedCharts(charts, changedColumns) {
  const changed = new Set((changedColumns || []).map(String));
  return (charts || []).filter((c) =>
    chartDeps(c).some((d) => changed.has(String(d))),
  );
}

// Per-mark opacity for cross-highlight on a categorical chart. Returns null
// when the highlight does not apply to this chart (its category dimension is
// not the highlighted column) so the caller leaves styling byte-for-byte
// unchanged. When it applies, the matching category is lit and the rest dimmed.
export function highlightOpacity(labels, categoryColumn, highlight, opts = {}) {
  const { dim = 0.22, lit = 1 } = opts;
  if (!highlight || !highlight.column || !categoryColumn) return null;
  if (String(highlight.column) !== String(categoryColumn)) return null;
  if (!Array.isArray(labels) || labels.length === 0) return null;
  const target = String(highlight.value);
  return labels.map((l) => (String(l) === target ? lit : dim));
}
