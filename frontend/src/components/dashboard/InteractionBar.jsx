import React from 'react';
import { useDashboardStore } from '../../dashboardStore';

// Phase 14 S14.3 — surfaces the active cross-highlight / filters as removable
// chips so an interaction is always visible and undoable. Renders nothing when
// there is no active interaction, so the dashboard looks unchanged by default.
export default function InteractionBar() {
  const highlight = useDashboardStore((s) => s.highlight);
  const filters = useDashboardStore((s) => s.filters);
  const setHighlight = useDashboardStore((s) => s.setHighlight);
  const removeFilter = useDashboardStore((s) => s.removeFilter);
  const clearFilters = useDashboardStore((s) => s.clearFilters);

  const active = !!highlight || (filters && filters.length > 0);
  if (!active) return null;

  const fmtVal = (v) => (Array.isArray(v) ? v.join('–') : String(v));

  return (
    <div
      className="glass-soft mb-6 p-3 flex flex-wrap items-center gap-2 border-sky-400/20"
      role="region"
      aria-label="Active dashboard focus"
    >
      <span className="text-[11px] uppercase tracking-[0.28em] text-slate-400 mr-1">
        <i className="fas fa-crosshairs mr-2 text-sky-300" />
        Focus
      </span>

      {highlight && (
        <button
          type="button"
          onClick={() => setHighlight(null)}
          className="neon-badge neon-blue inline-flex items-center gap-2"
          title="Clear highlight"
        >
          <span>{highlight.column}: {fmtVal(highlight.value)}</span>
          <i className="fas fa-times text-[10px]" />
        </button>
      )}

      {(filters || []).map((f) => (
        <button
          key={`${f.column}:${f.op || 'eq'}`}
          type="button"
          onClick={() => removeFilter(f.column, f.op || 'eq')}
          className="neon-badge inline-flex items-center gap-2"
          title="Remove filter"
        >
          <span>{f.column} {f.op || 'eq'} {fmtVal(f.value)}</span>
          <i className="fas fa-times text-[10px]" />
        </button>
      ))}

      <button
        type="button"
        onClick={clearFilters}
        className="ml-auto text-xs text-slate-300 hover:text-white inline-flex items-center gap-1.5"
      >
        <i className="fas fa-rotate-left" />
        Clear all
      </button>
    </div>
  );
}
