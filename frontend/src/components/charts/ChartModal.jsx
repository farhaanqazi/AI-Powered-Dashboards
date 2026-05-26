import React, { useEffect } from 'react';
import { createPortal } from 'react-dom';
import ChartRenderer from './ChartRenderer';

const ChartModal = ({ chart, onClose }) => {
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    // Lock background scroll while the modal is open.
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [onClose]);

  if (!chart) return null;
  const insight = chart.ai_insight;
  const title = chart.title || chart.id || 'Chart';

  // Portal to <body> so the fixed overlay escapes ancestors that establish a
  // containing block (transform / filter / will-change / backdrop-filter on
  // .dash-shell, .chart-card, etc.). Without this the "fixed" overlay is
  // positioned relative to a transformed ancestor and the user has to scroll
  // to find the enlarged chart.
  return createPortal(
    <div
      className="chart-modal-overlay"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
    >
      <div
        className="chart-modal-panel"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4 mb-4">
          <h3 className="text-lg md:text-xl font-semibold text-slate-100 truncate" title={title}>
            {title}
          </h3>
          <button
            onClick={onClose}
            aria-label="Close"
            className="flex-shrink-0 h-9 w-9 rounded-md text-slate-300 hover:bg-white/10"
          >
            <i className="fas fa-times" />
          </button>
        </div>

        {/* The enlarged view is the interaction surface: clicking a mark sets
            the cross-highlight. The grid cards only reflect it (their click
            opens this modal), so there is no click collision. */}
        <div className="chart-container chart-shell chart-modal-canvas">
          <ChartRenderer chartData={chart} interactive key={`modal-${chart.id || title}`} />
        </div>

        {insight && (
          <div className="chart-modal-insight">
            <div className="flex items-center gap-2 text-sky-300 text-xs uppercase tracking-[0.28em] mb-2">
              <i className="fas fa-wand-magic-sparkles" />
              <span>AI insight</span>
            </div>
            <p className="text-slate-200 text-sm leading-relaxed">{insight}</p>
          </div>
        )}
      </div>
    </div>,
    document.body
  );
};

export default ChartModal;
