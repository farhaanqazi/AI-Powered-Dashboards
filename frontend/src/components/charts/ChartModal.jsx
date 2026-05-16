import React, { useEffect } from 'react';
import ChartRenderer from './ChartRenderer';

const ChartModal = ({ chart, onClose }) => {
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  if (!chart) return null;
  const insight = chart.ai_insight;
  const title = chart.title || chart.id || 'Chart';

  return (
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

        <div className="chart-container chart-shell chart-modal-canvas">
          <ChartRenderer chartData={chart} key={`modal-${chart.id || title}`} />
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
    </div>
  );
};

export default ChartModal;
