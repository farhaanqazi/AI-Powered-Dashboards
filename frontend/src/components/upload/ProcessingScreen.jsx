import React, { useEffect, useRef, useState } from 'react';

const PHASES = [
  { key: 'reading',     label: 'Reading CSV',        icon: 'fa-cloud-arrow-up' },
  { key: 'preparing',   label: 'Preparing data',     icon: 'fa-shuffle' },
  { key: 'profiling',   label: 'Profiling columns',  icon: 'fa-magnifying-glass-chart' },
  { key: 'classifying', label: 'Classifying types',  icon: 'fa-tags' },
  { key: 'relating',    label: 'Finding relations',  icon: 'fa-diagram-project' },
  { key: 'eda',         label: 'Running EDA',        icon: 'fa-flask' },
  { key: 'kpis',        label: 'Computing KPIs',     icon: 'fa-gauge-high' },
  { key: 'rendering',   label: 'Building charts',    icon: 'fa-chart-line' },
];

const phaseIndex = (key) => {
  const i = PHASES.findIndex(p => p.key === key);
  return i < 0 ? 0 : i;
};

const formatFileSize = (bytes) => {
  if (!Number.isFinite(bytes)) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
};

const ProcessingScreen = ({ file, phase, message, onCancel }) => {
  const [activity, setActivity] = useState([]);
  const [elapsed, setElapsed] = useState(0);
  const startedAt = useRef(Date.now());
  const lastPhaseKey = useRef(null);
  const feedRef = useRef(null);

  // Append distinct phase messages to the activity feed
  useEffect(() => {
    if (!phase) return;
    if (phase === lastPhaseKey.current) return;
    lastPhaseKey.current = phase;
    setActivity(prev => {
      const next = [...prev, { ts: Date.now(), phase, message }];
      return next.slice(-8);
    });
  }, [phase, message]);

  // Tick the elapsed counter every 500ms
  useEffect(() => {
    const id = setInterval(() => {
      setElapsed(Math.round((Date.now() - startedAt.current) / 1000));
    }, 500);
    return () => clearInterval(id);
  }, []);

  // Auto-scroll the activity feed when it grows
  useEffect(() => {
    if (feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight;
    }
  }, [activity]);

  const currentIdx = phaseIndex(phase);
  const currentLabel = PHASES[currentIdx]?.label || 'Starting...';

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-gradient-to-br from-slate-950 via-blue-950 to-indigo-950 text-white">
      {/* Drifting particle field */}
      <div className="pointer-events-none absolute inset-0 opacity-60" id="proc-particles" />

      {/* Top-right cancel button */}
      <button
        onClick={onCancel}
        className="absolute top-6 right-6 z-20 inline-flex items-center gap-2 rounded-lg border border-white/15 bg-white/5 px-4 py-2 text-sm text-white/80 backdrop-blur-md transition-all hover:bg-white/10 hover:text-white"
      >
        <i className="fas fa-xmark" />
        Cancel
      </button>

      {/* Top-left file badge */}
      <div className="absolute top-6 left-6 z-20 flex items-center gap-3 rounded-lg border border-white/10 bg-white/5 px-4 py-2 backdrop-blur-md">
        <i className="fas fa-file-csv text-emerald-400" />
        <div className="flex flex-col leading-tight">
          <span className="max-w-[260px] truncate text-sm font-medium text-white" title={file?.name}>
            {file?.name || 'Processing...'}
          </span>
          <span className="text-xs text-white/50 tabular-nums">
            {formatFileSize(file?.size)} · {elapsed}s elapsed
          </span>
        </div>
      </div>

      {/* Center stack */}
      <div className="relative z-10 flex h-full w-full flex-col items-center justify-center px-6">
        {/* Animated orb */}
        <div className="relative mb-12 flex h-56 w-56 items-center justify-center">
          <div className="proc-ring proc-ring-1" />
          <div className="proc-ring proc-ring-2" />
          <div className="proc-ring proc-ring-3" />
          <div className="proc-pulse proc-pulse-a" />
          <div className="proc-pulse proc-pulse-b" />
          <div className="proc-orb relative flex h-28 w-28 items-center justify-center rounded-full">
            <i className={`fas ${PHASES[currentIdx]?.icon || 'fa-spinner'} text-3xl text-white/90`} />
          </div>
        </div>

        {/* Current phase label + sub-message */}
        <div className="mb-10 text-center min-h-[3.5rem]">
          <div key={currentLabel} className="proc-phase-fade text-2xl font-semibold tracking-tight md:text-3xl">
            {currentLabel}
          </div>
          <div className="mt-2 text-sm text-white/60 md:text-base">
            {message || 'Working...'}
          </div>
        </div>

        {/* Phase pipeline */}
        <div className="mb-12 flex w-full max-w-3xl items-center justify-between">
          {PHASES.map((p, i) => {
            const isDone = i < currentIdx;
            const isActive = i === currentIdx;
            return (
              <React.Fragment key={p.key}>
                <div className="flex flex-col items-center">
                  <div
                    className={[
                      'relative flex h-9 w-9 items-center justify-center rounded-full border transition-all duration-300',
                      isDone   ? 'border-emerald-400 bg-emerald-400/20 text-emerald-300' : '',
                      isActive ? 'border-blue-400 bg-blue-400/20 text-blue-200 proc-active-dot' : '',
                      !isDone && !isActive ? 'border-white/15 bg-white/5 text-white/30' : '',
                    ].join(' ')}
                    title={p.label}
                  >
                    {isDone ? (
                      <i className="fas fa-check text-xs" />
                    ) : (
                      <i className={`fas ${p.icon} text-xs`} />
                    )}
                  </div>
                  <span className={`mt-2 hidden md:block text-[10px] uppercase tracking-wider ${isActive ? 'text-blue-200' : isDone ? 'text-emerald-300/70' : 'text-white/30'}`}>
                    {p.label.split(' ')[0]}
                  </span>
                </div>
                {i < PHASES.length - 1 && (
                  <div className={`mx-1 h-px flex-1 ${i < currentIdx ? 'bg-emerald-400/40' : 'bg-white/10'}`} />
                )}
              </React.Fragment>
            );
          })}
        </div>

        {/* Live activity feed */}
        <div className="w-full max-w-md rounded-xl border border-white/10 bg-white/[0.03] p-4 backdrop-blur-md">
          <div className="mb-2 flex items-center justify-between text-[10px] uppercase tracking-widest text-white/40">
            <span><i className="fas fa-signal mr-2" /> Activity</span>
            <span className="tabular-nums">{activity.length}/{PHASES.length}</span>
          </div>
          <div ref={feedRef} className="max-h-32 space-y-1.5 overflow-y-auto pr-1 proc-feed">
            {activity.length === 0 && (
              <div className="text-xs text-white/30">Waiting for first event...</div>
            )}
            {activity.map((entry, i) => {
              const isLast = i === activity.length - 1;
              return (
                <div
                  key={`${entry.phase}-${entry.ts}`}
                  className={`proc-feed-item flex items-start gap-2 text-xs leading-relaxed ${isLast ? 'text-white' : 'text-white/50'}`}
                >
                  <span className={`mt-1 inline-block h-1.5 w-1.5 flex-shrink-0 rounded-full ${isLast ? 'bg-blue-400 proc-feed-dot' : 'bg-emerald-400/60'}`} />
                  <span className="break-words">{entry.message}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <script dangerouslySetInnerHTML={{
        __html: `
          (function () {
            const root = document.getElementById('proc-particles');
            if (!root || root.dataset.seeded === '1') return;
            root.dataset.seeded = '1';
            const N = 40;
            for (let i = 0; i < N; i++) {
              const p = document.createElement('div');
              p.className = 'proc-particle';
              const size = Math.random() * 3 + 1;
              p.style.width = size + 'px';
              p.style.height = size + 'px';
              p.style.left = (Math.random() * 100) + '%';
              p.style.bottom = '-10px';
              p.style.animationDuration = (Math.random() * 12 + 14) + 's';
              p.style.animationDelay = (Math.random() * 8) + 's';
              p.style.opacity = (Math.random() * 0.5 + 0.2).toFixed(2);
              root.appendChild(p);
            }
          })();
        `
      }} />

      <style>{`
        .proc-orb {
          background: radial-gradient(circle at 30% 30%, rgba(96,165,250,0.95), rgba(168,85,247,0.75) 60%, rgba(45,212,191,0.55));
          box-shadow:
            0 0 60px 10px rgba(96,165,250,0.45),
            inset 0 0 30px rgba(255,255,255,0.15);
          animation: proc-orb-pulse 3.2s ease-in-out infinite;
        }
        @keyframes proc-orb-pulse {
          0%, 100% { transform: scale(1); filter: brightness(1); }
          50%      { transform: scale(1.06); filter: brightness(1.15); }
        }

        .proc-ring {
          position: absolute;
          border-radius: 9999px;
          border: 1px solid rgba(255,255,255,0.12);
        }
        .proc-ring-1 { width: 160px; height: 160px; animation: proc-spin 14s linear infinite; border-top-color: rgba(96,165,250,0.5); }
        .proc-ring-2 { width: 200px; height: 200px; animation: proc-spin 22s linear infinite reverse; border-right-color: rgba(168,85,247,0.4); }
        .proc-ring-3 { width: 224px; height: 224px; animation: proc-spin 30s linear infinite; border-bottom-color: rgba(45,212,191,0.35); }
        @keyframes proc-spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

        .proc-pulse {
          position: absolute;
          border-radius: 9999px;
          border: 1px solid rgba(96,165,250,0.4);
          width: 120px; height: 120px;
          opacity: 0;
          animation: proc-pulse-out 2.8s ease-out infinite;
        }
        .proc-pulse-b { animation-delay: 1.4s; }
        @keyframes proc-pulse-out {
          0%   { transform: scale(0.7); opacity: 0.7; }
          80%  { opacity: 0.1; }
          100% { transform: scale(1.9); opacity: 0; }
        }

        .proc-active-dot {
          box-shadow: 0 0 0 0 rgba(96,165,250,0.55);
          animation: proc-dot-glow 1.6s ease-out infinite;
        }
        @keyframes proc-dot-glow {
          0%   { box-shadow: 0 0 0 0 rgba(96,165,250,0.55); }
          70%  { box-shadow: 0 0 0 10px rgba(96,165,250,0); }
          100% { box-shadow: 0 0 0 0 rgba(96,165,250,0); }
        }

        .proc-phase-fade { animation: proc-phase-in 320ms cubic-bezier(0.2, 0.7, 0.2, 1); }
        @keyframes proc-phase-in {
          from { opacity: 0; transform: translateY(6px); }
          to   { opacity: 1; transform: translateY(0); }
        }

        .proc-feed-item { animation: proc-feed-in 260ms ease-out; }
        @keyframes proc-feed-in {
          from { opacity: 0; transform: translateY(4px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .proc-feed-dot { animation: proc-dot-glow 1.6s ease-out infinite; }

        .proc-particle {
          position: absolute;
          background: rgba(255,255,255,0.7);
          border-radius: 9999px;
          animation: proc-particle-float linear infinite;
          pointer-events: none;
        }
        @keyframes proc-particle-float {
          0%   { transform: translateY(0); opacity: 0; }
          15%  { opacity: 1; }
          85%  { opacity: 1; }
          100% { transform: translateY(-105vh); opacity: 0; }
        }

        .proc-feed::-webkit-scrollbar { width: 4px; }
        .proc-feed::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 9999px; }
      `}</style>
    </div>
  );
};

export default ProcessingScreen;
