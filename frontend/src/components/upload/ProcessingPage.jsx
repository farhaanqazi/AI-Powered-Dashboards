import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { uploadFileStream, loadExternalSource } from '../../services/api';

const PHASES = [
  { key: 'reading',     label: 'Reading CSV',        icon: 'fa-cloud-arrow-up',         tint: 'from-sky-400 to-blue-500' },
  { key: 'preparing',   label: 'Preparing data',     icon: 'fa-shuffle',                tint: 'from-blue-400 to-indigo-500' },
  { key: 'profiling',   label: 'Profiling columns',  icon: 'fa-magnifying-glass-chart', tint: 'from-indigo-400 to-purple-500' },
  { key: 'classifying', label: 'Classifying types',  icon: 'fa-tags',                   tint: 'from-purple-400 to-fuchsia-500' },
  { key: 'relating',    label: 'Finding relations',  icon: 'fa-diagram-project',        tint: 'from-fuchsia-400 to-pink-500' },
  { key: 'eda',         label: 'Running EDA',        icon: 'fa-flask',                  tint: 'from-pink-400 to-rose-500' },
  { key: 'kpis',        label: 'Computing KPIs',     icon: 'fa-gauge-high',             tint: 'from-rose-400 to-orange-500' },
  { key: 'rendering',   label: 'Building charts',    icon: 'fa-chart-line',             tint: 'from-orange-400 to-amber-500' },
];

const QUOTES = [
  "Teaching the data its manners…",
  "Bribing the missing values to confess…",
  "Convincing the columns to behave…",
  "Whispering sweet nothings to the histogram…",
  "Asking outliers to please calm down…",
  "Counting the rows. Twice. Just to be sure…",
  "Untangling the correlation spaghetti…",
  "Reminding numbers they have feelings too…",
  "Politely asking categorical data to mingle…",
  "Pretending we understand p-values…",
  "Brewing a fresh pot of insight…",
  "Negotiating peace talks with NaN…",
  "Decoding the secrets of skewness…",
  "Reading between the rows…",
  "Letting pandas do their thing…",
  "Plotting graceful curves and gentle slopes…",
  "Hyping up the histogram…",
  "Convening a board meeting of integers…",
  "Asking the median to please step forward…",
  "Encouraging the variance to relax a little…",
  "Polishing each pixel before the big reveal…",
  "Persuading the data to spill the tea…",
  "Aligning the axes for maximum elegance…",
  "Making sure every bar earns its height…",
  "Reticulating splines (just kidding) (mostly)…",
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

const formatDuration = (ms) => {
  if (!Number.isFinite(ms) || ms < 0) return '';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms % 60_000) / 1000);
  return `${m}m ${s}s`;
};

const PHASE_LABEL = Object.fromEntries(PHASES.map(p => [p.key, p.label]));

const closeOutTimings = (timings, now = Date.now()) => {
  if (!timings.length) return timings;
  const last = timings[timings.length - 1];
  if (last.endedAt !== null) return timings;
  return [...timings.slice(0, -1), { ...last, endedAt: now }];
};

const persistTimings = (timings, meta) => {
  try {
    const phases = timings.map(t => ({
      key: t.key,
      ms: (t.endedAt ?? Date.now()) - t.startedAt,
    }));
    const entry = { ts: Date.now(), ...meta, phases };
    const prev = JSON.parse(localStorage.getItem('dataInsight:timings') || '[]');
    localStorage.setItem('dataInsight:timings', JSON.stringify([...prev, entry].slice(-20)));
    if (typeof console.table === 'function') {
      console.table(phases.map(p => ({ phase: PHASE_LABEL[p.key] || p.key, duration: formatDuration(p.ms), ms: p.ms })));
    }
  } catch (_) { /* localStorage may be unavailable */ }
};

const ProcessingPage = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const job = location.state || null;
  const file = job?.kind === 'file' ? job.file : null;
  const source = job?.kind === 'external' ? job.source : null;

  const [phaseKey, setPhaseKey] = useState('reading');
  const [phaseMessage, setPhaseMessage] = useState('Connecting to server…');
  const [activity, setActivity] = useState([]);
  const [timings, setTimings] = useState([]); // [{ key, startedAt, endedAt|null }]
  const [elapsed, setElapsed] = useState(0);
  const [quoteIdx, setQuoteIdx] = useState(() => Math.floor(Math.random() * QUOTES.length));
  const [error, setError] = useState('');

  const abortRef = useRef(null);
  const startedAt = useRef(Date.now());
  const lastPhaseKey = useRef(null);
  const feedRef = useRef(null);
  const startedOnce = useRef(false);

  const currentIdx = phaseIndex(phaseKey);
  const currentPhase = PHASES[currentIdx];
  const progressPct = Math.min(100, Math.round(((currentIdx + 1) / PHASES.length) * 100));

  // Per-phase durations (re-derived each elapsed tick so the live bar grows)
  const breakdown = useMemo(() => {
    const items = timings.map(t => ({
      key: t.key,
      ms: (t.endedAt ?? Date.now()) - t.startedAt,
      live: t.endedAt == null,
    }));
    const maxMs = items.reduce((m, x) => Math.max(m, x.ms), 0) || 1;
    const completed = items.filter(x => !x.live);
    const slowest = completed.reduce((s, x) => (!s || x.ms > s.ms ? x : s), null);
    const byKey = Object.fromEntries(items.map(x => [x.key, x]));
    return { items, byKey, maxMs, slowest };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [timings, elapsed]);

  // Redirect if landed here without a job
  useEffect(() => {
    if (!file && !source) {
      navigate('/', { replace: true });
    }
  }, [file, source, navigate]);

  // Start the actual work exactly once
  useEffect(() => {
    if (startedOnce.current) return;
    if (!file && !source) return;
    startedOnce.current = true;

    const controller = new AbortController();
    abortRef.current = controller;

    const run = async () => {
      try {
        if (file) {
          await uploadFileStream(
            file,
            (evt) => {
              if (evt.phase && evt.phase !== 'done' && evt.phase !== 'error') {
                setPhaseKey(evt.phase);
              }
              if (evt.message) setPhaseMessage(evt.message);
            },
            { signal: controller.signal },
          );
        } else if (source) {
          setPhaseMessage('Fetching remote dataset…');
          await loadExternalSource(source);
          setPhaseKey('rendering');
          setPhaseMessage('Almost there…');
        }
        const finishedAt = Date.now();
        setTimings(prev => {
          const finalized = closeOutTimings(prev, finishedAt);
          persistTimings(finalized, {
            filename: file?.name || source || 'unknown',
            size: file?.size ?? null,
            totalMs: finishedAt - startedAt.current,
          });
          return finalized;
        });
        navigate('/dashboard', { replace: true });
      } catch (err) {
        if (err.name === 'AbortError') {
          navigate('/', { replace: true });
          return;
        }
        setError(err?.response?.data?.detail || err.message || 'Processing failed');
      }
    };

    run();
    // No cleanup abort here: StrictMode double-mount would kill the upload.
    // The cancel button is the only intentional way to abort.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Activity feed + per-phase timings: capture on distinct phase changes
  useEffect(() => {
    if (!phaseKey) return;
    if (phaseKey === lastPhaseKey.current) return;
    lastPhaseKey.current = phaseKey;
    const now = Date.now();
    setActivity(prev => {
      const next = [...prev, { ts: now, phase: phaseKey, message: phaseMessage }];
      return next.slice(-8);
    });
    setTimings(prev => {
      const closed = closeOutTimings(prev, now);
      return [...closed, { key: phaseKey, startedAt: now, endedAt: null }];
    });
  }, [phaseKey, phaseMessage]);

  // Elapsed timer
  useEffect(() => {
    const id = setInterval(() => {
      setElapsed(Math.round((Date.now() - startedAt.current) / 1000));
    }, 500);
    return () => clearInterval(id);
  }, []);

  // Rotating witty quotes
  useEffect(() => {
    const id = setInterval(() => {
      setQuoteIdx(i => (i + 1 + Math.floor(Math.random() * (QUOTES.length - 1))) % QUOTES.length);
    }, 3800);
    return () => clearInterval(id);
  }, []);

  // Auto-scroll feed
  useEffect(() => {
    if (feedRef.current) feedRef.current.scrollTop = feedRef.current.scrollHeight;
  }, [activity]);

  const handleCancel = () => {
    if (abortRef.current) abortRef.current.abort();
    navigate('/', { replace: true });
  };

  const sourceLabel = useMemo(() => {
    if (file) return file.name;
    if (source) return source.length > 48 ? source.slice(0, 45) + '…' : source;
    return 'Processing…';
  }, [file, source]);

  const sourceSub = useMemo(() => {
    if (file) return `${formatFileSize(file.size)} · ${elapsed}s elapsed`;
    if (source) return `External source · ${elapsed}s elapsed`;
    return `${elapsed}s elapsed`;
  }, [file, source, elapsed]);

  return (
    <div className="relative min-h-[calc(100vh-8rem)] overflow-hidden bg-gradient-to-br from-slate-950 via-blue-950 to-indigo-950 text-white">
      {/* Aurora gradient mesh */}
      <div className="pointer-events-none absolute inset-0 opacity-70">
        <div className="proc-aurora proc-aurora-a" />
        <div className="proc-aurora proc-aurora-b" />
        <div className="proc-aurora proc-aurora-c" />
      </div>

      {/* Drifting particles */}
      <div className="pointer-events-none absolute inset-0" id="proc-particles" />

      {/* Floating data glyphs */}
      <div className="pointer-events-none absolute inset-0 hidden md:block">
        {['fa-table', 'fa-chart-pie', 'fa-percent', 'fa-sigma', 'fa-hashtag', 'fa-chart-area', 'fa-database', 'fa-square-root-variable'].map((g, i) => (
          <i
            key={g}
            className={`fas ${g} proc-glyph absolute text-white/[0.06]`}
            style={{
              left: `${(i * 13 + 7) % 92}%`,
              top: `${(i * 19 + 12) % 80}%`,
              fontSize: `${28 + (i % 4) * 14}px`,
              animationDelay: `${(i % 5) * 0.7}s`,
              animationDuration: `${10 + (i % 4) * 3}s`,
            }}
          />
        ))}
      </div>

      <div className="relative z-10 mx-auto flex max-w-6xl flex-col items-center px-6 py-12">
        {/* Top bar: source badge + cancel */}
        <div className="mb-10 flex w-full items-center justify-between">
          <div className="flex items-center gap-3 rounded-xl border border-white/10 bg-white/[0.04] px-4 py-2.5 backdrop-blur-md">
            <i className={`fas ${file ? 'fa-file-csv' : 'fa-link'} text-emerald-400`} />
            <div className="flex flex-col leading-tight">
              <span className="max-w-[280px] truncate text-sm font-medium text-white" title={sourceLabel}>
                {sourceLabel}
              </span>
              <span className="text-xs text-white/50 tabular-nums">{sourceSub}</span>
            </div>
          </div>

          <button
            onClick={handleCancel}
            className="inline-flex items-center gap-2 rounded-lg border border-white/15 bg-white/5 px-4 py-2 text-sm text-white/80 backdrop-blur-md transition-all hover:bg-white/10 hover:text-white"
          >
            <i className="fas fa-xmark" />
            Cancel
          </button>
        </div>

        {/* Hero: orb + headline */}
        <div className="grid w-full grid-cols-1 items-center gap-10 md:grid-cols-[auto_1fr]">
          {/* Orb cluster */}
          <div className="relative mx-auto flex h-64 w-64 items-center justify-center">
            <div className="proc-ring proc-ring-1" />
            <div className="proc-ring proc-ring-2" />
            <div className="proc-ring proc-ring-3" />
            <div className="proc-ring proc-ring-4" />
            <div className="proc-pulse proc-pulse-a" />
            <div className="proc-pulse proc-pulse-b" />
            <div className="proc-pulse proc-pulse-c" />
            <div className={`proc-orb relative flex h-32 w-32 items-center justify-center rounded-full bg-gradient-to-br ${currentPhase?.tint || 'from-blue-400 to-purple-500'}`}>
              <i key={currentPhase?.key} className={`fas ${currentPhase?.icon || 'fa-spinner'} proc-icon-swap text-4xl text-white`} />
            </div>
            {/* Orbiting dots */}
            <div className="proc-orbit proc-orbit-a"><span className="proc-orbit-dot" /></div>
            <div className="proc-orbit proc-orbit-b"><span className="proc-orbit-dot proc-orbit-dot-2" /></div>
            <div className="proc-orbit proc-orbit-c"><span className="proc-orbit-dot proc-orbit-dot-3" /></div>
          </div>

          {/* Headline + quote + progress */}
          <div className="text-center md:text-left">
            <div className="mb-1 text-xs uppercase tracking-[0.3em] text-white/40">Crunching numbers</div>
            <h1 key={currentPhase?.label} className="proc-phase-fade bg-gradient-to-r from-blue-300 via-fuchsia-300 to-amber-200 bg-clip-text text-3xl font-bold tracking-tight text-transparent md:text-5xl">
              {currentPhase?.label || 'Starting…'}
            </h1>
            <p className="mt-3 text-base text-white/70 md:text-lg">{phaseMessage || 'Working…'}</p>

            {/* Witty rotating quote */}
            <div className="relative mt-8 min-h-[3.25rem] overflow-hidden rounded-xl border border-white/10 bg-white/[0.04] p-4 backdrop-blur-md">
              <div className="flex items-start gap-3">
                <i className="fas fa-quote-left mt-1 text-white/30" />
                <div key={quoteIdx} className="proc-quote-fade text-sm italic text-white/80 md:text-base">
                  {QUOTES[quoteIdx]}
                </div>
              </div>
              <div className="proc-quote-shimmer pointer-events-none absolute inset-0" />
            </div>

            {/* Progress bar */}
            <div className="mt-8">
              <div className="mb-2 flex items-center justify-between text-xs uppercase tracking-widest text-white/40">
                <span>Pipeline</span>
                <span className="tabular-nums text-white/70">{progressPct}%</span>
              </div>
              <div className="relative h-2 w-full overflow-hidden rounded-full bg-white/10">
                <div
                  className="proc-bar absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-blue-400 via-fuchsia-400 to-amber-400 transition-all duration-700 ease-out"
                  style={{ width: `${progressPct}%` }}
                />
                <div className="proc-bar-shimmer pointer-events-none absolute inset-0" />
              </div>
            </div>
          </div>
        </div>

        {/* Phase pipeline */}
        <div className="mt-14 flex w-full max-w-3xl items-center justify-between">
          {PHASES.map((p, i) => {
            const isDone = i < currentIdx;
            const isActive = i === currentIdx;
            const t = breakdown.byKey[p.key];
            return (
              <React.Fragment key={p.key}>
                <div className="flex flex-col items-center">
                  <div
                    className={[
                      'relative flex h-10 w-10 items-center justify-center rounded-full border transition-all duration-300',
                      isDone   ? 'border-emerald-400 bg-emerald-400/20 text-emerald-300' : '',
                      isActive ? 'border-blue-400 bg-blue-400/25 text-blue-100 proc-active-dot' : '',
                      !isDone && !isActive ? 'border-white/15 bg-white/5 text-white/30' : '',
                    ].join(' ')}
                    title={t ? `${p.label} · ${formatDuration(t.ms)}` : p.label}
                  >
                    {isDone ? (
                      <i className="fas fa-check text-xs" />
                    ) : (
                      <i className={`fas ${p.icon} text-xs`} />
                    )}
                  </div>
                  <span className={`mt-2 hidden text-[10px] uppercase tracking-wider md:block ${isActive ? 'text-blue-200' : isDone ? 'text-emerald-300/70' : 'text-white/30'}`}>
                    {p.label.split(' ')[0]}
                  </span>
                </div>
                {i < PHASES.length - 1 && (
                  <div className="relative mx-1 h-px flex-1 bg-white/10">
                    <div
                      className="absolute inset-y-0 left-0 bg-gradient-to-r from-emerald-400/60 to-blue-400/60 transition-all duration-700"
                      style={{ width: i < currentIdx ? '100%' : isActive ? '50%' : '0%' }}
                    />
                    {t && (
                      <span
                        className={[
                          'absolute -top-5 left-1/2 -translate-x-1/2 rounded-full border px-1.5 py-0.5 text-[9px] font-medium tabular-nums backdrop-blur-md whitespace-nowrap',
                          t.live
                            ? 'border-blue-300/40 bg-blue-400/15 text-blue-100 proc-time-pulse'
                            : 'border-emerald-300/40 bg-emerald-400/15 text-emerald-100',
                        ].join(' ')}
                        title={`${p.label}: ${formatDuration(t.ms)}${t.live ? ' (running)' : ''}`}
                      >
                        {formatDuration(t.ms)}
                      </span>
                    )}
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>

        {/* Activity feed */}
        <div className="mt-12 w-full max-w-2xl rounded-2xl border border-white/10 bg-white/[0.03] p-5 backdrop-blur-md">
          <div className="mb-3 flex items-center justify-between text-[11px] uppercase tracking-widest text-white/40">
            <span className="flex items-center gap-2">
              <span className="proc-live-dot" />
              Live activity
            </span>
            <span className="tabular-nums">{activity.length}/{PHASES.length}</span>
          </div>
          <div ref={feedRef} className="proc-feed max-h-40 space-y-2 overflow-y-auto pr-1">
            {activity.length === 0 && (
              <div className="text-xs text-white/30">Warming up the engines…</div>
            )}
            {activity.map((entry, i) => {
              const isLast = i === activity.length - 1;
              const t = breakdown.byKey[entry.phase];
              return (
                <div
                  key={`${entry.phase}-${entry.ts}`}
                  className={`proc-feed-item flex items-start gap-3 text-sm leading-relaxed ${isLast ? 'text-white' : 'text-white/50'}`}
                >
                  <span className={`mt-1.5 inline-block h-1.5 w-1.5 flex-shrink-0 rounded-full ${isLast ? 'bg-blue-400 proc-feed-dot' : 'bg-emerald-400/60'}`} />
                  <span className="break-words flex-1 min-w-0">{entry.message}</span>
                  {t && (
                    <span
                      className={[
                        'flex-shrink-0 rounded-md px-1.5 py-0.5 text-[10px] font-medium tabular-nums',
                        t.live
                          ? 'bg-blue-400/15 text-blue-200 proc-time-pulse'
                          : 'bg-emerald-400/15 text-emerald-200/90',
                      ].join(' ')}
                    >
                      {formatDuration(t.ms)}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Timing breakdown — exposes bottlenecks for column/row-heavy datasets */}
        {breakdown.items.length > 0 && (
          <div className="mt-6 w-full max-w-2xl rounded-2xl border border-white/10 bg-white/[0.03] p-5 backdrop-blur-md">
            <div className="mb-3 flex items-center justify-between text-[11px] uppercase tracking-widest text-white/40">
              <span className="flex items-center gap-2">
                <i className="fas fa-stopwatch text-amber-300/70" />
                Timing breakdown
              </span>
              <span className="tabular-nums text-white/60">
                Total {formatDuration(elapsed * 1000)}
              </span>
            </div>
            <div className="space-y-1.5">
              {breakdown.items.map((item) => {
                const pct = (item.ms / breakdown.maxMs) * 100;
                const isSlowest = breakdown.slowest && breakdown.slowest.key === item.key;
                return (
                  <div key={item.key} className="flex items-center gap-3 text-xs">
                    <span className={`w-32 truncate ${isSlowest ? 'text-amber-200' : 'text-white/70'}`} title={PHASE_LABEL[item.key] || item.key}>
                      {PHASE_LABEL[item.key] || item.key}
                    </span>
                    <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-white/10">
                      <div
                        className={[
                          'absolute inset-y-0 left-0 rounded-full transition-all duration-500',
                          item.live
                            ? 'bg-gradient-to-r from-blue-400 to-fuchsia-400 proc-bar-shimmer-fast'
                            : isSlowest
                              ? 'bg-gradient-to-r from-amber-400 to-orange-400'
                              : 'bg-gradient-to-r from-emerald-400 to-cyan-400',
                        ].join(' ')}
                        style={{ width: `${Math.max(2, Math.min(100, pct))}%` }}
                      />
                    </div>
                    <span className={`w-16 text-right tabular-nums ${isSlowest ? 'text-amber-200' : 'text-white/70'}`}>
                      {formatDuration(item.ms)}
                      {item.live && <span className="ml-1 text-blue-300/70">·</span>}
                    </span>
                  </div>
                );
              })}
            </div>
            {breakdown.slowest && (
              <div className="mt-3 flex items-center gap-2 text-[11px] text-amber-200/80">
                <i className="fas fa-bolt" />
                Slowest step:
                <span className="font-medium text-amber-100">{PHASE_LABEL[breakdown.slowest.key]}</span>
                <span className="tabular-nums text-amber-200/60">({formatDuration(breakdown.slowest.ms)})</span>
              </div>
            )}
          </div>
        )}

        {/* Error state */}
        {error && (
          <div className="mt-10 flex w-full max-w-2xl flex-col items-center gap-4 rounded-2xl border border-red-400/30 bg-red-500/10 p-6 text-center backdrop-blur-md">
            <i className="fas fa-triangle-exclamation text-3xl text-red-300" />
            <div>
              <div className="text-lg font-semibold text-red-100">Something went sideways</div>
              <div className="mt-1 text-sm text-red-200/80">{error}</div>
            </div>
            <button
              onClick={() => navigate('/', { replace: true })}
              className="inline-flex items-center gap-2 rounded-lg border border-white/20 bg-white/10 px-4 py-2 text-sm text-white transition hover:bg-white/20"
            >
              <i className="fas fa-arrow-left" />
              Back to upload
            </button>
          </div>
        )}
      </div>

      <script dangerouslySetInnerHTML={{
        __html: `
          (function () {
            const root = document.getElementById('proc-particles');
            if (!root || root.dataset.seeded === '1') return;
            root.dataset.seeded = '1';
            const N = 55;
            for (let i = 0; i < N; i++) {
              const p = document.createElement('div');
              p.className = 'proc-particle';
              const size = Math.random() * 3 + 1;
              p.style.width = size + 'px';
              p.style.height = size + 'px';
              p.style.left = (Math.random() * 100) + '%';
              p.style.bottom = '-10px';
              p.style.animationDuration = (Math.random() * 14 + 14) + 's';
              p.style.animationDelay = (Math.random() * 10) + 's';
              p.style.opacity = (Math.random() * 0.5 + 0.2).toFixed(2);
              root.appendChild(p);
            }
          })();
        `
      }} />

      <style>{`
        /* Aurora gradient blobs */
        .proc-aurora {
          position: absolute;
          border-radius: 9999px;
          filter: blur(80px);
          opacity: 0.45;
          mix-blend-mode: screen;
          animation: proc-drift 22s ease-in-out infinite;
        }
        .proc-aurora-a { width: 520px; height: 520px; top: -120px; left: -120px; background: radial-gradient(circle, rgba(96,165,250,0.9), transparent 70%); }
        .proc-aurora-b { width: 600px; height: 600px; bottom: -160px; right: -160px; background: radial-gradient(circle, rgba(168,85,247,0.8), transparent 70%); animation-delay: -7s; }
        .proc-aurora-c { width: 460px; height: 460px; top: 30%; left: 45%; background: radial-gradient(circle, rgba(45,212,191,0.6), transparent 70%); animation-delay: -14s; }
        @keyframes proc-drift {
          0%, 100% { transform: translate(0, 0) scale(1); }
          33%      { transform: translate(60px, -40px) scale(1.08); }
          66%      { transform: translate(-50px, 50px) scale(0.95); }
        }

        /* Orb */
        .proc-orb {
          box-shadow:
            0 0 80px 12px rgba(96,165,250,0.55),
            0 0 160px 30px rgba(168,85,247,0.25),
            inset 0 0 40px rgba(255,255,255,0.18);
          animation: proc-orb-pulse 3.2s ease-in-out infinite;
        }
        @keyframes proc-orb-pulse {
          0%, 100% { transform: scale(1); filter: brightness(1); }
          50%      { transform: scale(1.06); filter: brightness(1.18); }
        }
        .proc-icon-swap { animation: proc-icon-pop 380ms cubic-bezier(0.2, 0.7, 0.2, 1); }
        @keyframes proc-icon-pop {
          0%   { opacity: 0; transform: scale(0.7) rotate(-12deg); }
          100% { opacity: 1; transform: scale(1) rotate(0); }
        }

        /* Concentric rings */
        .proc-ring {
          position: absolute;
          border-radius: 9999px;
          border: 1px solid rgba(255,255,255,0.10);
        }
        .proc-ring-1 { width: 180px; height: 180px; animation: proc-spin 14s linear infinite; border-top-color: rgba(96,165,250,0.6); }
        .proc-ring-2 { width: 224px; height: 224px; animation: proc-spin 22s linear infinite reverse; border-right-color: rgba(168,85,247,0.5); }
        .proc-ring-3 { width: 248px; height: 248px; animation: proc-spin 30s linear infinite; border-bottom-color: rgba(45,212,191,0.45); }
        .proc-ring-4 { width: 264px; height: 264px; animation: proc-spin 40s linear infinite reverse; border-left-color: rgba(251,191,36,0.35); }
        @keyframes proc-spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

        /* Expanding pulses */
        .proc-pulse {
          position: absolute;
          border-radius: 9999px;
          border: 1px solid rgba(96,165,250,0.4);
          width: 130px; height: 130px;
          opacity: 0;
          animation: proc-pulse-out 3s ease-out infinite;
        }
        .proc-pulse-b { animation-delay: 1s; border-color: rgba(168,85,247,0.4); }
        .proc-pulse-c { animation-delay: 2s; border-color: rgba(45,212,191,0.4); }
        @keyframes proc-pulse-out {
          0%   { transform: scale(0.7); opacity: 0.75; }
          80%  { opacity: 0.1; }
          100% { transform: scale(2.1); opacity: 0; }
        }

        /* Orbiting dots */
        .proc-orbit {
          position: absolute;
          inset: 0;
          border-radius: 9999px;
          animation: proc-spin 10s linear infinite;
        }
        .proc-orbit-b { animation: proc-spin 16s linear infinite reverse; }
        .proc-orbit-c { animation: proc-spin 24s linear infinite; }
        .proc-orbit-dot {
          position: absolute;
          top: 4px;
          left: 50%;
          width: 8px; height: 8px;
          border-radius: 9999px;
          background: rgba(96,165,250,0.95);
          box-shadow: 0 0 12px 2px rgba(96,165,250,0.7);
          transform: translateX(-50%);
        }
        .proc-orbit-dot-2 { background: rgba(168,85,247,0.95); box-shadow: 0 0 12px 2px rgba(168,85,247,0.7); }
        .proc-orbit-dot-3 { background: rgba(45,212,191,0.95); box-shadow: 0 0 12px 2px rgba(45,212,191,0.7); }

        /* Active phase dot glow */
        .proc-active-dot {
          box-shadow: 0 0 0 0 rgba(96,165,250,0.55);
          animation: proc-dot-glow 1.6s ease-out infinite;
        }
        @keyframes proc-dot-glow {
          0%   { box-shadow: 0 0 0 0 rgba(96,165,250,0.55); }
          70%  { box-shadow: 0 0 0 14px rgba(96,165,250,0); }
          100% { box-shadow: 0 0 0 0 rgba(96,165,250,0); }
        }

        /* Phase headline fade */
        .proc-phase-fade { animation: proc-phase-in 380ms cubic-bezier(0.2, 0.7, 0.2, 1); }
        @keyframes proc-phase-in {
          from { opacity: 0; transform: translateY(8px); }
          to   { opacity: 1; transform: translateY(0); }
        }

        /* Quote fade */
        .proc-quote-fade { animation: proc-quote-in 600ms cubic-bezier(0.2, 0.7, 0.2, 1); }
        @keyframes proc-quote-in {
          from { opacity: 0; transform: translateY(6px); filter: blur(2px); }
          to   { opacity: 1; transform: translateY(0); filter: blur(0); }
        }
        .proc-quote-shimmer {
          background: linear-gradient(110deg, transparent 30%, rgba(255,255,255,0.06) 50%, transparent 70%);
          background-size: 220% 100%;
          animation: proc-shimmer 6s linear infinite;
        }
        @keyframes proc-shimmer {
          0%   { background-position: 200% 0; }
          100% { background-position: -200% 0; }
        }

        /* Progress bar */
        .proc-bar { box-shadow: 0 0 16px rgba(96,165,250,0.6); }
        .proc-bar-shimmer {
          background: linear-gradient(110deg, transparent 40%, rgba(255,255,255,0.35) 50%, transparent 60%);
          background-size: 200% 100%;
          animation: proc-shimmer 2.4s linear infinite;
          mix-blend-mode: overlay;
        }

        /* Live dot */
        .proc-live-dot {
          display: inline-block;
          width: 8px; height: 8px;
          border-radius: 9999px;
          background: rgb(96,165,250);
          box-shadow: 0 0 0 0 rgba(96,165,250,0.7);
          animation: proc-dot-glow 1.4s ease-out infinite;
        }

        /* Feed */
        .proc-feed-item { animation: proc-feed-in 320ms ease-out; }
        @keyframes proc-feed-in {
          from { opacity: 0; transform: translateY(4px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .proc-feed-dot { animation: proc-dot-glow 1.6s ease-out infinite; }
        .proc-feed::-webkit-scrollbar { width: 4px; }
        .proc-feed::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 9999px; }

        /* Particles */
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

        /* Floating glyphs */
        .proc-glyph {
          animation: proc-glyph-bob ease-in-out infinite;
        }
        @keyframes proc-glyph-bob {
          0%, 100% { transform: translateY(0) rotate(0deg); }
          50%      { transform: translateY(-22px) rotate(8deg); }
        }

        /* Live timing chip — subtle pulse while phase is running */
        .proc-time-pulse {
          animation: proc-time-pulse-kf 1.6s ease-in-out infinite;
        }
        @keyframes proc-time-pulse-kf {
          0%, 100% { opacity: 0.85; }
          50%      { opacity: 1; }
        }

        /* Faster shimmer on the live phase bar in the breakdown */
        .proc-bar-shimmer-fast {
          position: relative;
        }
        .proc-bar-shimmer-fast::after {
          content: '';
          position: absolute;
          inset: 0;
          background: linear-gradient(110deg, transparent 40%, rgba(255,255,255,0.45) 50%, transparent 60%);
          background-size: 200% 100%;
          animation: proc-shimmer 1.6s linear infinite;
          mix-blend-mode: overlay;
        }
      `}</style>
    </div>
  );
};

export default ProcessingPage;
