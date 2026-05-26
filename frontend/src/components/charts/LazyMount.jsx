import React, { useEffect, useRef, useState } from 'react';
import { useDashboardStore } from '../../dashboardStore';

// Phase 14 S14.3 — defer the heavy Plotly mount until the card scrolls near
// the viewport. The Overview tab can queue ~9–14 charts; mounting them all in
// one frame locks the main thread (the visible mouse-lag the user reported).
// IntersectionObserver caps concurrent Plotly inits to what is actually on
// screen. Once shown we keep it mounted (no re-init churn on scroll-back).
//
// Eager (immediate) mount happens when: `eager` is set, a PDF export is in
// flight (the export surface lives at left:-100000px where an observer never
// fires — it must paint all charts NOW), or IntersectionObserver is absent
// (jsdom in tests, very old browsers).
export default function LazyMount({ children, eager = false, minHeight = 240, rootMargin = '300px' }) {
  const ref = useRef(null);
  const exporting = useDashboardStore((s) => s.exporting);
  const forceEager = eager || exporting;
  const [seen, setSeen] = useState(
    () => forceEager || typeof IntersectionObserver === 'undefined',
  );

  useEffect(() => {
    if (seen) return undefined;
    if (forceEager || typeof IntersectionObserver === 'undefined') {
      setSeen(true);
      return undefined;
    }
    const el = ref.current;
    if (!el) return undefined;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setSeen(true);
          io.disconnect();
        }
      },
      { rootMargin },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [seen, forceEager, rootMargin]);

  return (
    <div ref={ref} style={{ height: '100%', minHeight }}>
      {seen ? children : (
        <div
          aria-hidden="true"
          style={{
            height: '100%',
            minHeight,
            borderRadius: 12,
            background: 'rgba(148,163,184,0.06)',
          }}
        />
      )}
    </div>
  );
}
