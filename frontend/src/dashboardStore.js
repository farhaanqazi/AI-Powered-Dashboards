import { create } from 'zustand';
import { getDashboardData, patchRegistry, grantAiConsent } from './services/api';

const GUEST_KEY = 'dataInsight:guestMode';
const GUEST_SID_KEY = 'dataInsight:guestSessionId';
const THEME_KEY = 'dataInsight:theme';
const readGuest = () => {
  try { return typeof localStorage !== 'undefined' && localStorage.getItem(GUEST_KEY) === '1'; }
  catch { return false; }
};
const readTheme = () => {
  try {
    const t = localStorage.getItem(THEME_KEY);
    return t === 'light' ? 'light' : 'dark';
  } catch { return 'dark'; }
};
// Mirror the theme onto <html data-theme> so global (non-.dash-shell) surfaces
// — e.g. the body-portaled chart modal — can react via CSS too.
const applyThemeToDom = (theme) => {
  try { document.documentElement.setAttribute('data-theme', theme); } catch {}
};
const ensureGuestSessionId = () => {
  try {
    let sid = localStorage.getItem(GUEST_SID_KEY);
    if (!sid) {
      sid = (crypto?.randomUUID?.() || `g-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`);
      localStorage.setItem(GUEST_SID_KEY, sid);
    }
    return sid;
  } catch { return 'anonymous'; }
};

export const useDashboardStore = create((set, get) => ({
  data: null,
  loading: false,
  error: null,
  lastUpdated: 0,
  exporting: false,
  exportHandler: null,
  // { index, total, label } while a PDF export runs; null otherwise.
  exportProgress: null,
  setExportProgress: (p) => set({ exportProgress: p || null }),
  isGuest: readGuest(),
  // 'dark' (default futuristic) | 'light' (white contrast theme).
  theme: readTheme(),
  setTheme: (theme) => {
    const t = theme === 'light' ? 'light' : 'dark';
    try { localStorage.setItem(THEME_KEY, t); } catch {}
    applyThemeToDom(t);
    set({ theme: t });
  },
  toggleTheme: () => get().setTheme(get().theme === 'dark' ? 'light' : 'dark'),
  enableGuest: () => {
    try { localStorage.setItem(GUEST_KEY, '1'); } catch {}
    ensureGuestSessionId();
    set({ isGuest: true });
  },
  disableGuest: () => {
    try { localStorage.removeItem(GUEST_KEY); } catch {}
    set({ isGuest: false });
  },
  setExportHandler: (fn) => set({ exportHandler: fn }),
  async runExport() {
    const fn = get().exportHandler;
    if (!fn || get().exporting) return;
    set({ exporting: true });
    try {
      await fn();
    } catch (err) {
      console.error('PDF export failed', err);
    } finally {
      set({ exporting: false, exportProgress: null });
    }
  },
  async refresh() {
    // The PDF export cycles activeTab through every tab; the tab-change
    // effect must NOT trigger a backend refetch each time (4 redundant
    // round-trips per export). Data is already loaded — keep it.
    if (get().exporting) return;
    if (get().loading) return;
    set({ loading: true, error: null });
    try {
      const data = await getDashboardData();
      set({ data, loading: false, error: null, lastUpdated: Date.now() });
    } catch (err) {
      set({
        loading: false,
        error: err?.response?.data?.detail || err.message || 'Failed to load dashboard data',
      });
    }
  },
  // Phase 7 (S7.5): true while the human schema review is being submitted.
  reviewSubmitting: false,
  reviewError: null,
  // The data-quality verdict the backend threaded into the payload.
  schemaReview() {
    return get().data?.dataset_profile?.data_quality?.report || null;
  },
  needsSchemaReview() {
    const r = get().schemaReview();
    return !!r && r.status && r.status !== 'ok';
  },
  async submitSchemaReview(overrides) {
    if (get().reviewSubmitting) return;
    set({ reviewSubmitting: true, reviewError: null });
    try {
      const resp = await patchRegistry(get().data?.trace_id, overrides);
      set({
        data: resp.data || resp,
        reviewSubmitting: false,
        lastUpdated: Date.now(),
      });
    } catch (err) {
      set({
        reviewSubmitting: false,
        reviewError:
          err?.response?.data?.detail || err.message || 'Schema review failed',
      });
      throw err;
    }
  },
  // PII consent: the dashboard already built; this opts AI Insights in.
  aiConsentSubmitting: false,
  aiConsentError: null,
  async grantAiConsent() {
    if (get().aiConsentSubmitting) return;
    set({ aiConsentSubmitting: true, aiConsentError: null });
    try {
      const resp = await grantAiConsent(get().data?.trace_id);
      set({
        data: resp.data || resp,
        aiConsentSubmitting: false,
        lastUpdated: Date.now(),
      });
    } catch (err) {
      set({
        aiConsentSubmitting: false,
        aiConsentError:
          err?.response?.data?.detail || err.message ||
          'Could not enable AI for this dataset',
      });
      throw err;
    }
  },

  // --- Phase 14 S14.3: Stage 1 interaction state -------------------------
  // Cross-highlight + simple narrowing run client-side over data already in
  // the shipped chart specs (zero server call, not WASM). Server-backed
  // recompute (new aggregations) goes through services.runInteraction.
  filters: [],
  highlight: null,        // { column, value } | null
  excludedKeys: [],
  addFilter(filter) {
    if (!filter || !filter.column) return;
    const rest = get().filters.filter(
      (f) => !(f.column === filter.column && f.op === (filter.op || 'eq')),
    );
    set({ filters: [...rest, { op: 'eq', ...filter }] });
  },
  removeFilter(column, op = 'eq') {
    set({
      filters: get().filters.filter(
        (f) => !(f.column === column && f.op === op),
      ),
    });
  },
  clearFilters: () => set({ filters: [], highlight: null }),
  setHighlight: (highlight) => set({ highlight: highlight || null }),
  toggleExclude(key) {
    const k = String(key);
    const cur = get().excludedKeys;
    set({
      excludedKeys: cur.includes(k)
        ? cur.filter((x) => x !== k)
        : [...cur, k],
    });
  },
  hasInteractions() {
    const s = get();
    return s.filters.length > 0 || !!s.highlight || s.excludedKeys.length > 0;
  },
}));

// Reflect the persisted theme onto <html> immediately so the first paint
// (modal CSS, etc.) matches the stored preference without a flash.
applyThemeToDom(readTheme());
