import { create } from 'zustand';
import { getDashboardData, patchRegistry } from './services/api';

const GUEST_KEY = 'dataInsight:guestMode';
const GUEST_SID_KEY = 'dataInsight:guestSessionId';
const readGuest = () => {
  try { return typeof localStorage !== 'undefined' && localStorage.getItem(GUEST_KEY) === '1'; }
  catch { return false; }
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
  isGuest: readGuest(),
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
      set({ exporting: false });
    }
  },
  async refresh() {
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
}));
