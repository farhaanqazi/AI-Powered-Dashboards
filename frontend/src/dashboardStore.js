import { create } from 'zustand';
import { getDashboardData } from './services/api';

export const useDashboardStore = create((set, get) => ({
  data: null,
  loading: false,
  error: null,
  lastUpdated: 0,
  exporting: false,
  exportHandler: null,
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
}));
