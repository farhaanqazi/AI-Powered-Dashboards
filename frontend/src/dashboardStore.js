import { create } from 'zustand';
import { getDashboardData } from './services/api';

export const useDashboardStore = create((set, get) => ({
  data: null,
  loading: false,
  error: null,
  lastUpdated: 0,
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
