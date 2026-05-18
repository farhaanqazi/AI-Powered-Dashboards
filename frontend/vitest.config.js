import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

// Phase 12 S12.2 — unit/component test net. Kept separate from vite.config.js
// so the production build path is untouched.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.js'],
    include: ['src/**/*.{test,spec}.{js,jsx}'],
    css: false,
  },
});
