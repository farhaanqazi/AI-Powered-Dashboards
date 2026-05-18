import { defineConfig, devices } from '@playwright/test';

// Phase 12 S12.2 — Playwright smoke. Browsers are NOT downloaded in this
// environment (mirrors the billing/infra-gated CI posture); run
// `npx playwright install chromium` where browser download is permitted.
// `npm run dev` must be serving on :5173 (or set PW_BASE_URL).
export default defineConfig({
  testDir: './tests/e2e',
  timeout: 30_000,
  retries: 0,
  use: {
    baseURL: process.env.PW_BASE_URL || 'http://localhost:5173',
    headless: true,
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
});
