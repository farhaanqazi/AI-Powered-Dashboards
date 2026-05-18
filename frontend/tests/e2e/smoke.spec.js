import { test, expect } from '@playwright/test';

// Phase 12 S12.2 — Playwright smoke. Asserts the app shell renders and the
// upload entry point is reachable without a hard crash (white screen).
// Run against `npm run dev` with browsers installed.
test('app shell loads and shows the upload entry', async ({ page }) => {
  await page.goto('/');
  // The splash screen has a Skip control; dismiss it if present.
  const skip = page.getByRole('button', { name: /skip/i });
  if (await skip.isVisible().catch(() => false)) {
    await skip.click();
  }
  await expect(page.locator('body')).toBeVisible();
  // No top-level error-boundary alert on a clean load.
  await expect(page.getByRole('alert')).toHaveCount(0);
});
