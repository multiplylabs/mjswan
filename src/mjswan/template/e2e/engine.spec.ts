import { test, expect } from '@playwright/test';

// Runtime-tier acceptance: a real browser loads the React-free harness, `createEngine`
// builds a scene from `.mjz` bytes, and the captured frame must be a non-blank render.
test('createEngine renders a scene from bytes, React-free', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (err) => pageErrors.push(String(err)));
  // The XR hand rig is spliced into every scene a WebXR-capable browser loads, and
  // headless Chromium is one. Any complaint means the splice stopped compiling.
  const xrWarnings: string[] = [];
  page.on('console', (message) => {
    if (message.text().includes('[XR]')) xrWarnings.push(message.text());
  });

  await page.goto('/harness.html');
  await page.waitForFunction(() => window.__harness !== undefined, undefined, { timeout: 60_000 });

  const result = await page.evaluate(() => window.__harness);
  expect(result?.ok, result?.error).toBe(true);
  expect(result?.running).toBe(true);
  expect(result?.nonBlank, `luminance range ${JSON.stringify(result?.luminanceRange)}`).toBe(true);
  // An app has to be able to choose the term seed and read back the one in use, or a
  // recorded session has nothing to replay from. The harness's 0xc0ffee differs from
  // the built-in default, so a dropped option shows up rather than falling back.
  expect(result?.termSeed).toBe(0xc0ffee);
  expect(xrWarnings).toEqual([]);
  expect(pageErrors).toEqual([]);
});
