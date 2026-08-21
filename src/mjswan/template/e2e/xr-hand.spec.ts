import { test, expect } from '@playwright/test';

// Physics-tier acceptance for VR hand tracking: the injected rig has to compile, a
// tracked hand has to push a free body, an untracked one has to touch nothing, and a
// pinch has to carry the body up and let it fall again.
test('a mocap hand pushes, holds and releases a free body', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (err) => pageErrors.push(String(err)));

  await page.goto('/xr-hand.html');
  await page.waitForFunction(() => window.__xrHarness !== undefined, undefined, {
    timeout: 60_000,
  });

  const result = await page.evaluate(() => window.__xrHarness);
  expect(result?.ok, result?.error).toBe(true);
  expect(result?.bound).toBe(true);
  // Two hands of 11 joints each, plus one weld per hand.
  expect(result?.bodies?.mocap).toBe(22);
  expect(result?.bodies?.equalities).toBe(2);

  // Contact: the sweep must move the cube, and must not while the hand is untracked.
  expect(result?.pushed ?? 0).toBeGreaterThan(0.02);
  expect(Math.abs(result?.pushedWhileUntracked ?? 1)).toBeLessThan(0.001);

  // Grasp: the weld holds, and the cube rises with the hand rather than trailing it.
  expect(result?.grabbedDuringLift).toBe(true);
  expect(result?.lifted ?? 0).toBeGreaterThan(0.15);
  expect(result?.lifted ?? 0).toBeCloseTo(result?.handLifted ?? 0, 1);

  // Release: the constraint lets go and gravity takes over.
  expect(result?.grabbedAfterRelease).toBe(false);
  expect(result?.dropped ?? 0).toBeGreaterThan(0.1);

  expect(pageErrors).toEqual([]);
});
