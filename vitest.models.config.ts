import { defineConfig } from "vitest/config";

/**
 * The pre-release gate: everything that has to be checked against the real
 * model.
 *
 * Run with `npm run test:models`, never by CI. These tests need ~1.1 GB of
 * weights and golden files dumped from the reference implementation, none of
 * which a CI runner can produce.
 *
 * Two decisions are deliberate and worth keeping.
 *
 * **They live in their own config, not behind a skip.** Making the CI suite
 * include them and skip when the fixtures are absent would let a green run
 * mean "checked" and "did not check" interchangeably, which is the failure
 * mode CLAUDE.md exists to prevent. Instead the CI suite does not know this
 * directory exists (`vitest.config.ts` excludes it), and a missing fixture
 * here is a failure carrying the command that rebuilds it.
 *
 * **This is a release gate, not a nicety.** CI cannot catch a numerical
 * regression in a ported kernel — no weights, no goldens, no GPU. Only this
 * suite can. Leaving that implicit invites "CI is green, ship it", which would
 * be true of a build that had silently started producing noise.
 */
export default defineConfig({
  test: {
    include: ["tests/models/**/*.test.ts"],
    // Real weights: loading and unpacking alone runs into seconds.
    testTimeout: 120_000,
    hookTimeout: 600_000,
    // No coverage thresholds. Coverage is the CI suite's job; this suite is
    // measured by whether the model still sounds right, not by lines reached.
  },
});
