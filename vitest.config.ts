import { defaultExclude, defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
    // tests/models needs ~1.1 GB of weights and goldens dumped from the
    // reference implementation, so CI can never run it. It is excluded rather
    // than skipped from the inside: a suite that goes green whether or not it
    // checked anything is worse than one CI simply does not have. It runs as
    // `npm run test:models`, a release gate — see vitest.models.config.ts.
    exclude: [...defaultExclude, "tests/models/**"],
    coverage: {
      provider: "v8",
      reporter: ["text", "lcov"],
      include: ["src/**/*.ts"],
      // Entry points only: they re-export and hold no logic of their own,
      // and tests/index.test.ts and tests/engine/miotts/index.test.ts assert
      // the surface each one promises.
      exclude: ["src/**/*.d.ts", "src/index.ts", "src/engine/miotts/index.ts"],
      thresholds: {
        statements: 90,
        branches: 90,
        functions: 90,
        lines: 90,
      },
    },
  },
});
