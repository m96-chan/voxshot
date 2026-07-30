import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
    coverage: {
      provider: "v8",
      reporter: ["text"],
      // main.ts and the workers drive real browser APIs (DOM, Web Worker,
      // WebGPU) and can only be exercised in a browser; unit-testable logic
      // lives in extracted modules like model-cache.ts.
      include: ["src/model-cache.ts"],
      thresholds: {
        statements: 90,
        branches: 90,
        functions: 90,
        lines: 90,
      },
    },
  },
});
