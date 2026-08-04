import { defineConfig } from "vitest/config";

// web-xpu-ops ships raw TypeScript (`./ops/*` -> `ops/*/index.ts`) and is
// unpublished, so it arrives here as a `file:` symlink. Vitest does not
// transform node_modules by default, and a symlinked source tree is still
// node_modules as far as that rule is concerned — hence the inline.
//
// No GPU tests live here. Dawn's native module runs fine in plain Node but kills
// a Vitest 4 worker before a single test executes — measured, in 2.6 seconds,
// which is before any GPU work starts. Upstream hit the same wall on Vitest 2
// and lists four pool configurations that did not help. The GPU check is
// `npm run check:gpu`, a script that exits non-zero.
export default defineConfig({
  test: {
    include: ["*.test.ts"],
    testTimeout: 120_000,
    hookTimeout: 600_000,
    server: { deps: { inline: [/web-xpu-ops/] } },
  },
});
