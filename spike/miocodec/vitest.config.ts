import { defineConfig } from "vitest/config";

// web-xpu-ops ships raw TypeScript (`./ops/*` -> `ops/*/index.ts`) and is
// unpublished, so it arrives here as a `file:` symlink. Vitest does not
// transform node_modules by default, and a symlinked source tree is still
// node_modules as far as that rule is concerned — hence the inline.
export default defineConfig({
  test: {
    include: ["*.test.ts"],
    server: { deps: { inline: [/web-xpu-ops/] } },
  },
});
