import { defineConfig } from "vite";

export default defineConfig({
  resolve: {
    // zerovox is linked via file:../.., so its lazy
    // import("@huggingface/transformers") would otherwise resolve from the
    // repository root (where the optional peer is not installed) and end up
    // as an empty stub. Dedupe forces resolution from this example's
    // node_modules.
    dedupe: ["@huggingface/transformers"],
  },
  // Transformers.js resolves its WASM / worker assets relative to its own
  // module URL; pre-bundling would break those paths, so serve it as-is.
  optimizeDeps: {
    exclude: ["@huggingface/transformers"],
  },
});
