/**
 * Copy the WGSL this port dispatches out of web-xpu-ops and into `kernels.ts`.
 *
 *     node gen-kernels.mjs
 *
 * Same reasoning as `spike/dacvae`'s: the source has to load under a bundler and
 * in a browser, and each has its own spelling for "import this file as text".
 * One generated module is understood by both.
 *
 * Every contract comment is copied from the kernel's own header — binding order
 * and uniform field order are the authority there, and nothing below is
 * re-derived. Getting one of them wrong does not throw; it computes something.
 */

import { createRequire } from "node:module";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

const require = createRequire(import.meta.url);
const packageJson = require.resolve("web-xpu-ops/package.json");
const UPSTREAM = dirname(packageJson);
const version = JSON.parse(readFileSync(packageJson, "utf8")).version;

const KERNELS = [
  {
    name: "MATMUL",
    op: "matmul",
    entry: "kernel",
    contract: [
      "bindings: [a f32 [M, K], b f32 [K, N], out f32 [M, N], uniform]",
      "uniform:  [u32 M, N, K]",
      "dispatch: [ceil(N / 16), ceil(M / 16)]   — TILE is 16 in the kernel",
    ],
  },
  {
    name: "RMSNORM",
    op: "rmsnorm",
    entry: "kernel",
    contract: [
      "bindings: [input f32 [N, D], weight f32 [G, D], out f32 [N, D], uniform]",
      "uniform:  [u32 N, u32 D, f32 eps, u32 G]",
      "dispatch: [N]   — one workgroup per row",
      "Row n uses weight row n % G. G = 0 is read as 1, for callers older than",
      "the group parameter; this port always passes it.",
    ],
  },
  {
    name: "LAYERNORM",
    op: "layernorm",
    entry: "kernel",
    contract: [
      "bindings: [input f32 [N, D], weight f32 [D], bias f32 [D], out f32 [N, D], uniform]",
      "uniform:  [u32 N, u32 D, f32 eps]",
      "dispatch: [N]",
      "The bias binding is required; ModernBERT has norm_bias: false, so this",
      "port passes zeros rather than leaving a binding unfilled.",
    ],
  },
  {
    name: "ROPE",
    op: "rope",
    entry: "kernel",
    contract: [
      "bindings: [input f32 [N, heads, head_dim], cache f32, out f32, uniform]",
      "uniform:  [u32 N, num_heads, head_dim, pos_offset, cache_positions,",
      "           f32 effective_base, interpolation_factor, ramp_low, ramp_high,",
      "           attention_factor, u32 head_offset, head_count]",
      "dispatch: [ceil(N * num_heads * head_dim / 2 / 256)]",
      "cache_positions 0 means no cache; the `cache` binding still has to exist,",
      "so this port binds a one-element buffer.",
      "head_offset/head_count are how `_apply_rotary_half` is expressed: heads",
      "outside the range are copied through, not zeroed.",
    ],
  },
  {
    name: "ACTIVATION",
    op: "activation",
    entry: "kernel",
    contract: [
      "bindings: [input f32 [N], out f32 [N], uniform]",
      "uniform:  [u32 N, u32 activation_type, f32 alpha]",
      "dispatch: [ceil(N / 256)]",
      "type 1 = SiLU, 4 = GELU (erf, torch's default), 5 = GELU tanh.",
      "4 and 5 are different functions, not two spellings of one.",
    ],
  },
  {
    name: "GATHER",
    op: "gather",
    entry: "kernel",
    contract: [
      "bindings: [table f32 [rows, D], indices i32 [N], out f32 [N, D], uniform]",
      "uniform:  [u32 N, u32 D, u32 rows]",
      "dispatch: [ceil(N * D / 256)]",
      "`rows` is the TABLE's row count. An index outside [0, rows) yields a row",
      "of zeros, silently — which is how the CPU path first embedded nothing.",
    ],
  },
  {
    name: "ELEMENTWISE",
    op: "elementwise",
    entry: "kernel",
    contract: [
      "bindings: [a f32 [N], b f32 [N], out f32 [N], uniform]",
      "uniform:  [u32 N, u32 kind]",
      "dispatch: [ceil(N / 256)]",
      "kind 0 = add, 1 = multiply. Same length on both sides — no broadcast,",
      "which is why this port expands with `gather` where it needs one.",
    ],
  },
  {
    name: "PERMUTE",
    op: "permute",
    entry: "kernel",
    contract: [
      "bindings: [input f32 [dim0, dim1, D], out f32 [dim1, dim0, D], uniform]",
      "uniform:  [u32 dim0, u32 dim1, u32 D]",
      "dispatch: [ceil(dim0 * dim1 * D / 256)]",
      "Swaps the two leading axes and keeps D contiguous — exactly the",
      "[tokens, heads, headDim] <-> [heads, tokens, headDim] move attention needs.",
    ],
  },
  {
    name: "ATTENTION_SCORES",
    op: "attention",
    entry: "scores",
    contract: [
      "bindings: [q f32 [B, H, L, D], k f32 [B, H, S, D], mask f32, probs f32 [B, H, L, S], uniform]",
      "uniform:  [u32 H, L, S, D, f32 scale, u32 causal, i32 query_offset,",
      "           u32 mask_batch, mask_heads, mask_rows]",
      "dispatch: [L, H, B]   — one workgroup per query row",
      "Softmax is inside this kernel; `probs` comes out normalised.",
      "The mask is an additive bias of shape [mask_batch, mask_heads, mask_rows]",
      "x S, each of the first three either 1 or its full extent.",
    ],
  },
  {
    name: "ATTENTION_CONTEXT",
    op: "attention",
    entry: "context",
    contract: [
      "bindings: [probs f32 [B, H, L, S], v f32 [B, H, S, Dv], out f32 [B, H, L, Dv], uniform]",
      "uniform:  [u32 H, L, S, Dv]",
      "dispatch: [L, H, B]",
    ],
  },
];

const parts = [
  "// Generated by gen-kernels.mjs — do not edit.",
  `// Source: web-xpu-ops @ ${version} (from node_modules, pinned by package-lock.json)`,
  "//",
  "// WGSL for every kernel entry point the Irodori port dispatches. Contract",
  "// comments are copied from each kernel's own header, which is the authority",
  "// for binding and uniform order; nothing is re-derived here.",
  "",
];

for (const { name, op, entry, contract } of KERNELS) {
  const source = readFileSync(join(UPSTREAM, "dist", "ops", op, "wgsl", `${entry}.wgsl`), "utf8");
  parts.push("/**");
  parts.push(` * ops/${op}/wgsl/${entry}.wgsl`);
  parts.push(" *");
  for (const line of contract) parts.push(` * ${line}`);
  parts.push(" */");
  // `JSON.stringify` rather than a template literal: these kernels carry prose
  // comments and backticks. Escaping by hand is a rule someone has to
  // remember; quoting is a rule nobody can forget.
  parts.push(`export const ${name} = ${JSON.stringify(source)};`);
  parts.push("");
}

parts.push(`export const UPSTREAM_VERSION = ${JSON.stringify(version)};`);
parts.push("");

const out = new URL("./kernels.ts", import.meta.url).pathname;
writeFileSync(out, parts.join("\n"));
console.log(`wrote ${out} from web-xpu-ops @ ${version} — ${KERNELS.length} kernels`);
