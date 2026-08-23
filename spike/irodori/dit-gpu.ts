import { ACTIVATION } from "web-xpu-ops/ops/activation";

import type { Context, DitBlockWeights, DitShape } from "./dit.js";
import type { Gpu, Tensor } from "./gpu.js";
import type { ModelWeights } from "./model-weights.js";

/**
 * The DiT on the device — the same graph as `dit.ts`, with nothing crossing
 * back to JavaScript inside a block.
 *
 * `dit.ts` is the definition of correct and `check-dit.ts` compares this
 * against the same goldens, so the two must stay the same graph. Where they
 * differ it is in *how* an expression is reached, never in which one.
 *
 * ## Two rewrites, both because a kernel does not exist
 *
 * **Broadcast is a `gather`.** `elementwise` takes two operands of equal
 * length; AdaLN needs `x[b, t, d] * (1 + scale[b, d]) + shift[b, d]`. Rather
 * than write a kernel, the `[batch, dim]` vector is expanded to
 * `[batch * tokens, dim]` by gathering row `b` once per token — one dispatch,
 * no new WGSL.
 *
 * Its **per-member** part is unverified, and saying so is worth more than a
 * green check. Guidance varies the *context*, not the timestep, so
 * `cond_module`'s three batch members are bit-identical and every member wants
 * the same modulation. Replacing the indices with all-zeros changes nothing and
 * `check-dit-gpu.ts` still passes. It is written per member because the shape
 * says so, not because anything here has demonstrated it matters.
 *
 * **`sigmoid` is folded into two weights.** The attention gate is
 * `y * sigmoid(g)`, and web-xpu-ops has `tanh` but not `sigmoid`. With
 * `sigmoid(g) = (tanh(g/2) + 1) / 2`,
 *
 *     wo(y * sigmoid(g)) = (wo/2) . (y * (tanh((Wgate/2) . h) + 1))
 *
 * so halving `Wgate` and `wo` **at load** removes every constant. This is the
 * one place the port changes a weight to suit an operation, which is worth
 * knowing before comparing a tensor against the checkpoint.
 *
 * ## What stays on the host
 *
 * The timestep embedding, `cond_module` and AdaLN's low-rank refinement. They
 * are `[batch, dim]`-sized — about 10 MFLOP a step against the DiT's 40 GFLOP —
 * so moving them would add dispatches and complexity to buy nothing. The host
 * hands down three `[batch, dim]` vectors per AdaLN, with `1 + scale` and
 * `tanh(gate)` already applied.
 *
 * ## Slots
 *
 * Scratch buffers are pooled by name, so two tensors live at the same moment
 * must not share one. Every block and every step reuses the same set, which is
 * what keeps fifteen thousand dispatches from building fifteen thousand bind
 * groups.
 */

/** `elementwise`'s kinds. */
const ADD = 0;
const MULTIPLY = 1;

export interface GpuBlockWeights {
  wq: Tensor;
  wk: Tensor;
  wv: Tensor;
  /** Halved — see the module note. */
  gate: Tensor;
  /** Halved. */
  wo: Tensor;
  qNorm: Tensor;
  kNorm: Tensor;
  w1: Tensor;
  w2: Tensor;
  w3: Tensor;
  /** Context keys and values for this block, already concatenated. */
  contextK: Tensor;
  contextV: Tensor;
}

export interface GpuDit {
  gpu: Gpu;
  shape: DitShape;
  batch: number;
  tokens: number;
  /** Context tokens per batch member. */
  contextTokens: number;
  latentDim: number;
  blocks: GpuBlockWeights[];
  outNorm: Tensor;
  outProj: Tensor;
  outProjBias: Tensor;
  inProj: Tensor;
  inProjBias: Tensor;
  /** `[batch * tokens]` of `b`, to expand a per-member vector over tokens. */
  broadcast: Tensor;
  /** `[batch * tokens, dim]` of ones, for `tanh(g) + 1`. */
  ones: Tensor;
  /** `[batch, keys]` additive bias: 0 for the latent, -inf for masked context. */
  keyBias: Tensor;
  /** `[batch * tokens, dim]` of the in_proj bias, expanded once. */
  inProjBiasRows: Tensor;
  /** `[batch * tokens, latentDim]` of the out_proj bias, expanded once. */
  outProjBiasRows: Tensor;
}

function halve(source: Float32Array): Float32Array {
  const out = new Float32Array(source.length);
  for (let i = 0; i < source.length; i += 1) out[i] = source[i]! * 0.5;
  return out;
}

/** One row per output row, so a `[rows, D]` bias can be added with `elementwise`. */
function repeatRows(vector: Float32Array, rows: number): Float32Array {
  const out = new Float32Array(rows * vector.length);
  for (let row = 0; row < rows; row += 1) out.set(vector, row * vector.length);
  return out;
}

/**
 * Upload everything that does not change, once.
 *
 * The context projections are the expensive part and the reason this is
 * separate from the step: `wk`/`wv` run over 892 tokens, more than anything in
 * a step, and they are identical across all 32 of them.
 */
export function prepareGpu(args: {
  gpu: Gpu;
  weights: ModelWeights;
  contexts: Record<string, Context>;
  batch: number;
  tokens: number;
}): GpuDit {
  const { gpu, weights, contexts, batch, tokens } = args;
  const { dit, config } = weights;
  const { dim, heads, eps } = dit.shape;
  const headDim = dim / heads;

  const blocks: GpuBlockWeights[] = [];
  let contextTokens = 0;
  let sharedKeep: boolean[] = [];

  dit.blocks.forEach((block: DitBlockWeights, index: number) => {
    const parts: { k: Tensor; v: Tensor; keep: boolean[]; tokens: number }[] = [];
    let total = 0;
    const kNorm = gpu.weight(block.kNorm);
    // Insertion order is the reference's concat order: text, speaker, caption.
    for (const name of Object.keys(block.contexts)) {
      const context = contexts[name];
      if (!context) throw new Error(`block wants a ${name} context and none was given`);
      const { wk, wv, dim: contextDim } = block.contexts[name]!;
      const rows = context.keep.length;
      if (rows % batch !== 0) throw new Error(`${name} context has ${rows} rows, not a multiple of ${batch}`);
      // Pooled, not freshly allocated: a server renders many utterances and
      // `alloc` never frees. The text state changes every request, the shape
      // does not.
      const state = gpu.writeInto(`prep.state.${batch}.${index}.${name}`, context.state);
      const raw = gpu.matmul(state, gpu.weight(wk), rows, dim, contextDim, `prep.k.${index}.${name}`);
      const k = gpu.rmsnorm(raw, kNorm, rows * heads, headDim, eps, `prep.kn.${index}.${name}`, heads);
      const v = gpu.matmul(state, gpu.weight(wv), rows, dim, contextDim, `prep.v.${index}.${name}`);
      parts.push({ k, v, keep: context.keep, tokens: rows / batch });
      total += rows / batch;
    }

    // Concatenated along tokens *within* each batch member, so the pieces
    // interleave rather than append.
    const contextK = gpu.scratch(`prep.ck.${batch}.${index}`, batch * total * dim);
    const contextV = gpu.scratch(`prep.cv.${batch}.${index}`, batch * total * dim);
    const keep = new Array<boolean>(batch * total);
    for (let b = 0; b < batch; b += 1) {
      let at = b * total;
      for (const part of parts) {
        const bytes = part.tokens * dim * 4;
        gpu.copy(part.k, b * part.tokens * dim * 4, contextK, at * dim * 4, bytes);
        gpu.copy(part.v, b * part.tokens * dim * 4, contextV, at * dim * 4, bytes);
        for (let j = 0; j < part.tokens; j += 1) keep[at + j] = part.keep[b * part.tokens + j]!;
        at += part.tokens;
      }
    }
    contextTokens = total;
    sharedKeep = keep;

    blocks.push({
      wq: gpu.weight(block.wq),
      wk: gpu.weight(block.wk),
      wv: gpu.weight(block.wv),
      gate: gpu.weight(halve(block.gate)),
      wo: gpu.weight(halve(block.wo)),
      qNorm: gpu.weight(block.qNorm),
      kNorm,
      w1: gpu.weight(block.w1),
      w2: gpu.weight(block.w2),
      w3: gpu.weight(block.w3),
      contextK,
      contextV,
    });
  });

  // The latent's own positions are never masked; `self_mask` defaults to all
  // ones and nothing in the sampling path overrides it.
  const keys = tokens + contextTokens;
  const bias = new Float32Array(batch * keys);
  for (let b = 0; b < batch; b += 1) {
    for (let j = 0; j < contextTokens; j += 1) {
      bias[b * keys + tokens + j] = sharedKeep[b * contextTokens + j] ? 0 : -Infinity;
    }
  }

  const indices = new Int32Array(batch * tokens);
  for (let b = 0; b < batch; b += 1) indices.fill(b, b * tokens, (b + 1) * tokens);

  return {
    gpu,
    shape: dit.shape,
    batch,
    tokens,
    contextTokens,
    latentDim: config.latent_dim,
    blocks,
    outNorm: gpu.weight(dit.outNorm),
    outProj: gpu.weight(dit.outProjWeight),
    outProjBias: gpu.weight(dit.outProjBias),
    inProj: gpu.weight(dit.inProjWeight),
    inProjBias: gpu.weight(dit.inProjBias),
    broadcast: gpu.writeIntsInto(`prep.idx.${batch}`, indices),
    ones: gpu.writeInto(`prep.ones.${batch}`, new Float32Array(batch * tokens * dim).fill(1)),
    keyBias: gpu.writeInto(`prep.bias.${batch}`, bias),
    inProjBiasRows: gpu.writeInto(`prep.inb.${batch}`, repeatRows(dit.inProjBias, batch * tokens)),
    outProjBiasRows: gpu.writeInto(`prep.outb.${batch}`, repeatRows(dit.outProjBias, batch * tokens)),
  };
}

/** `[batch, dim]` expanded to `[batch * tokens, dim]`. */
function spread(dit: GpuDit, vector: Tensor, slot: string): Tensor {
  const { gpu, batch, tokens, shape } = dit;
  return gpu.gather(vector, dit.broadcast, batch * tokens, shape.dim, batch, slot);
}

/** The three AdaLN vectors the host computed, as they arrive. */
export interface Modulation {
  /** `1 + scale`, `[batch, dim]`. */
  scale: Tensor;
  shift: Tensor;
  /** `tanh(gate)`, `[batch, dim]`. */
  gate: Tensor;
}

/** `rmsnorm(x) * scale + shift`, with scale and shift broadcast over tokens. */
function modulate(dit: GpuDit, x: Tensor, mod: Modulation, slot: string): Tensor {
  const { gpu, batch, tokens, shape } = dit;
  const rows = batch * tokens;
  const normed = gpu.rmsnorm(x, dit.ones, rows, shape.dim, shape.eps, `${slot}.norm`);
  const scaled = gpu.elementwise(normed, spread(dit, mod.scale, `${slot}.s`), MULTIPLY, `${slot}.mul`);
  return gpu.elementwise(scaled, spread(dit, mod.shift, `${slot}.f`), ADD, `${slot}.add`);
}

/** `x + gate * delta`, gate broadcast over tokens. */
function gatedAdd(dit: GpuDit, x: Tensor, delta: Tensor, gate: Tensor, slot: string): Tensor {
  const { gpu } = dit;
  const scaled = gpu.elementwise(delta, spread(dit, gate, `${slot}.g`), MULTIPLY, `${slot}.mul`);
  return gpu.elementwise(x, scaled, ADD, slot);
}

export function ditBlockGpu(
  dit: GpuDit,
  x: Tensor,
  attentionMod: Modulation,
  mlpMod: Modulation,
  index: number,
): Tensor {
  const { gpu, shape, batch, tokens, contextTokens } = dit;
  const { dim, heads, mlpHidden, eps } = shape;
  const headDim = dim / heads;
  const rows = batch * tokens;
  const keys = tokens + contextTokens;
  const block = dit.blocks[index]!;
  const at = (name: string) => `b.${name}`;

  const h = modulate(dit, x, attentionMod, at("attn"));

  // Q and K: per-head norm, then RoPE over the first half of the heads only.
  const q = rotateHalf(
    dit,
    gpu.rmsnorm(gpu.matmul(h, block.wq, rows, dim, dim, at("q")), block.qNorm, rows * heads, headDim, eps, at("qn"), heads),
    at("qr"),
  );
  const k = rotateHalf(
    dit,
    gpu.rmsnorm(gpu.matmul(h, block.wk, rows, dim, dim, at("k")), block.kNorm, rows * heads, headDim, eps, at("kn"), heads),
    at("kr"),
  );
  const v = gpu.matmul(h, block.wv, rows, dim, dim, at("v"));

  // Concatenate first, permute second — the order `dit.ts` uses, and the order
  // that matters. The context keys and values are token-major, so permuting the
  // latent's own before joining them would splice two different layouts into
  // one tensor. It runs, it is fast, and every block after the first is wrong.
  const kcat = gpu.scratch(at("kcat"), batch * keys * dim);
  const vcat = gpu.scratch(at("vcat"), batch * keys * dim);
  for (let b = 0; b < batch; b += 1) {
    const selfBytes = tokens * dim * 4;
    const ctxBytes = contextTokens * dim * 4;
    // Self keys first, then the contexts — `torch.cat([k_self, k_text, ...])`.
    gpu.copy(k, b * selfBytes, kcat, b * keys * dim * 4, selfBytes);
    gpu.copy(v, b * selfBytes, vcat, b * keys * dim * 4, selfBytes);
    gpu.copy(block.contextK, b * ctxBytes, kcat, (b * keys + tokens) * dim * 4, ctxBytes);
    gpu.copy(block.contextV, b * ctxBytes, vcat, (b * keys + tokens) * dim * 4, ctxBytes);
  }

  // `[batch, n, heads, headDim]` to `[batch, heads, n, headDim]`, one member at
  // a time: `permute` swaps two axes and knows nothing about a third.
  const qh = gpu.scratch(at("qh"), q.length);
  const kh = gpu.scratch(at("kh"), batch * keys * dim);
  const vh = gpu.scratch(at("vh"), batch * keys * dim);
  for (let b = 0; b < batch; b += 1) {
    gpu.copy(headMajor(dit, q, b, tokens, at("qp")), 0, qh, b * tokens * dim * 4, tokens * dim * 4);
    gpu.copy(headMajor(dit, kcat, b, keys, at("kp")), 0, kh, b * keys * dim * 4, keys * dim * 4);
    gpu.copy(headMajor(dit, vcat, b, keys, at("vp")), 0, vh, b * keys * dim * 4, keys * dim * 4);
  }

  const attended = gpu.attention({
    q: qh,
    k: kh,
    v: vh,
    mask: dit.keyBias,
    maskShape: [batch, 1, 1],
    B: batch,
    H: heads,
    L: tokens,
    S: keys,
    D: headDim,
    scale: 1 / Math.sqrt(headDim),
    slot: at("attn.out"),
  });

  // Back to token-major, per member.
  const y = gpu.scratch(at("y"), rows * dim);
  for (let b = 0; b < batch; b += 1) {
    const stride = tokens * dim * 4;
    gpu.copy(tokenMajor(dit, attended, b, tokens, at("yp")), 0, y, b * stride, stride);
  }

  // `y * sigmoid(gate(h))`, with the halves folded into `gate` and `wo`.
  const gated = gpu.elementwise(
    y,
    gpu.elementwise(
      gpu.activation(gpu.matmul(h, block.gate, rows, dim, dim, at("g")), ACTIVATION.tanh, at("gt")),
      dit.ones,
      ADD,
      at("g1"),
    ),
    MULTIPLY,
    at("gy"),
  );
  const out = gatedAdd(dit, x, gpu.matmul(gated, block.wo, rows, dim, dim, at("wo")), attentionMod.gate, at("res1"));

  const mlpIn = modulate(dit, out, mlpMod, at("mlp"));
  const up = gpu.elementwise(
    gpu.activation(gpu.matmul(mlpIn, block.w1, rows, mlpHidden, dim, at("w1")), ACTIVATION.silu, at("w1a")),
    gpu.matmul(mlpIn, block.w3, rows, mlpHidden, dim, at("w3")),
    MULTIPLY,
    at("swiglu"),
  );
  return gatedAdd(dit, out, gpu.matmul(up, block.w2, rows, dim, mlpHidden, at("w2")), mlpMod.gate, at("res2"));
}

/**
 * RoPE over the first half of the heads, one batch member at a time.
 *
 * `rope`'s position is the token index it derives from the flat offset, so a
 * single call over `batch * tokens` would give the second member positions
 * starting at `tokens` — and its dispatch is sized for one member, so the rest
 * of the buffer is never written at all. Both are silent.
 */
function rotateHalf(dit: GpuDit, input: Tensor, slot: string): Tensor {
  const { gpu, shape, batch, tokens } = dit;
  const { dim, heads } = shape;
  const headDim = dim / heads;
  if (batch === 1) {
    return gpu.rope(input, tokens, heads, headDim, 10000, slot, 0, Math.floor(heads / 2));
  }
  const out = gpu.scratch(slot, input.length);
  const stride = tokens * dim * 4;
  for (let b = 0; b < batch; b += 1) {
    const slice = gpu.scratch(`${slot}.slice`, tokens * dim);
    gpu.copy(input, b * stride, slice, 0, stride);
    const rotated = gpu.rope(slice, tokens, heads, headDim, 10000, `${slot}.one`, 0, Math.floor(heads / 2));
    gpu.copy(rotated, 0, out, b * stride, stride);
  }
  return out;
}

/** One batch member of `[batch, n, heads, headDim]` to `[heads, n, headDim]`. */
function headMajor(dit: GpuDit, source: Tensor, member: number, n: number, slot: string): Tensor {
  const { gpu, shape } = dit;
  const { dim, heads } = shape;
  const slice = gpu.scratch(`${slot}.slice.${n}`, n * dim);
  gpu.copy(source, member * n * dim * 4, slice, 0, n * dim * 4);
  return gpu.permute(slice, n, heads, dim / heads, `${slot}.${n}`);
}

/** The inverse, for one member of `[batch, heads, n, headDim]`. */
function tokenMajor(dit: GpuDit, source: Tensor, member: number, n: number, slot: string): Tensor {
  const { gpu, shape } = dit;
  const { dim, heads } = shape;
  const slice = gpu.scratch(`${slot}.slice.${n}`, n * dim);
  gpu.copy(source, member * n * dim * 4, slice, 0, n * dim * 4);
  return gpu.permute(slice, heads, n, dim / heads, `${slot}.${n}`);
}

/** `in_proj`, twelve blocks, `out_norm`, `out_proj`. */
export function velocityGpu(
  dit: GpuDit,
  x: Tensor,
  modulation: { attention: Modulation; mlp: Modulation }[],
  trace?: (stage: string, tensor: Tensor) => void,
): Tensor {
  const { gpu, shape, batch, tokens, latentDim } = dit;
  const rows = batch * tokens;
  let h = gpu.elementwise(
    gpu.matmul(x, dit.inProj, rows, shape.dim, latentDim, "in.proj"),
    dit.inProjBiasRows,
    ADD,
    "in.bias",
  );
  trace?.("in_proj", h);
  for (let index = 0; index < dit.blocks.length; index += 1) {
    const mod = modulation[index]!;
    h = ditBlockGpu(dit, h, mod.attention, mod.mlp, index);
    trace?.(`blocks.${index}`, h);
    // The block writes into pooled slots, so its result has to be copied out
    // before the next block reuses them.
    const kept = gpu.scratch(`layer.${index % 2}`, h.length);
    gpu.copy(h, 0, kept, 0, h.length * 4);
    h = kept;
  }
  const normed = gpu.rmsnorm(h, dit.outNorm, rows, shape.dim, shape.eps, "out.norm");
  trace?.("out_norm", normed);
  return gpu.elementwise(
    gpu.matmul(normed, dit.outProj, rows, latentDim, shape.dim, "out.proj"),
    dit.outProjBiasRows,
    ADD,
    "out.bias",
  );
}
