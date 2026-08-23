import { ACTIVATION, activation } from "web-xpu-ops/ops/activation";
import { attention } from "web-xpu-ops/ops/attention";
import { rmsnorm } from "web-xpu-ops/ops/rmsnorm";
import { rope } from "web-xpu-ops/ops/rope";

import {
  ROPE_THETA,
  addInto,
  fromHeads,
  headNorm,
  linear,
  sigmoidInto,
  swiglu,
  toHeads,
} from "./blocks.js";

/**
 * `DiffusionBlock` — the twelve layers `sample_rf` runs 32 times each, and half
 * the reference's wall clock.
 *
 * Two things here have no counterpart anywhere else in the model.
 *
 * **Joint attention.** Queries come from the latent, but keys and values are
 * the latent's own *concatenated with* the text state, the speaker state and
 * the caption state — one attention over 989 keys rather than a self-attention
 * followed by three cross-attentions. Each context is projected by its own
 * `wk_*` / `wv_*` into the model dimension, and the concatenated mask is what
 * lets an absent condition (a caption nobody supplied) be present in the tensor
 * and blocked in the arithmetic.
 *
 * The *order* of the concatenation does not matter — attention sums over the
 * key axis, so permuting keys, values and mask together is exactly invariant.
 * Reordering them is the one change to this file that a golden comparison
 * cannot see, and correctly so. It follows the reference anyway, because
 * matching the source is cheaper to read than an argument about why it need
 * not.
 *
 * **RoPE covers half the heads.** `_apply_rotary_half` chunks along the *head*
 * axis and rotates only the first half — heads 0 to 9 of 20 — leaving the rest
 * untouched, and the context keys are never rotated at all. Rotating all of
 * them is the obvious reading of the name and produces a model that runs and
 * mispronounces, so the split is taken from the code rather than from the
 * word "half".
 *
 * **Batch is not always one.** The reference runs classifier-free guidance as a
 * batch of three — conditional plus two unconditional variants — through one
 * forward pass, so `x` arrives as `[3, tokens, dim]` and the concatenated
 * context as `[3, keys, dim]`. Treating that as one long sequence lets the
 * three variants attend to each other, which produces a number rather than an
 * error and was the first thing this port got wrong.
 */

export interface AdaLnWeights {
  shiftDown: Float32Array;
  scaleDown: Float32Array;
  gateDown: Float32Array;
  shiftUp: Float32Array;
  scaleUp: Float32Array;
  gateUp: Float32Array;
  shiftBias: Float32Array;
  scaleBias: Float32Array;
  gateBias: Float32Array;
}

export interface DitBlockWeights {
  wq: Float32Array;
  wk: Float32Array;
  wv: Float32Array;
  gate: Float32Array;
  wo: Float32Array;
  qNorm: Float32Array;
  kNorm: Float32Array;
  contexts: Record<string, { wk: Float32Array; wv: Float32Array; dim: number }>;
  w1: Float32Array;
  w2: Float32Array;
  w3: Float32Array;
  attentionAdaLn: AdaLnWeights;
  mlpAdaLn: AdaLnWeights;
}

export interface DitShape {
  dim: number;
  heads: number;
  mlpHidden: number;
  rank: number;
  eps: number;
}

export interface Context {
  /** `[batch, tokens, contextDim]`. */
  state: Float32Array;
  /** `batch * tokens` flags, in the same order. */
  keep: boolean[];
}

/** RMSNorm with no learnable scale — `x * rsqrt(mean(x^2) + eps)`. */
const ONES = new Map<number, Float32Array>();
function plainRms(x: Float32Array, rows: number, D: number, eps: number): Float32Array {
  let ones = ONES.get(D);
  if (!ones) {
    ones = new Float32Array(D).fill(1);
    ONES.set(D, ones);
  }
  return rmsnorm({ input: x, weight: ones, N: rows, D, eps });
}

/**
 * `LowRankAdaLN`: modulate `x`, and return the residual gate alongside.
 *
 * `cond_embed` arrives as one `3 * dim` vector and splits into shift, scale and
 * gate. Each is refined by its own low-rank pass — `up(down(silu(c))) + c`, a
 * residual, so the rank-192 branch corrects a full-width signal rather than
 * replacing it. The gate comes back through `tanh`, which is why a block can
 * contribute nothing at all early in the flow.
 */
function adaLn(
  x: Float32Array,
  cond: Float32Array,
  w: AdaLnWeights,
  tokens: number,
  batch: number,
  shape: DitShape,
): { h: Float32Array; gate: Float32Array } {
  const { dim, rank, eps } = shape;
  // One (shift, scale, gate) per batch member: `cond_embed` is `[batch, 1,
  // 3 * dim]` and the three parts are `chunk(3, dim=-1)` of it.
  const refine = (part: number, down: Float32Array, up: Float32Array, bias: Float32Array) => {
    const source = new Float32Array(batch * dim);
    for (let b = 0; b < batch; b += 1) {
      source.set(cond.subarray(b * 3 * dim + part * dim, b * 3 * dim + (part + 1) * dim), b * dim);
    }
    const activated = activation({ input: source.slice(), kind: ACTIVATION.silu });
    const out = linear(linear(activated, down, batch, dim, rank), up, batch, rank, dim, bias);
    for (let i = 0; i < out.length; i += 1) out[i]! += source[i]!;
    return out;
  };

  const shift = refine(0, w.shiftDown, w.shiftUp, w.shiftBias);
  const scale = refine(1, w.scaleDown, w.scaleUp, w.scaleBias);
  const gate = refine(2, w.gateDown, w.gateUp, w.gateBias);
  for (let i = 0; i < gate.length; i += 1) gate[i] = Math.tanh(gate[i]!);

  // Within a batch member, one vector modulates every token.
  const h = plainRms(x, batch * tokens, dim, eps);
  for (let b = 0; b < batch; b += 1) {
    for (let token = 0; token < tokens; token += 1) {
      const at = (b * tokens + token) * dim;
      for (let d = 0; d < dim; d += 1) {
        h[at + d] = h[at + d]! * (1 + scale[b * dim + d]!) + shift[b * dim + d]!;
      }
    }
  }
  return { h, gate };
}

/**
 * Rotate the first half of the heads and leave the rest.
 *
 * `rope`'s `headOffset` / `headCount` express exactly this, so the pass-through
 * half is copied by the op rather than stitched back here.
 */
function rotateHalf(
  x: Float32Array,
  tokens: number,
  heads: number,
  headDim: number,
  batch: number,
): Float32Array {
  // Per batch member, because `rope`'s position is its token index: one call
  // over `batch * tokens` would give the second member positions starting at
  // `tokens` instead of at zero.
  const out = new Float32Array(x.length);
  const stride = tokens * heads * headDim;
  for (let b = 0; b < batch; b += 1) {
    out.set(
      rope({
        input: x.subarray(b * stride, (b + 1) * stride),
        N: tokens,
        numHeads: heads,
        headDim,
        posOffset: 0,
        thetaBase: ROPE_THETA,
        headOffset: 0,
        headCount: Math.floor(heads / 2),
      }),
      b * stride,
    );
  }
  return out;
}

/**
 * The context keys and values, projected once and reused by every step.
 *
 * `project_context_kv` exists in the reference for the same reason: the text,
 * speaker and caption states do not change across the 32 flow steps or the 12
 * blocks' inputs, so projecting them per step would be 32x the work for the
 * same numbers. Each block still has its own projections, so this is per block.
 */
export interface ContextKv {
  /** `[batch, tokens, dim]`. */
  k: Float32Array;
  v: Float32Array;
  /** `batch * tokens` flags. */
  keep: boolean[];
  /** Per batch member. */
  tokens: number;
  batch: number;
}

export function projectContexts(
  contexts: Record<string, Context>,
  w: DitBlockWeights,
  shape: DitShape,
  batch: number,
): ContextKv {
  const { dim, heads, eps } = shape;
  const headDim = dim / heads;
  const parts: { k: Float32Array; v: Float32Array; keep: boolean[]; tokens: number }[] = [];
  let tokens = 0;
  // Insertion order is the reference's concat order: text, speaker, caption.
  for (const name of Object.keys(w.contexts)) {
    const context = contexts[name];
    if (!context) throw new Error(`block wants a ${name} context and none was given`);
    const { wk, wv, dim: contextDim } = w.contexts[name]!;
    const rows = context.keep.length;
    if (rows % batch !== 0) throw new Error(`${name} context has ${rows} rows, not a multiple of batch ${batch}`);
    parts.push({
      // Context keys are normalised like the latent's own, and — unlike them —
      // never rotated.
      k: headNorm(linear(context.state, wk, rows, contextDim, dim), w.kNorm, rows, heads, headDim, eps),
      v: linear(context.state, wv, rows, contextDim, dim),
      keep: context.keep,
      tokens: rows / batch,
    });
    tokens += rows / batch;
  }

  // The concat is along the token axis *within* each batch member, so the
  // pieces interleave rather than append.
  const k = new Float32Array(batch * tokens * dim);
  const v = new Float32Array(batch * tokens * dim);
  const keep = new Array<boolean>(batch * tokens);
  for (let b = 0; b < batch; b += 1) {
    let at = b * tokens;
    for (const part of parts) {
      k.set(part.k.subarray(b * part.tokens * dim, (b + 1) * part.tokens * dim), at * dim);
      v.set(part.v.subarray(b * part.tokens * dim, (b + 1) * part.tokens * dim), at * dim);
      for (let j = 0; j < part.tokens; j += 1) keep[at + j] = part.keep[b * part.tokens + j]!;
      at += part.tokens;
    }
  }
  return { k, v, keep, tokens, batch };
}

export function ditBlock(
  x: Float32Array,
  cond: Float32Array,
  w: DitBlockWeights,
  context: ContextKv,
  tokens: number,
  shape: DitShape,
): Float32Array {
  const { dim, heads, mlpHidden, eps } = shape;
  const headDim = dim / heads;
  const batch = context.batch;
  const rows = batch * tokens;
  const out = x.slice();

  const { h, gate: attentionGate } = adaLn(out, cond, w.attentionAdaLn, tokens, batch, shape);

  const project = (weight: Float32Array) => linear(h, weight, rows, dim, dim);
  const q = rotateHalf(headNorm(project(w.wq), w.qNorm, rows, heads, headDim, eps), tokens, heads, headDim, batch);
  const kSelf = rotateHalf(headNorm(project(w.wk), w.kNorm, rows, heads, headDim, eps), tokens, heads, headDim, batch);
  const vSelf = project(w.wv);

  // Self keys first, then the contexts — `torch.cat([k_self, k_text, ...])`,
  // per batch member.
  const keys = tokens + context.tokens;
  const k = new Float32Array(batch * keys * dim);
  const v = new Float32Array(batch * keys * dim);
  const keyBias = new Float32Array(batch * keys);
  for (let b = 0; b < batch; b += 1) {
    k.set(kSelf.subarray(b * tokens * dim, (b + 1) * tokens * dim), b * keys * dim);
    v.set(vSelf.subarray(b * tokens * dim, (b + 1) * tokens * dim), b * keys * dim);
    k.set(context.k.subarray(b * context.tokens * dim, (b + 1) * context.tokens * dim), (b * keys + tokens) * dim);
    v.set(context.v.subarray(b * context.tokens * dim, (b + 1) * context.tokens * dim), (b * keys + tokens) * dim);
    // The latent's own positions are never masked; `self_mask` defaults to all
    // ones and nothing in the sampling path overrides it.
    for (let j = 0; j < context.tokens; j += 1) {
      keyBias[b * keys + tokens + j] = context.keep[b * context.tokens + j] ? 0 : -Infinity;
    }
  }

  const { output } = attention({
    q: toHeads(q, tokens, heads, headDim, batch),
    k: toHeads(k, keys, heads, headDim, batch),
    v: toHeads(v, keys, heads, headDim, batch),
    B: batch,
    H: heads,
    L: tokens,
    S: keys,
    D: headDim,
    Dv: headDim,
    mask: keyBias,
    maskShape: [batch, 1, 1],
  });

  const y = fromHeads(output, tokens, heads, headDim, batch);
  // The gate is a projection of the AdaLN output `h`, not of the block input.
  sigmoidInto(y, linear(h, w.gate, rows, dim, dim));
  const attended = linear(y, w.wo, rows, dim, dim);
  const gateInto = (delta: Float32Array, gate: Float32Array) => {
    for (let b = 0; b < batch; b += 1) {
      for (let token = 0; token < tokens; token += 1) {
        const at = (b * tokens + token) * dim;
        for (let d = 0; d < dim; d += 1) out[at + d]! += gate[b * dim + d]! * delta[at + d]!;
      }
    }
  };
  gateInto(attended, attentionGate);

  const { h: mlpIn, gate: mlpGate } = adaLn(out, cond, w.mlpAdaLn, tokens, batch, shape);
  gateInto(swiglu(mlpIn, w.w1, w.w2, w.w3, rows, dim, mlpHidden), mlpGate);
  return out;
}

export { addInto };
