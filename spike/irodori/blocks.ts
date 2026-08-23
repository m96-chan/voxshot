import { ACTIVATION, activation } from "web-xpu-ops/ops/activation";
import { attention } from "web-xpu-ops/ops/attention";
import { matmul } from "web-xpu-ops/ops/matmul";
import { rmsnorm } from "web-xpu-ops/ops/rmsnorm";
import { rope } from "web-xpu-ops/ops/rope";

/**
 * The pieces Irodori builds its own blocks out of, shared by the speaker
 * encoder and the DiT.
 *
 * None of these are ModernBERT's. The two halves of this model do not agree on
 * a single convention:
 *
 * | | ModernBERT-ja | Irodori's own blocks |
 * | --- | --- | --- |
 * | norm | LayerNorm, no bias | RMSNorm |
 * | MLP | GeGLU, exact `gelu` | SwiGLU, `silu` |
 * | RoPE lanes | split halves | **adjacent pairs** |
 * | RoPE base | 160000 / 10000 by layer | 10000 |
 * | Q/K norm | none | per-head RMSNorm before RoPE |
 *
 * The RoPE row is the one that costs something: ModernBERT needed its Q and K
 * weights permuted to use web-xpu-ops' adjacent-lane `rope`, and these blocks
 * do not, because `apply_rotary_emb` already reshapes to `(-1, 2)` and
 * multiplies as complex. Same op, opposite conclusion, and only reading both
 * says which.
 */

export const ROPE_THETA = 10000;

/** `y = x @ W.T (+ b)`, with `W` already stored as `[in, out]`. */
export function linear(
  x: Float32Array,
  weight: Float32Array,
  rows: number,
  inDim: number,
  outDim: number,
  bias?: Float32Array,
): Float32Array {
  const out = matmul({ a: x, b: weight, M: rows, N: outDim, K: inDim });
  if (bias) {
    for (let row = 0; row < rows; row += 1) {
      for (let d = 0; d < outDim; d += 1) out[row * outDim + d]! += bias[d]!;
    }
  }
  return out;
}

/** `[out, in]` to `[in, out]`, done once at load. */
export function transpose(source: Float32Array, out: number, inn: number): Float32Array {
  const result = new Float32Array(source.length);
  for (let o = 0; o < out; o += 1) {
    const from = o * inn;
    for (let i = 0; i < inn; i += 1) result[i * out + o] = source[from + i]!;
  }
  return result;
}

export function norm(x: Float32Array, weight: Float32Array, rows: number, D: number, eps: number) {
  return rmsnorm({ input: x, weight, N: rows, D, eps });
}

/**
 * Per-head RMSNorm over the head dimension, `weight` shaped `[heads, headDim]`.
 *
 * `rmsnorm`'s `groups` selects the weight row as `row % groups`, which is
 * exactly right when the input is laid out token-major — row `token * heads +
 * head` picks weight row `head`. No reshaping needed on either side.
 */
export function headNorm(
  x: Float32Array,
  weight: Float32Array,
  tokens: number,
  heads: number,
  headDim: number,
  eps: number,
): Float32Array {
  return rmsnorm({ input: x, weight, N: tokens * heads, D: headDim, eps, groups: heads });
}

/** `w2(silu(w1(x)) * w3(x))`. */
export function swiglu(
  x: Float32Array,
  w1: Float32Array,
  w2: Float32Array,
  w3: Float32Array,
  rows: number,
  dim: number,
  hidden: number,
): Float32Array {
  const gate = activation({ input: linear(x, w1, rows, dim, hidden), kind: ACTIVATION.silu });
  const up = linear(x, w3, rows, dim, hidden);
  for (let i = 0; i < gate.length; i += 1) gate[i]! *= up[i]!;
  return linear(gate, w2, rows, hidden, dim);
}

/** `[batch, tokens, heads * headDim]` to `[batch, heads, tokens, headDim]`. */
export function toHeads(
  source: Float32Array,
  tokens: number,
  heads: number,
  headDim: number,
  batch = 1,
) {
  const out = new Float32Array(source.length);
  const stride = tokens * heads * headDim;
  for (let b = 0; b < batch; b += 1) {
    for (let token = 0; token < tokens; token += 1) {
      for (let head = 0; head < heads; head += 1) {
        const from = b * stride + (token * heads + head) * headDim;
        const to = b * stride + (head * tokens + token) * headDim;
        for (let d = 0; d < headDim; d += 1) out[to + d] = source[from + d]!;
      }
    }
  }
  return out;
}

/** The inverse of {@link toHeads}. */
export function fromHeads(
  source: Float32Array,
  tokens: number,
  heads: number,
  headDim: number,
  batch = 1,
) {
  const out = new Float32Array(source.length);
  const stride = tokens * heads * headDim;
  for (let b = 0; b < batch; b += 1) {
    for (let head = 0; head < heads; head += 1) {
      for (let token = 0; token < tokens; token += 1) {
        const from = b * stride + (head * tokens + token) * headDim;
        const to = b * stride + (token * heads + head) * headDim;
        for (let d = 0; d < headDim; d += 1) out[to + d] = source[from + d]!;
      }
    }
  }
  return out;
}

/**
 * Additive bias that blocks the columns `keep` says are padding.
 *
 * `[1, 1, 1] x S` — one row, broadcast over every query. `SelfAttention` passes
 * `key_mask[:, None, None, :]`, which is the same shape and the same meaning.
 */
export function keyBias(keep: boolean[]): Float32Array {
  const bias = new Float32Array(keep.length);
  for (let j = 0; j < keep.length; j += 1) bias[j] = keep[j] ? 0 : -Infinity;
  return bias;
}

export function addInto(target: Float32Array, delta: Float32Array): void {
  for (let i = 0; i < target.length; i += 1) target[i]! += delta[i]!;
}

export function sigmoidInto(target: Float32Array, gate: Float32Array): void {
  for (let i = 0; i < target.length; i += 1) target[i]! *= 1 / (1 + Math.exp(-gate[i]!));
}

export interface SelfAttentionWeights {
  wq: Float32Array;
  wk: Float32Array;
  wv: Float32Array;
  wo: Float32Array;
  gate: Float32Array;
  qNorm: Float32Array;
  kNorm: Float32Array;
}

/**
 * Irodori's `SelfAttention`: gated, with Q/K normalised per head before RoPE.
 *
 * The gate is the part that has no counterpart in ModernBERT — a fifth
 * projection of the *input*, applied as `y * sigmoid(gate)` after attention and
 * before `wo`. Dropping it leaves a working attention block that attends
 * correctly and scales everything wrongly.
 */
export function selfAttention(
  x: Float32Array,
  w: SelfAttentionWeights,
  keep: boolean[],
  dim: number,
  heads: number,
  eps: number,
): Float32Array {
  const tokens = keep.length;
  const headDim = dim / heads;
  const project = (weight: Float32Array) => linear(x, weight, tokens, dim, dim);

  const q = headNorm(project(w.wq), w.qNorm, tokens, heads, headDim, eps);
  const k = headNorm(project(w.wk), w.kNorm, tokens, heads, headDim, eps);
  const v = project(w.wv);
  const gate = project(w.gate);

  const rotate = (input: Float32Array) =>
    rope({ input, N: tokens, numHeads: heads, headDim, posOffset: 0, thetaBase: ROPE_THETA });

  const { output } = attention({
    q: toHeads(rotate(q), tokens, heads, headDim),
    k: toHeads(rotate(k), tokens, heads, headDim),
    v: toHeads(v, tokens, heads, headDim),
    B: 1,
    H: heads,
    L: tokens,
    S: tokens,
    D: headDim,
    Dv: headDim,
    mask: keyBias(keep),
    maskShape: [1, 1, 1],
  });

  const y = fromHeads(output, tokens, heads, headDim);
  sigmoidInto(y, gate);
  return linear(y, w.wo, tokens, dim, dim);
}

export interface TextBlockWeights {
  attentionNorm: Float32Array;
  attention: SelfAttentionWeights;
  mlpNorm: Float32Array;
  w1: Float32Array;
  w2: Float32Array;
  w3: Float32Array;
}

/** `x + attn(norm(x))`, then `x + mlp(norm(x))`. */
export function textBlock(
  x: Float32Array,
  w: TextBlockWeights,
  keep: boolean[],
  dim: number,
  heads: number,
  hidden: number,
  eps: number,
): Float32Array {
  const tokens = keep.length;
  const out = x.slice();
  addInto(out, selfAttention(norm(out, w.attentionNorm, tokens, dim, eps), w.attention, keep, dim, heads, eps));
  addInto(out, swiglu(norm(out, w.mlpNorm, tokens, dim, eps), w.w1, w.w2, w.w3, tokens, dim, hidden));
  return out;
}
