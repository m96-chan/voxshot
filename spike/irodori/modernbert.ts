import { ACTIVATION, activation } from "web-xpu-ops/ops/activation";
import { attention } from "web-xpu-ops/ops/attention";
import { gather } from "web-xpu-ops/ops/gather";
import { layernorm } from "web-xpu-ops/ops/layernorm";
import { matmul } from "web-xpu-ops/ops/matmul";
import { rope } from "web-xpu-ops/ops/rope";

import type { BertLayer, BertWeights } from "./bert-weights.js";

/**
 * ModernBERT-ja, on web-xpu-ops.
 *
 * 25 encoder layers, batch 1, no KV cache — this is an encoder and every
 * position is computed once. Six ops carry it: `gather`, `layernorm`, `matmul`,
 * `rope`, `attention` and `activation`.
 *
 * Three things about this model are not the usual defaults, and each one is a
 * silent wrong answer if missed.
 *
 * **Two attention patterns alternate.** Layers 0, 3, 6, ... attend everywhere;
 * the other two thirds see only `|i - j| <= 64`. Which is which comes from
 * `config.layer_types`, read from the checkpoint rather than recomputed.
 *
 * **Each pattern has its own RoPE base** — 160000 for the global layers, 10000
 * for the sliding ones. One theta for all 25 gives a model that still produces
 * fluent-looking hidden states.
 *
 * **Layer 0 has no `attn_norm`.** `ModernBertEncoderLayer` substitutes
 * `nn.Identity` there, and the checkpoint has no tensor for it.
 *
 * The GeGLU halves are the other easy inversion: `Wi` produces `[input, gate]`
 * in that order and the activation goes on the *first* half.
 */

/** LayerNorm with `norm_bias: false`. The op wants a bias; this is it. */
const ZERO_BIAS = new Map<number, Float32Array>();
function zeros(size: number): Float32Array {
  let found = ZERO_BIAS.get(size);
  if (!found) {
    found = new Float32Array(size);
    ZERO_BIAS.set(size, found);
  }
  return found;
}

/**
 * Additive attention bias for one layer type, as `[1, 1, L] x S`.
 *
 * `keep[j]` false blocks column `j` for every row — that is key padding, and it
 * is what makes the truncation in {@link realLength} exact. A sliding layer
 * additionally blocks anything further than `window`, measured as `|i - j| <=
 * window` inclusive on both sides. That predicate was read off the mask
 * `transformers` actually builds rather than inferred from the name: with
 * `sliding_window = 3` and no padding, row 3 admits columns 0 through 6.
 *
 * Rows that end up with nothing allowed are left that way. torch's SDPA returns
 * exactly zeros for such a row, and so does web-xpu-ops' `attention` — both
 * measured, because the arithmetic says NaN and only the implementations say
 * otherwise.
 */
function attentionBias(keep: boolean[], window: number | null): Float32Array {
  const size = keep.length;
  const bias = new Float32Array(size * size);
  for (let i = 0; i < size; i += 1) {
    for (let j = 0; j < size; j += 1) {
      const blocked = !keep[j] || (window !== null && Math.abs(i - j) > window);
      bias[i * size + j] = blocked ? -Infinity : 0;
    }
  }
  return bias;
}

/** `[L, heads * headDim]` token-major to `[heads, L, headDim]`. */
function toHeads(source: Float32Array, L: number, heads: number, headDim: number): Float32Array {
  const out = new Float32Array(source.length);
  for (let token = 0; token < L; token += 1) {
    for (let head = 0; head < heads; head += 1) {
      const from = (token * heads + head) * headDim;
      const to = (head * L + token) * headDim;
      for (let d = 0; d < headDim; d += 1) out[to + d] = source[from + d]!;
    }
  }
  return out;
}

/** The inverse of {@link toHeads}. */
function fromHeads(source: Float32Array, L: number, heads: number, headDim: number): Float32Array {
  const out = new Float32Array(source.length);
  for (let head = 0; head < heads; head += 1) {
    for (let token = 0; token < L; token += 1) {
      const from = (head * L + token) * headDim;
      const to = (token * heads + head) * headDim;
      for (let d = 0; d < headDim; d += 1) out[to + d] = source[from + d]!;
    }
  }
  return out;
}

function addInto(target: Float32Array, delta: Float32Array): void {
  for (let i = 0; i < target.length; i += 1) target[i]! += delta[i]!;
}

export interface BertRun {
  /** `[L, hidden]` after `final_norm`, before any masking. */
  hidden: Float32Array;
  /** `[L, hidden]` for every layer index in `capture`. */
  captured: Map<string, Float32Array>;
  length: number;
}

export interface BertArgs {
  weights: BertWeights;
  /** One id per position, already padded the way Irodori pads. */
  inputIds: Int32Array;
  /** `attention_mask`, one flag per position. */
  keep: boolean[];
  /** Layer indices to record, plus `"embeddings"`. */
  capture?: Iterable<string>;
  /** Called after each layer, for progress on a 25-layer CPU run. */
  onLayer?: (index: number, total: number) => void;
}

/**
 * How many leading positions actually matter.
 *
 * Irodori pads every text to `max_text_len = 256`, so a seven-token sentence
 * runs 25 layers over 256 positions. It does not have to: key padding blocks
 * every column `j >= n` in **both** layer types, so rows `0..n-1` attend only to
 * rows `0..n-1`, at every layer. Truncating to `n` is exact, not an
 * approximation — and for this sentence it is 36x less arithmetic.
 *
 * The claim is checked rather than argued: `check-bert.ts` runs the full 256
 * and the truncated length and compares the rows they share.
 *
 * This holds because the padding is on the right. A left-padded input would put
 * real tokens at positions the RoPE angles differ for, and the two runs would
 * legitimately disagree.
 */
export function realLength(keep: boolean[]): number {
  let last = -1;
  for (let i = 0; i < keep.length; i += 1) if (keep[i]) last = i;
  return last + 1;
}

export function runModernBert({ weights, inputIds, keep, capture, onLayer }: BertArgs): BertRun {
  const { config } = weights;
  const L = inputIds.length;
  if (keep.length !== L) throw new Error(`${L} ids but ${keep.length} mask flags`);
  const H = config.hiddenSize;
  const heads = config.numHeads;
  const headDim = H / heads;
  const I = config.intermediateSize;
  const wanted = new Set(capture ?? []);
  const captured = new Map<string, Float32Array>();

  let hidden = layernorm({
    // `rows` is the *table's* row count, not the number of indices: `gather`
    // skips any index outside `[0, rows)` and leaves that output row zero,
    // silently. Passing the sequence length here produces an embedding of
    // nothing for every id above it, which still runs and still ends in
    // plausible-looking hidden states.
    input: gather({ table: weights.tokenEmbeddings, indices: inputIds, rows: config.vocabSize, D: H }),
    weight: weights.embeddingNorm,
    bias: zeros(H),
    N: L,
    D: H,
    eps: config.normEps,
  });
  if (wanted.has("embeddings")) captured.set("embeddings", hidden.slice());

  // One bias per pattern, not per layer: 25 layers share two masks.
  const biases = new Map<string, Float32Array>();
  const biasFor = (layer: BertLayer): Float32Array => {
    const key = layer.sliding ? "sliding" : "full";
    let found = biases.get(key);
    if (!found) {
      found = attentionBias(keep, layer.sliding ? config.slidingWindow : null);
      biases.set(key, found);
    }
    return found;
  };

  for (let index = 0; index < config.numLayers; index += 1) {
    const layer = weights.layers[index]!;

    const normed = layer.attnNorm
      ? layernorm({ input: hidden, weight: layer.attnNorm, bias: zeros(H), N: L, D: H, eps: config.normEps })
      : hidden;

    const qkv = matmul({ a: normed, b: layer.wqkv, M: L, N: 3 * H, K: H });
    // `qkv.view(*shape, 3, -1, head_dim)`: the three projections are contiguous
    // blocks of the feature axis, each already laid out head-major.
    const parts: Float32Array[] = [];
    for (let which = 0; which < 3; which += 1) {
      const part = new Float32Array(L * H);
      for (let token = 0; token < L; token += 1) {
        const from = token * 3 * H + which * H;
        for (let d = 0; d < H; d += 1) part[token * H + d] = qkv[from + d]!;
      }
      parts.push(part);
    }
    const [rawQ, rawK, rawV] = parts as [Float32Array, Float32Array, Float32Array];

    const rotate = (input: Float32Array): Float32Array =>
      rope({ input, N: L, numHeads: heads, headDim, posOffset: 0, thetaBase: layer.ropeTheta });

    const { output } = attention({
      q: toHeads(rotate(rawQ), L, heads, headDim),
      k: toHeads(rotate(rawK), L, heads, headDim),
      v: toHeads(rawV, L, heads, headDim),
      B: 1,
      H: heads,
      L,
      S: L,
      D: headDim,
      Dv: headDim,
      mask: biasFor(layer),
      maskShape: [1, 1, L],
    });

    // Safe to write into `hidden` even on layer 0, where `normed` aliases it:
    // `qkv` was read out above and `normed` is not touched again.
    addInto(hidden, matmul({ a: fromHeads(output, L, heads, headDim), b: layer.wo, M: L, N: H, K: H }));

    const mlpIn = layernorm({
      input: hidden,
      weight: layer.mlpNorm,
      bias: zeros(H),
      N: L,
      D: H,
      eps: config.normEps,
    });
    const wide = matmul({ a: mlpIn, b: layer.mlpWi, M: L, N: 2 * I, K: H });
    // `Wi(x).chunk(2, dim=-1)` names them `input, gate` and applies the
    // activation to `input` — the first half. Swapping them runs, and is wrong.
    const gated = new Float32Array(L * I);
    for (let token = 0; token < L; token += 1) {
      const row = token * 2 * I;
      for (let d = 0; d < I; d += 1) gated[token * I + d] = wide[row + d]!;
    }
    const activated = activation({ input: gated, kind: ACTIVATION.gelu });
    for (let token = 0; token < L; token += 1) {
      const row = token * 2 * I + I;
      for (let d = 0; d < I; d += 1) activated[token * I + d]! *= wide[row + d]!;
    }
    addInto(hidden, matmul({ a: activated, b: layer.mlpWo, M: L, N: H, K: I }));

    if (wanted.has(`layers.${index}`)) captured.set(`layers.${index}`, hidden.slice());
    onLayer?.(index, config.numLayers);
  }

  const final = layernorm({
    input: hidden,
    weight: weights.finalNorm,
    bias: zeros(H),
    N: L,
    D: H,
    eps: config.normEps,
  });
  return { hidden: final, captured, length: L };
}
