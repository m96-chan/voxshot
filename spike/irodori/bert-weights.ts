import { readFileSync } from "node:fs";
import { join } from "node:path";

/**
 * ModernBERT-ja's weights, as `dump_bert.py` wrote them, arranged for the ops.
 *
 * Two rearrangements happen here rather than in the hot loop, because both are
 * per-weight and neither depends on the input.
 *
 * **Linear weights are transposed.** torch stores `nn.Linear.weight` as
 * `[out, in]` and computes `x @ W.T`; `matmul` wants `b` as `[K, N]`. Doing it
 * once at load costs one pass over each weight instead of one per token.
 *
 * **Q and K rows are interleaved**, which needs more than a sentence — below.
 */

/** RoPE pairs, and why the weights are permuted to suit them. */
export const ROPE_NOTE = `
web-xpu-ops' \`rope\` rotates **adjacent** lanes: \`(x[2i], x[2i+1])\` by angle
\`i\`. HF's ModernBERT rotates **split halves** via \`rotate_half\`:
\`(x[i], x[i+d/2])\`, same angle \`i\`. Different layouts, same rotation.

Rather than write a second RoPE, the weights are permuted so the layouts agree:
row \`2i\` of Wq/Wk takes old row \`i\`, row \`2i+1\` takes old row \`i+d/2\`.
Then adjacent-lane RoPE computes exactly what split-half RoPE would, permuted
the same way — and **attention does not notice**, because \`q·k\` is invariant
under a permutation applied to both. V is untouched, so nothing downstream sees
the permuted axis at all.

This is only valid because the frequency per pair matches: with no scaling
web-xpu-ops' \`invFreq\` is \`base^(-2*pair/headDim)\`, which is HF's
\`inv_freq\`. Checked in the source, not assumed.
`.trim();

export interface BertConfig {
  hiddenSize: number;
  numLayers: number;
  numHeads: number;
  intermediateSize: number;
  normEps: number;
  layerTypes: string[];
  slidingWindow: number;
  ropeTheta: Record<string, number>;
  vocabSize: number;
}

export interface BertLayer {
  /** Absent on layer 0, where `ModernBertEncoderLayer` uses `nn.Identity`. */
  attnNorm: Float32Array | undefined;
  /** `[hidden, 3 * hidden]`, Q and K rows interleaved for RoPE. */
  wqkv: Float32Array;
  /** `[hidden, hidden]`. */
  wo: Float32Array;
  mlpNorm: Float32Array;
  /** `[hidden, 2 * intermediate]` — the GeGLU input and gate, in that order. */
  mlpWi: Float32Array;
  /** `[intermediate, hidden]`. */
  mlpWo: Float32Array;
  ropeTheta: number;
  sliding: boolean;
}

export interface BertWeights {
  config: BertConfig;
  tokenEmbeddings: Float32Array;
  embeddingNorm: Float32Array;
  finalNorm: Float32Array;
  layers: BertLayer[];
  /** How many float32 the weights occupy. */
  floats: number;
}

interface Entry {
  shape: number[];
  bytes: number;
}

interface BertIndex {
  config: {
    hidden_size: number;
    num_hidden_layers: number;
    num_attention_heads: number;
    intermediate_size: number;
    norm_eps: number;
    layer_types: string[];
    sliding_window: number;
    rope_theta: Record<string, number>;
    vocab_size: number;
  };
  weights: Record<string, Entry>;
  activations: Record<string, Entry>;
}

function readF32(dir: string, name: string, entry: Entry): Float32Array {
  const bytes = readFileSync(join(dir, `${name}.f32`));
  if (bytes.byteLength !== entry.bytes) {
    throw new Error(`${name}.f32 is ${bytes.byteLength} bytes, index says ${entry.bytes}`);
  }
  // `Buffer` is a view into a pooled `ArrayBuffer` at an arbitrary offset, and
  // `Float32Array` needs 4-byte alignment, so the range is copied out.
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return new Float32Array(copy.buffer);
}

/** `[out, in]` to `[in, out]`. */
function transpose(source: Float32Array, out: number, inn: number): Float32Array {
  const result = new Float32Array(source.length);
  for (let o = 0; o < out; o += 1) {
    const from = o * inn;
    for (let i = 0; i < inn; i += 1) result[i * out + o] = source[from + i]!;
  }
  return result;
}

/**
 * Interleave the Q and K rows of a `[3 * hidden, hidden]` Wqkv. See
 * {@link ROPE_NOTE}.
 *
 * The output features are ordered `[3][heads][headDim]` — that is the layout
 * `qkv.view(*shape, 3, -1, head_dim).unbind(-3)` reads — so the permutation
 * applies within each 64-row head block of the Q and K thirds, and V is copied
 * through untouched.
 */
function interleaveQK(
  source: Float32Array,
  hidden: number,
  heads: number,
  headDim: number,
): Float32Array {
  const result = new Float32Array(source.length);
  const half = headDim / 2;
  for (let which = 0; which < 3; which += 1) {
    for (let head = 0; head < heads; head += 1) {
      const block = ((which * heads + head) * headDim) * hidden;
      if (which === 2) {
        result.set(source.subarray(block, block + headDim * hidden), block);
        continue;
      }
      for (let i = 0; i < half; i += 1) {
        result.set(source.subarray(block + i * hidden, block + (i + 1) * hidden), block + 2 * i * hidden);
        const upper = block + (i + half) * hidden;
        result.set(source.subarray(upper, upper + hidden), block + (2 * i + 1) * hidden);
      }
    }
  }
  return result;
}

export function loadBertWeights(dir: string): { weights: BertWeights; index: BertIndex } {
  const index = JSON.parse(readFileSync(join(dir, "index.json"), "utf8")) as BertIndex;
  const raw = index.config;
  const config: BertConfig = {
    hiddenSize: raw.hidden_size,
    numLayers: raw.num_hidden_layers,
    numHeads: raw.num_attention_heads,
    intermediateSize: raw.intermediate_size,
    normEps: raw.norm_eps,
    layerTypes: raw.layer_types,
    slidingWindow: raw.sliding_window,
    ropeTheta: raw.rope_theta,
    vocabSize: raw.vocab_size,
  };

  const get = (name: string): Float32Array => {
    const entry = index.weights[name];
    if (!entry) throw new Error(`${name} is not in golden/bert/index.json`);
    return readF32(dir, name, entry);
  };

  const { hiddenSize: H, numHeads: heads, intermediateSize: I } = config;
  const headDim = H / heads;
  const layers: BertLayer[] = [];
  for (let index_ = 0; index_ < config.numLayers; index_ += 1) {
    const at = `layers.${index_}`;
    const type = config.layerTypes[index_]!;
    layers.push({
      // Layer 0's `attn_norm` is `nn.Identity`, so the checkpoint has no tensor
      // for it. Reading one would be reading a weight the reference never had.
      attnNorm: index_ === 0 ? undefined : get(`${at}.attn_norm.weight`),
      wqkv: transpose(interleaveQK(get(`${at}.attn.Wqkv.weight`), H, heads, headDim), 3 * H, H),
      wo: transpose(get(`${at}.attn.Wo.weight`), H, H),
      mlpNorm: get(`${at}.mlp_norm.weight`),
      mlpWi: transpose(get(`${at}.mlp.Wi.weight`), 2 * I, H),
      mlpWo: transpose(get(`${at}.mlp.Wo.weight`), H, I),
      ropeTheta: config.ropeTheta[type] ?? 0,
      sliding: type === "sliding_attention",
    });
    if (!config.ropeTheta[type]) throw new Error(`no rope theta for layer type ${type}`);
  }

  const weights: BertWeights = {
    config,
    tokenEmbeddings: get("embeddings.tok_embeddings.weight"),
    embeddingNorm: get("embeddings.norm.weight"),
    finalNorm: get("final_norm.weight"),
    layers,
    floats: Object.values(index.weights).reduce((sum, entry) => sum + entry.bytes / 4, 0),
  };
  return { weights, index };
}

export type { BertIndex, Entry };
