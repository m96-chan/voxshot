import { activation, ACTIVATION } from "web-xpu-ops/ops/activation";
import { elementwise, ELEMENTWISE } from "web-xpu-ops/ops/elementwise";
import { gather } from "web-xpu-ops/ops/gather";
import { groupedAttention } from "web-xpu-ops/ops/gqa";
import { matmul } from "web-xpu-ops/ops/matmul";
import { matvec } from "web-xpu-ops/ops/matvec";
import { rmsnorm } from "web-xpu-ops/ops/rmsnorm";
import { KVCache } from "../../../web-xpu-ops/llm/kv-cache.js";
import { mergeHeadsMajor, splitHeadsMajor } from "../../../web-xpu-ops/llm/reshape.js";
import type { Tensor, Weights } from "./weights-cache.js";

/**
 * Qwen3-0.6B (MioTTS's LM), as an f32 CPU reference — the test oracle the GPU
 * port will be measured against, never part of the run path.
 *
 * Everything numeric goes through web-xpu-ops' reference ops; the only math
 * implemented here is HF's "rotate_half" RoPE, which `ops/rope` deliberately
 * does not speak (it rotates adjacent pairs; the GPU path will bridge the two
 * conventions by permuting weight channels instead — see
 * `llm/weights.ts permuteRopeChannels`).
 *
 * Forward structure (transformers 5.3 `modeling_qwen3.py`, read not guessed):
 * pre-norm; `h += attn(input_layernorm(h))`; `h += mlp(post_attention_layernorm(h))`;
 * attention projects q/k, applies **per-head RMSNorm** (one shared [128] gamma,
 * every head), then RoPE, then causal GQA at scale 1/sqrt(128); the MLP is
 * `down(silu(gate(x)) * up(x))`; final `model.norm`, then logits against the
 * tied embedding matrix.
 */

export interface Qwen3Config {
  numLayers: number;
  hidden: number;
  heads: number;
  kvHeads: number;
  headDim: number;
  ffn: number;
  vocab: number;
  ropeTheta: number;
  rmsEps: number;
  eosIds: number[];
}

/** The numbers `golden/index.json` records for Aratako/MioTTS-0.6B. */
export const MIOTTS_06B: Qwen3Config = {
  numLayers: 28,
  hidden: 1024,
  heads: 16,
  kvHeads: 8,
  headDim: 128,
  ffn: 3072,
  vocab: 164480,
  ropeTheta: 1_000_000,
  rmsEps: 1e-6,
  eosIds: [151645, 151643],
};

export interface LayerWeights {
  inputNorm: Float32Array;
  postAttnNorm: Float32Array;
  /** `[out, in]`, torch Linear layout, no biases anywhere in this model. */
  wq: Tensor;
  wk: Tensor;
  wv: Tensor;
  wo: Tensor;
  /** `[headDim]` — one gamma shared by every head (Qwen3's QK-norm). */
  qNorm: Float32Array;
  kNorm: Float32Array;
  wGate: Tensor;
  wUp: Tensor;
  wDown: Tensor;
}

export interface ModelWeights {
  /** `[vocab, hidden]` — the embedding table, and (tied) the LM head. */
  embed: Tensor;
  finalNorm: Float32Array;
  layers: LayerWeights[];
}

function expectShape(tensor: Tensor, shape: number[], name: string): Tensor {
  if (tensor.shape.length !== shape.length || tensor.shape.some((d, i) => d !== shape[i])) {
    throw new Error(`${name} is [${tensor.shape.join(", ")}], expected [${shape.join(", ")}]`);
  }
  return tensor;
}

/**
 * Every tensor the forward needs, pulled once and shape-checked against the
 * config. Shape checks here rather than downstream: a swapped checkpoint
 * should name the tensor, not surface as a wrong-length matmul.
 */
export function buildWeights(weights: Weights, cfg: Qwen3Config): ModelWeights {
  const { hidden, heads, kvHeads, headDim, ffn, vocab } = cfg;
  const layers: LayerWeights[] = [];
  for (let i = 0; i < cfg.numLayers; i += 1) {
    const p = `model.layers.${i}.`;
    layers.push({
      inputNorm: expectShape(weights.get(`${p}input_layernorm.weight`), [hidden], `${p}input_layernorm`).data,
      postAttnNorm: expectShape(
        weights.get(`${p}post_attention_layernorm.weight`),
        [hidden],
        `${p}post_attention_layernorm`,
      ).data,
      wq: expectShape(weights.get(`${p}self_attn.q_proj.weight`), [heads * headDim, hidden], `${p}q_proj`),
      wk: expectShape(weights.get(`${p}self_attn.k_proj.weight`), [kvHeads * headDim, hidden], `${p}k_proj`),
      wv: expectShape(weights.get(`${p}self_attn.v_proj.weight`), [kvHeads * headDim, hidden], `${p}v_proj`),
      wo: expectShape(weights.get(`${p}self_attn.o_proj.weight`), [hidden, heads * headDim], `${p}o_proj`),
      qNorm: expectShape(weights.get(`${p}self_attn.q_norm.weight`), [headDim], `${p}q_norm`).data,
      kNorm: expectShape(weights.get(`${p}self_attn.k_norm.weight`), [headDim], `${p}k_norm`).data,
      wGate: expectShape(weights.get(`${p}mlp.gate_proj.weight`), [ffn, hidden], `${p}gate_proj`),
      wUp: expectShape(weights.get(`${p}mlp.up_proj.weight`), [ffn, hidden], `${p}up_proj`),
      wDown: expectShape(weights.get(`${p}mlp.down_proj.weight`), [hidden, ffn], `${p}down_proj`),
    });
  }
  // Tied embeddings: there is no lm_head tensor in the checkpoint, asserted
  // rather than assumed so an untied future checkpoint fails here.
  if (weights.maybe("lm_head.weight")) {
    throw new Error("checkpoint has an untied lm_head.weight; this port assumes tied embeddings");
  }
  return {
    embed: expectShape(weights.get("model.embed_tokens.weight"), [vocab, hidden], "embed_tokens"),
    finalNorm: expectShape(weights.get("model.norm.weight"), [hidden], "model.norm").data,
    layers,
  };
}

/**
 * HF's "rotate_half" RoPE on a token-major `[tokens, heads, headDim]` tensor.
 *
 * The pairing is `(j, j + headDim/2)` — the two halves of a head — not the
 * adjacent `(2j, 2j+1)` pairing `ops/rope` implements. Both are valid RoPE
 * conventions; the checkpoint was trained under this one, so the reference
 * speaks it directly (the GPU path bridges by permuting weight channels
 * instead). Frequencies: `inv_freq[j] = theta^(-2j/headDim)`, angle
 * `position * inv_freq[j]`, one shared angle per pair.
 *
 * cos/sin are computed in f64, as JS arithmetic naturally is; torch computes
 * its table in f32. Measured (model.test.ts's noise floors): l0_q_roped's
 * worst rel equals l0_q_normed's 5.6e-7, i.e. the rotation adds nothing above
 * the norm's own floor — far under the 1e-4 bound there.
 */
export function ropeRotateHalf(
  x: Float32Array,
  tokens: number,
  heads: number,
  headDim: number,
  positionBase: number,
  theta: number,
): Float32Array {
  const half = headDim / 2;
  const out = new Float32Array(x.length);
  for (let t = 0; t < tokens; t += 1) {
    const position = positionBase + t;
    for (let j = 0; j < half; j += 1) {
      const angle = position * theta ** (-2 * j / headDim);
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      for (let h = 0; h < heads; h += 1) {
        const base = (t * heads + h) * headDim;
        const a = x[base + j]!;
        const b = x[base + half + j]!;
        out[base + j] = a * cos - b * sin;
        out[base + half + j] = b * cos + a * sin;
      }
    }
  }
  return out;
}

/**
 * `x @ W^T` for a torch-layout `[out, in]` weight, no bias (this model has
 * none).
 *
 * One row goes through `matvec`, which takes `[out, in]` directly; more rows
 * go through `matmul`, which wants `[in, out]`, so the weight is turned once
 * and cached by array identity (`Weights` keeps identities stable). Both ops
 * accumulate in f64 over ascending `k`, so the two paths agree bitwise —
 * which the decode-vs-prefill parity test leans on.
 */
const transposed = new WeakMap<Float32Array, Float32Array>();

function linear(x: Float32Array, rows: number, weight: Tensor): Float32Array {
  const [outFeatures, inFeatures] = weight.shape as [number, number];
  if (rows === 1) {
    return matvec({ matrix: weight.data, vector: x, M: outFeatures, K: inFeatures });
  }
  let b = transposed.get(weight.data);
  if (!b) {
    b = new Float32Array(inFeatures * outFeatures);
    for (let o = 0; o < outFeatures; o += 1) {
      for (let i = 0; i < inFeatures; i += 1) {
        b[i * outFeatures + o] = weight.data[o * inFeatures + i]!;
      }
    }
    transposed.set(weight.data, b);
  }
  return matmul({ a: x, b, M: rows, N: outFeatures, K: inFeatures });
}

function addInPlace(a: Float32Array, b: Float32Array): Float32Array {
  for (let i = 0; i < a.length; i += 1) a[i] = a[i]! + b[i]!;
  return a;
}

/**
 * Observability without thirty ad-hoc globals: the forward calls `trace` with
 * each stage's tensor under its **golden name** (`embedding`, `layer00`…,
 * `l0_q_proj`, `l0_q_normed`, `l0_q_roped`, `l0_attn_out`, `final_norm`,
 * `logits_last`). Tensors that the forward later mutates in place (`h`) are
 * copied before tracing; the rest are handed over as-is and never touched
 * again. No trace, no copies, no cost — greedy decoding passes nothing.
 */
export type TraceFn = (stage: string, data: Float32Array) => void;

/**
 * One generation's state: the KV cache and the position counter. `prefill`
 * once, then `decodeStep` per token; both return the last position's logits
 * over the full vocabulary.
 */
export class Qwen3Runner {
  readonly cache: KVCache;
  private position = 0;

  constructor(
    private readonly weights: ModelWeights,
    readonly cfg: Qwen3Config,
    maxSeqLen: number,
  ) {
    this.cache = new KVCache(cfg.numLayers, cfg.kvHeads, cfg.headDim, maxSeqLen);
  }

  get positions(): number {
    return this.position;
  }

  prefill(tokens: number[], trace?: TraceFn): Float32Array {
    if (this.position !== 0) throw new Error("prefill on a runner that has already run");
    return this.forward(tokens, trace);
  }

  decodeStep(token: number, trace?: TraceFn): Float32Array {
    if (this.position === 0) throw new Error("decodeStep before prefill");
    return this.forward([token], trace);
  }

  private forward(tokens: number[], trace?: TraceFn): Float32Array {
    const { cfg, weights: w } = this;
    const { hidden, heads, kvHeads, headDim, rmsEps, ropeTheta } = cfg;
    const T = tokens.length;
    const base = this.position;

    const indices = new Int32Array(tokens);
    // `rows` is the table's row count — indices outside it gather zeros, so
    // passing T here zeroed every embedding (caught by the embedding stage).
    const h = gather({ table: w.embed.data, indices, rows: cfg.vocab, D: hidden });
    trace?.("embedding", h.slice());

    for (let l = 0; l < cfg.numLayers; l += 1) {
      const L = w.layers[l]!;
      const first = l === 0;

      // ---- attention ----
      const normed = rmsnorm({ input: h, weight: L.inputNorm, N: T, D: hidden, eps: rmsEps });
      const q = linear(normed, T, L.wq);
      if (first) trace?.("l0_q_proj", q);
      const k = linear(normed, T, L.wk);
      const v = linear(normed, T, L.wv);

      // QK-norm: RMSNorm over each head's 128 channels, one shared gamma.
      // Token-major [T, heads, headDim] flattens to N = T*heads rows of D.
      const qn = rmsnorm({ input: q, weight: L.qNorm, N: T * heads, D: headDim, eps: rmsEps });
      if (first) trace?.("l0_q_normed", qn);
      const kn = rmsnorm({ input: k, weight: L.kNorm, N: T * kvHeads, D: headDim, eps: rmsEps });

      const qr = ropeRotateHalf(qn, T, heads, headDim, base, ropeTheta);
      if (first) trace?.("l0_q_roped", qr);
      const kr = ropeRotateHalf(kn, T, kvHeads, headDim, base, ropeTheta);

      this.cache.write(
        l,
        base,
        splitHeadsMajor(kr, T, kvHeads, headDim),
        splitHeadsMajor(v, T, kvHeads, headDim),
        T,
      );
      const S = base + T;
      const cached = this.cache.read(l, S);
      const att = groupedAttention({
        q: splitHeadsMajor(qr, T, heads, headDim),
        k: cached.k,
        v: cached.v,
        B: 1,
        H: heads,
        kvHeads,
        L: T,
        S,
        D: headDim,
        Dv: headDim,
        causal: true,
        queryOffset: base,
        scale: 1 / Math.sqrt(headDim),
      });
      const context = mergeHeadsMajor(att.output, heads, T, headDim);
      const attnOut = linear(context, T, L.wo);
      if (first) trace?.("l0_attn_out", attnOut);
      addInPlace(h, attnOut);

      // ---- mlp ----
      const normed2 = rmsnorm({ input: h, weight: L.postAttnNorm, N: T, D: hidden, eps: rmsEps });
      const gate = linear(normed2, T, L.wGate);
      const up = linear(normed2, T, L.wUp);
      const gated = elementwise({
        a: activation({ input: gate, kind: ACTIVATION.silu }),
        b: up,
        kind: ELEMENTWISE.multiply,
      });
      addInPlace(h, linear(gated, T, L.wDown));

      trace?.(`layer${String(l).padStart(2, "0")}`, h.slice());
    }

    this.position += T;

    const fin = rmsnorm({ input: h, weight: w.finalNorm, N: T, D: hidden, eps: rmsEps });
    trace?.("final_norm", fin);

    // Tied head: logits = final_norm's last row against the embedding table,
    // which matvec reads in its native [vocab, hidden] layout — no transpose.
    const lastRow = fin.subarray((T - 1) * hidden, T * hidden);
    const logits = matvec({ matrix: w.embed.data, vector: lastRow, M: cfg.vocab, K: hidden });
    trace?.("logits_last", logits);
    return logits;
  }
}

function argmax(logits: Float32Array): number {
  let best = 0;
  for (let i = 1; i < logits.length; i += 1) if (logits[i]! > logits[best]!) best = i;
  return best;
}

/**
 * Greedy continuation: one prefill, then the KV-cached decode path.
 *
 * `stopAtEos: true` is `generate()`'s semantics — the eos id is appended and
 * generation stops. `stopAtEos: false` is the golden's manual-loop semantics:
 * argmax for exactly `maxNew` steps, eos or not.
 */
export function greedyGenerate(
  promptIds: number[],
  maxNew: number,
  weights: ModelWeights,
  cfg: Qwen3Config,
  opts: { stopAtEos?: boolean } = {},
): number[] {
  const stopAtEos = opts.stopAtEos ?? true;
  const runner = new Qwen3Runner(weights, cfg, promptIds.length + maxNew);
  let logits = runner.prefill(promptIds);
  const out: number[] = [];
  for (let step = 0; step < maxNew; step += 1) {
    const id = argmax(logits);
    out.push(id);
    if (stopAtEos && cfg.eosIds.includes(id)) break;
    if (step + 1 < maxNew) logits = runner.decodeStep(id);
  }
  return out;
}
