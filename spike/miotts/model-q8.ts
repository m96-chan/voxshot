import { activation, ACTIVATION } from "web-xpu-ops/ops/activation";
import { elementwise, ELEMENTWISE } from "web-xpu-ops/ops/elementwise";
import { groupedAttention } from "web-xpu-ops/ops/gqa";
import { matvecQ8 } from "web-xpu-ops/ops/matvec";
import { rmsnorm } from "web-xpu-ops/ops/rmsnorm";
import { rope } from "web-xpu-ops/ops/rope";
import { KVCache } from "../../../web-xpu-ops/llm/kv-cache.js";
import { mergeHeadsMajor, splitHeadsMajor } from "../../../web-xpu-ops/llm/reshape.js";
import { sampleNext, type SamplerOptions } from "../../../web-xpu-ops/llm/sampler.js";
import { gatherDequantRow, type PackedQ8, type Qwen3WeightsQ8 } from "./weights-q8.js";

/**
 * The Qwen3-0.6B forward pass over the q8 artifacts — the CPU oracle for the
 * GPU q8 engine, shaped like `model.ts`'s f32 reference but with two
 * deliberate differences:
 *
 *   - RoPE is `ops/rope`'s **adjacent-pair** rotation, not HF's rotate-half.
 *     That is only correct because the q8 artifacts' wq/wk rows and q/k
 *     gammas are already permuted into adjacent-pair channel order
 *     (`WeightsQ8Manifest.ropePermuted`; the loader refuses anything else).
 *   - every projection runs through `matvecQ8`'s reference — the exact
 *     arithmetic the WGSL kernel implements — so this oracle's numbers are
 *     what the GPU should reproduce, not merely approximate.
 *
 * The graph itself is factored over an injected `linear` (the `Graph<W>`
 * parameterisation): the parity test in model-q8.test.ts runs this same
 * forward with f32 matrices and `ops/matvec` injected, proving the
 * adjacent-pair wiring equals the HF graph *before* quantization enters. A
 * hardcoded matvecQ8 would leave "graph bug" and "int8 loss" entangled in
 * every measurement.
 */

export interface GraphConfig {
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

export interface GraphLayer<W> {
  attnNorm: Float32Array;
  /** Rows in ops/rope adjacent-pair channel order. */
  wq: W;
  /** Rows in ops/rope adjacent-pair channel order. */
  wk: W;
  wv: W;
  wo: W;
  /** `[headDim]`, permuted with wq's rows — one gamma shared by every head. */
  qNorm: Float32Array;
  /** `[headDim]`, permuted with wk's rows. */
  kNorm: Float32Array;
  ffnNorm: Float32Array;
  wGate: W;
  wUp: W;
  wDown: W;
}

/** The forward pass's inputs, parameterised over how a weight multiplies a vector. */
export interface Graph<W> {
  config: GraphConfig;
  layers: GraphLayer<W>[];
  finalNorm: Float32Array;
  /** One embedding row for one token id (out-of-vocab ids yield zeros). */
  embedRow(id: number): Float32Array;
  /** `weight @ vector` — matvecQ8 for the q8 graph, ops/matvec for f32 parity. */
  linear(weight: W, vector: Float32Array): Float32Array;
  /** Full-vocab logits for one final-norm row (the tied lm_head). */
  logits(finalRow: Float32Array): Float32Array;
}

/**
 * One generation's state over any `Graph`: the KV cache and the position
 * counter. `prefill` once, then `decodeStep` per token; both return the last
 * position's logits over the full vocabulary. Mirrors `model.ts Qwen3Runner`
 * stage for stage; see that file for the shape of the forward itself.
 */
export class GraphRunner<W> {
  readonly cache: KVCache;
  private position = 0;

  constructor(
    private readonly graph: Graph<W>,
    maxSeqLen: number,
  ) {
    const { numLayers, kvHeads, headDim } = graph.config;
    this.cache = new KVCache(numLayers, kvHeads, headDim, maxSeqLen);
  }

  get positions(): number {
    return this.position;
  }

  prefill(tokens: number[]): Float32Array {
    if (this.position !== 0) throw new Error("prefill on a runner that has already run");
    return this.forward(tokens);
  }

  decodeStep(token: number): Float32Array {
    if (this.position === 0) throw new Error("decodeStep before prefill");
    return this.forward([token]);
  }

  /**
   * `x` as `[T, inDim]` rows, each through `graph.linear`. The q8 wire format
   * has no multi-row form (`matvecQ8` is a GEMV), so prefill is a row loop —
   * fine for an oracle whose prompt is tens of tokens, and it keeps decode
   * and prefill on the *same* arithmetic path, unlike model.ts's
   * matvec/matmul split.
   */
  private projectRows(x: Float32Array, T: number, weight: W): Float32Array {
    const inDim = x.length / T;
    let out: Float32Array | null = null;
    let outDim = 0;
    for (let t = 0; t < T; t += 1) {
      const row = this.graph.linear(weight, x.subarray(t * inDim, (t + 1) * inDim));
      if (out === null) {
        outDim = row.length;
        out = new Float32Array(T * outDim);
      }
      out.set(row, t * outDim);
    }
    return out!;
  }

  private forward(tokens: number[]): Float32Array {
    const { graph } = this;
    const { hidden, heads, kvHeads, headDim, rmsEps, ropeTheta } = graph.config;
    const T = tokens.length;
    const base = this.position;

    const h = new Float32Array(T * hidden);
    for (let t = 0; t < T; t += 1) h.set(graph.embedRow(tokens[t]!), t * hidden);

    for (let l = 0; l < graph.config.numLayers; l += 1) {
      const L = graph.layers[l]!;

      // ---- attention ----
      const normed = rmsnorm({ input: h, weight: L.attnNorm, N: T, D: hidden, eps: rmsEps });
      const q = this.projectRows(normed, T, L.wq);
      const k = this.projectRows(normed, T, L.wk);
      const v = this.projectRows(normed, T, L.wv);

      // QK-norm over each head's channels, one shared (permuted) gamma.
      const qn = rmsnorm({ input: q, weight: L.qNorm, N: T * heads, D: headDim, eps: rmsEps });
      const kn = rmsnorm({ input: k, weight: L.kNorm, N: T * kvHeads, D: headDim, eps: rmsEps });

      // Adjacent-pair RoPE — correct against these weights *because* their
      // channel order was permuted at conversion time.
      const qr = rope({
        input: qn,
        N: T,
        numHeads: heads,
        headDim,
        posOffset: base,
        thetaBase: ropeTheta,
      });
      const kr = rope({
        input: kn,
        N: T,
        numHeads: kvHeads,
        headDim,
        posOffset: base,
        thetaBase: ropeTheta,
      });

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
      const attnOut = this.projectRows(context, T, L.wo);
      for (let i = 0; i < h.length; i += 1) h[i] = h[i]! + attnOut[i]!;

      // ---- mlp ----
      const normed2 = rmsnorm({ input: h, weight: L.ffnNorm, N: T, D: hidden, eps: rmsEps });
      const gate = this.projectRows(normed2, T, L.wGate);
      const up = this.projectRows(normed2, T, L.wUp);
      const gated = elementwise({
        a: activation({ input: gate, kind: ACTIVATION.silu }),
        b: up,
        kind: ELEMENTWISE.multiply,
      });
      const down = this.projectRows(gated, T, L.wDown);
      for (let i = 0; i < h.length; i += 1) h[i] = h[i]! + down[i]!;
    }

    this.position += T;

    const fin = rmsnorm({ input: h, weight: graph.finalNorm, N: T, D: hidden, eps: rmsEps });
    return graph.logits(fin.subarray((T - 1) * hidden, T * hidden));
  }
}

/** `GraphConfig` from the manifest's `Q8Config` field names. */
function configOf(weights: Qwen3WeightsQ8): GraphConfig {
  const c = weights.config;
  return {
    numLayers: c.numLayers,
    hidden: c.hiddenSize,
    heads: c.numHeads,
    kvHeads: c.numKvHeads,
    headDim: c.headDim,
    ffn: c.ffnHidden,
    vocab: c.vocabSize,
    ropeTheta: c.ropeTheta,
    rmsEps: c.rmsNormEps,
    eosIds: c.eosIds,
  };
}

const q8Linear = (w: PackedQ8, vector: Float32Array): Float32Array =>
  matvecQ8({ weight: w.packed, scale: w.scale, vector, N: w.rows, K: w.cols });

/** The q8 graph: every projection (and the tied lm_head) through `matvecQ8`. */
export function q8Graph(weights: Qwen3WeightsQ8): Graph<PackedQ8> {
  return {
    config: configOf(weights),
    layers: weights.perLayer.map((L) => ({
      attnNorm: L.attnNorm,
      wq: L.wq,
      wk: L.wk,
      wv: L.wv,
      wo: L.wo,
      qNorm: L.qNorm,
      kNorm: L.kNorm,
      ffnNorm: L.ffnNorm,
      wGate: L.wGate,
      wUp: L.wUp,
      wDown: L.wDown,
    })),
    finalNorm: weights.finalNorm,
    embedRow: (id) => gatherDequantRow(weights.embedTokens, id),
    linear: q8Linear,
    logits: (row) => q8Linear(weights.embedTokens, row),
  };
}

export class Qwen3RunnerQ8 extends GraphRunner<PackedQ8> {
  constructor(weights: Qwen3WeightsQ8, maxSeqLen: number) {
    super(q8Graph(weights), maxSeqLen);
  }
}

export interface GenerateQ8Options {
  /** Stop at (and include) the first eos id. Default true. */
  stopAtEos?: boolean;
  /** Defaults to greedy. Top-p is what the reference server runs. */
  sampler?: SamplerOptions;
}

/**
 * One prefill, then the KV-cached decode path, sampling each step with
 * `llm/sampler.js`. Greedy by default — `{mode: "greedy"}` is argmax, so the
 * default call is comparable against the golden's greedy dumps.
 */
export function generateQ8(
  promptIds: number[],
  maxNew: number,
  weights: Qwen3WeightsQ8,
  opts: GenerateQ8Options = {},
): number[] {
  const stopAtEos = opts.stopAtEos ?? true;
  const sampler: SamplerOptions = opts.sampler ?? { mode: "greedy" };
  const runner = new Qwen3RunnerQ8(weights, promptIds.length + maxNew);
  let logits = runner.prefill(promptIds);
  const out: number[] = [];
  for (let step = 0; step < maxNew; step += 1) {
    const id = sampleNext(logits, [...promptIds, ...out], sampler);
    out.push(id);
    if (stopAtEos && weights.config.eosIds.includes(id)) break;
    if (step + 1 < maxNew) logits = runner.decodeStep(id);
  }
  return out;
}
