import { ACTIVATION, activation } from "web-xpu-ops/ops/activation";
import { rmsnorm } from "web-xpu-ops/ops/rmsnorm";

import { linear } from "./blocks.js";
import type { AdaLnWeights } from "./dit.js";
import { type Context, type ContextKv, ditBlock, projectContexts } from "./dit.js";
import type { ModelWeights } from "./model-weights.js";

/**
 * The rectified-flow sampler: `sample_euler_rf_cfg`, plus the DiT's own head
 * and tail.
 *
 * Rectified flow is a straight line. `x_t = (1 - t) * x0 + t * noise`, so the
 * velocity the model predicts is a constant `noise - x0` along that line, and
 * sampling is Euler integration from `t = 0.999` down to `t = 0`:
 * `x <- x + v * (t_next - t)`. Thirty-two steps, no schedule to learn and no
 * variance to track — which is why this model is non-autoregressive and why 32
 * forward passes cover any length of audio.
 *
 * **Guidance runs as a batch, and only for the first half.** With
 * `cfg_guidance_mode: "independent"` the conditional and each unconditional
 * variant go through one forward pass stacked on the batch axis, and the
 * velocities are combined as
 *
 *     v = v_cond + sum over each dropped condition of
 *             scale * (v_cond - v_without_that_condition)
 *
 * The reference gates that on `0.5 <= t <= 1.0`, and with the linear schedule
 * `t_i = 0.999 * (1 - i / 32)` the crossing lands between step 15 and step 16.
 * That is visible in the goldens without reading a line of `rf.py`: the first
 * recorded fire is `[3, 97, 1280]` and the ones at steps 16 and 31 are
 * `[1, 97, 1280]`.
 */

/** `get_timestep_embedding` — sinusoidal, and scaled by 1000 rather than by t. */
export function timestepEmbedding(t: number, dim: number, batch = 1): Float32Array {
  if (dim % 2 !== 0) throw new Error(`timestep embedding needs an even dim, got ${dim}`);
  const half = dim / 2;
  const out = new Float32Array(batch * dim);
  for (let index = 0; index < half; index += 1) {
    // `1000.0 * exp(-log(10000) * i / half)` — the 1000 is a factor on the
    // frequency, not on the timestep, and dropping it changes which part of the
    // sinusoid every step lands on.
    const freq = 1000 * Math.exp((-Math.log(10000) * index) / half);
    const angle = t * freq;
    for (let b = 0; b < batch; b += 1) {
      out[b * dim + index] = Math.cos(angle);
      out[b * dim + half + index] = Math.sin(angle);
    }
  }
  return out;
}

export interface CondModuleWeights {
  in1: Float32Array;
  in2: Float32Array;
  out: Float32Array;
  embedDim: number;
}

/** `Linear -> SiLU -> Linear -> SiLU -> Linear`, all without bias. */
export function condModule(
  embedding: Float32Array,
  w: CondModuleWeights,
  dim: number,
  batch: number,
): Float32Array {
  let h = activation({
    input: linear(embedding, w.in1, batch, w.embedDim, dim),
    kind: ACTIVATION.silu,
  });
  h = activation({ input: linear(h, w.in2, batch, dim, dim), kind: ACTIVATION.silu });
  return linear(h, w.out, batch, dim, 3 * dim);
}

/** `t_schedule = (1 - linspace(0, 1, steps + 1)) * 0.999`. */
export function linearSchedule(steps: number): Float64Array {
  const schedule = new Float64Array(steps + 1);
  for (let index = 0; index <= steps; index += 1) schedule[index] = (1 - index / steps) * 0.999;
  return schedule;
}

export interface Conditions {
  /** Conditional, and one entry per guided condition, stacked on the batch axis. */
  guided: Record<string, Context>;
  /** Conditional only, for the steps below `cfgMinT`. */
  plain: Record<string, Context>;
  /** Scale per dropped condition, in the order they are stacked after the conditional. */
  scales: number[];
}

export interface SampleArgs {
  weights: ModelWeights;
  cond: CondModuleWeights;
  /** `[tokens, latentDim]`, the reference's own initial noise. */
  noise: Float32Array;
  conditions: Conditions;
  steps: number;
  cfgMinT?: number;
  cfgMaxT?: number;
  /** Called with `x_t` before each step, for comparing against recorded ones. */
  onStep?: (step: number, t: number, x: Float32Array) => void;
}

/** `out_norm` then `out_proj` — model dim back to the latent. */
function head(x: Float32Array, weights: ModelWeights, rows: number): Float32Array {
  const { dit } = weights;
  const normed = rmsnorm({
    input: x,
    weight: dit.outNorm,
    N: rows,
    D: dit.shape.dim,
    eps: dit.shape.eps,
  });
  return linear(normed, dit.outProjWeight, rows, dit.shape.dim, weights.config.latent_dim, dit.outProjBias);
}

export function velocityFor(
  x: Float32Array,
  t: number,
  weights: ModelWeights,
  contexts: ContextKv[],
  tokens: number,
  batch: number,
): Float32Array {
  const { dit, config } = weights;
  const embedded = condModule(
    timestepEmbedding(t, config.timestep_embed_dim, batch),
    weights.cond,
    dit.shape.dim,
    batch,
  );
  let h = linear(x, dit.inProjWeight, batch * tokens, config.latent_dim, dit.shape.dim, dit.inProjBias);
  for (let index = 0; index < dit.blocks.length; index += 1) {
    h = ditBlock(h, embedded, dit.blocks[index]!, contexts[index]!, tokens, dit.shape);
  }
  return head(h, weights, batch * tokens);
}

/**
 * AdaLN's low-rank refinement, on the host.
 *
 * `[batch, dim]`-sized — about 2 MFLOP a block against the DiT's 40 GFLOP a
 * step — so the device version keeps it here too. What crosses down is already
 * `1 + scale` and `tanh(gate)`, which is why the graph on the device has no
 * constants in it.
 */
export function adaLnVectors(
  cond: Float32Array,
  w: AdaLnWeights,
  dim: number,
  rank: number,
  batch: number,
): { scale: Float32Array; shift: Float32Array; gate: Float32Array } {
  const refine = (part: number, down: Float32Array, up: Float32Array, bias: Float32Array) => {
    const source = new Float32Array(batch * dim);
    for (let b = 0; b < batch; b += 1) {
      source.set(cond.subarray(b * 3 * dim + part * dim, b * 3 * dim + (part + 1) * dim), b * dim);
    }
    const out = linear(
      linear(activation({ input: source.slice(), kind: ACTIVATION.silu }), down, batch, dim, rank),
      up,
      batch,
      rank,
      dim,
      bias,
    );
    for (let i = 0; i < out.length; i += 1) out[i]! += source[i]!;
    return out;
  };
  const shift = refine(0, w.shiftDown, w.shiftUp, w.shiftBias);
  const scale = refine(1, w.scaleDown, w.scaleUp, w.scaleBias);
  const gate = refine(2, w.gateDown, w.gateUp, w.gateBias);
  for (let i = 0; i < scale.length; i += 1) scale[i]! += 1;
  for (let i = 0; i < gate.length; i += 1) gate[i] = Math.tanh(gate[i]!);
  return { scale, shift, gate };
}

export function sample({
  weights,
  cond,
  noise,
  conditions,
  steps,
  cfgMinT = 0.5,
  cfgMaxT = 1.0,
  onStep,
}: SampleArgs): Float32Array {
  const { dit, config } = weights;
  const tokens = noise.length / config.latent_dim;
  const guidedBatch = 1 + conditions.scales.length;

  // The contexts do not change across steps, so their projections are built
  // once per block — which is what `build_context_kv_cache` is for in the
  // reference. Two sets, because the guided steps run a wider batch.
  const project = (contexts: Record<string, Context>, batch: number) =>
    dit.blocks.map((block) => projectContexts(contexts, block, dit.shape, batch));
  const guidedKv = project(conditions.guided, guidedBatch);
  const plainKv = project(conditions.plain, 1);

  const schedule = linearSchedule(steps);
  let x = noise.slice();

  for (let step = 0; step < steps; step += 1) {
    const t = schedule[step]!;
    onStep?.(step, t, x);
    const guided = t >= cfgMinT && t <= cfgMaxT;

    let v: Float32Array;
    if (guided) {
      // One forward pass over the stacked batch, then the velocities combine.
      const stacked = new Float32Array(guidedBatch * x.length);
      for (let b = 0; b < guidedBatch; b += 1) stacked.set(x, b * x.length);
      const out = velocityFor(stacked, t, weights, guidedKv, tokens, guidedBatch);
      const width = tokens * config.latent_dim;
      v = out.slice(0, width);
      for (let which = 0; which < conditions.scales.length; which += 1) {
        const scale = conditions.scales[which]!;
        const dropped = out.subarray((which + 1) * width, (which + 2) * width);
        for (let i = 0; i < width; i += 1) v[i]! += scale * (out[i]! - dropped[i]!);
      }
    } else {
      v = velocityFor(x, t, weights, plainKv, tokens, 1);
    }

    const dt = schedule[step + 1]! - t;
    for (let i = 0; i < x.length; i += 1) x[i]! += v[i]! * dt;
  }
  onStep?.(steps, schedule[steps]!, x);
  return x;
}
