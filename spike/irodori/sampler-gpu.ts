import type { Context } from "./dit.js";
import { type Modulation, prepareGpu, velocityGpu } from "./dit-gpu.js";
import type { Gpu, Tensor } from "./gpu.js";
import type { ModelWeights } from "./model-weights.js";
import { adaLnVectors, condModule, linearSchedule, timestepEmbedding } from "./sampler.js";

/**
 * The flow loop with the DiT on the device.
 *
 * Same schedule, same guidance, same Euler update as `sampler.ts` — that file
 * is the definition and this one is the fast path. What differs is only where
 * the arithmetic happens.
 *
 * ## What still crosses back each step
 *
 * `x_t`, and only `x_t`: `[tokens, 32]`, about 12 KB. The Euler update and the
 * guidance combination are host arithmetic over that, which costs nothing next
 * to a forward pass and keeps the two samplers obviously the same algorithm.
 *
 * Everything else stays down. The context projections are built once — `wk` and
 * `wv` run over 892 tokens, more than anything in a step, and they are
 * identical across all 32 — and the modulation vectors are uploaded per step at
 * `[batch, 1280]` each.
 *
 * ## Two prepared graphs, not one
 *
 * Guidance runs at batch 3 while `t >= 0.5` and batch 1 after. Shapes are baked
 * into bind groups, so the two are prepared separately and the loop picks. The
 * weights are shared: `Gpu.weight` is keyed on the array, so the second
 * preparation uploads only its own contexts.
 */

export interface GpuSampleArgs {
  gpu: Gpu;
  weights: ModelWeights;
  /** `[tokens, latentDim]`, the initial noise. */
  noise: Float32Array;
  /** Conditional stacked with each guided variant, on the batch axis. */
  guided: Record<string, Context>;
  /** Conditional alone. */
  plain: Record<string, Context>;
  /** One scale per dropped condition, in the order they are stacked. */
  scales: number[];
  steps: number;
  cfgMinT?: number;
  cfgMaxT?: number;
  onStep?: (step: number, t: number, guided: boolean) => void;
}

export async function sampleGpu({
  gpu,
  weights,
  noise,
  guided,
  plain,
  scales,
  steps,
  cfgMinT = 0.5,
  cfgMaxT = 1.0,
  onStep,
}: GpuSampleArgs): Promise<Float32Array> {
  const { dit, config } = weights;
  const { dim, rank } = dit.shape;
  const width = noise.length;
  const tokens = width / config.latent_dim;
  const guidedBatch = 1 + scales.length;

  const prepared = new Map<number, ReturnType<typeof prepareGpu>>();
  prepared.set(guidedBatch, prepareGpu({ gpu, weights, contexts: guided, batch: guidedBatch, tokens }));
  prepared.set(1, prepareGpu({ gpu, weights, contexts: plain, batch: 1, tokens }));

  // One upload slot per (batch, block, which, field) — the values change every
  // step, the buffers do not, so the bind groups built on them stay cached.
  const modulationFor = (batch: number, cond: Float32Array): { attention: Modulation; mlp: Modulation }[] =>
    dit.blocks.map((block, index) => {
      const one = (name: string, w: typeof block.attentionAdaLn): Modulation => {
        const { scale, shift, gate } = adaLnVectors(cond, w, dim, rank, batch);
        return {
          scale: gpu.writeInto(`mod.${batch}.${index}.${name}.s`, scale),
          shift: gpu.writeInto(`mod.${batch}.${index}.${name}.f`, shift),
          gate: gpu.writeInto(`mod.${batch}.${index}.${name}.g`, gate),
        };
      };
      return { attention: one("attn", block.attentionAdaLn), mlp: one("mlp", block.mlpAdaLn) };
    });

  const schedule = linearSchedule(steps);
  let x = noise.slice();

  for (let step = 0; step < steps; step += 1) {
    const t = schedule[step]!;
    const isGuided = t >= cfgMinT && t <= cfgMaxT;
    const batch = isGuided ? guidedBatch : 1;
    onStep?.(step, t, isGuided);

    const cond = condModule(
      timestepEmbedding(t, config.timestep_embed_dim, batch),
      weights.cond,
      dim,
      batch,
    );

    const stacked = new Float32Array(batch * width);
    for (let b = 0; b < batch; b += 1) stacked.set(x, b * width);

    const out = await gpu.read(
      velocityGpu(
        prepared.get(batch)!,
        gpu.writeInto(`x.${batch}`, stacked),
        modulationFor(batch, cond),
      ),
    );

    const v = out.slice(0, width);
    for (let which = 0; which < scales.length && isGuided; which += 1) {
      const scale = scales[which]!;
      const dropped = out.subarray((which + 1) * width, (which + 2) * width);
      for (let i = 0; i < width; i += 1) v[i]! += scale * (out[i]! - dropped[i]!);
    }

    const dt = schedule[step + 1]! - t;
    for (let i = 0; i < width; i += 1) x[i]! += v[i]! * dt;
  }
  return x;
}

export type { Tensor };
