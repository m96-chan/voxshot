import type { Context } from "./dit.js";
import { type Modulation, prepareGpu, velocityGpu } from "./dit-gpu.js";
import type { Gpu, Tensor } from "./gpu.js";
import type { ModelWeights } from "./model-weights.js";
import { linearSchedule, timestepEmbedding } from "./sampler.js";

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
 * ## The modulation is computed for every step at once
 *
 * `cond_module` and AdaLN's low-rank refinement used to run on the host, per
 * step, on the reasoning that they are `[batch, dim]`-sized and negligible
 * against the DiT's 40 GFLOP. **That reasoning was wrong**, and measuring it is
 * what showed the error: the comparison was against the DiT's cost *on the
 * device*, while these ran on web-xpu-ops' CPU reference, which is about a
 * hundred times slower per FLOP. 24 refinements a step came to **2.3 seconds**
 * of a 4.7-second loop — half of a stage that had already been called fast.
 *
 * They are on the device now, and computed for all 32 steps in one go rather
 * than per step: the timesteps are known before the loop starts, so the whole
 * schedule goes through as `[steps * batch, dim]` and each step binds a window
 * onto its own rows. About 450 dispatches for the run instead of per step.
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

  const schedule = linearSchedule(steps);

  /**
   * Every step's `cond_module` output, as `[3, steps * batch, dim]`.
   *
   * `Linear -> SiLU -> Linear -> SiLU -> Linear`, then one `permute` so each of
   * shift, scale and gate is a contiguous window instead of every third slice
   * of a row.
   */
  const condFor = (batch: number, which: number[]): Tensor => {
    const rows = which.length * batch;
    const embed = new Float32Array(rows * config.timestep_embed_dim);
    which.forEach((step, at) => {
      embed.set(
        timestepEmbedding(schedule[step]!, config.timestep_embed_dim, batch),
        at * batch * config.timestep_embed_dim,
      );
    });
    const input = gpu.writeInto(`cond.in.${batch}`, embed);
    const h1 = gpu.activation(
      gpu.matmul(input, gpu.weight(weights.cond.in1), rows, dim, config.timestep_embed_dim, `cond.1.${batch}`),
      1, // SiLU
      `cond.1a.${batch}`,
    );
    const h2 = gpu.activation(
      gpu.matmul(h1, gpu.weight(weights.cond.in2), rows, dim, dim, `cond.2.${batch}`),
      1,
      `cond.2a.${batch}`,
    );
    const wide = gpu.matmul(h2, gpu.weight(weights.cond.out), rows, 3 * dim, dim, `cond.3.${batch}`);
    return gpu.permute(wide, rows, 3, dim, `cond.p.${batch}`);
  };

  /**
   * `up(down(silu(c))) + c`, then `1 + scale` and `tanh(gate)`, for every step.
   *
   * The bias is added by gathering a single row over every output row — the
   * same standing-in-for-broadcast trick the DiT's AdaLN uses.
   */
  const refineAll = (batch: number, rows: number, cond: Tensor) => {
    const ones = gpu.writeInto(`ada.ones.${batch}`, new Float32Array(rows * dim).fill(1));
    const zeros = gpu.writeIntsInto(`ada.zero.${batch}`, new Int32Array(rows));
    return dit.blocks.map((block, index) => {
      const one = (name: string, w: typeof block.attentionAdaLn) => {
        const part = (which: number, down: Float32Array, up: Float32Array, bias: Float32Array, tag: string) => {
          const slot = `ada.${batch}.${index}.${name}.${tag}`;
          const source = gpu.view(cond, which * rows * dim, rows * dim);
          const low = gpu.matmul(gpu.activation(source, 1, `${slot}.a`), gpu.weight(down), rows, rank, dim, `${slot}.d`);
          const wide = gpu.matmul(low, gpu.weight(up), rows, dim, rank, `${slot}.u`);
          const biased = gpu.elementwise(
            wide,
            gpu.gather(gpu.weight(bias), zeros, rows, dim, 1, `${slot}.b`),
            0,
            `${slot}.bb`,
          );
          return gpu.elementwise(biased, source, 0, slot);
        };
        const shift = part(0, w.shiftDown, w.shiftUp, w.shiftBias, "f");
        const scale = gpu.elementwise(part(1, w.scaleDown, w.scaleUp, w.scaleBias, "s"), ones, 0, `ada.${batch}.${index}.${name}.s1`);
        const gate = gpu.activation(part(2, w.gateDown, w.gateUp, w.gateBias, "g"), 3, `ada.${batch}.${index}.${name}.gt`);
        return { shift, scale, gate };
      };
      return { attention: one("attn", block.attentionAdaLn), mlp: one("mlp", block.mlpAdaLn) };
    });
  };

  // Two schedules, because guidance changes the batch and shapes are baked into
  // bind groups. Which steps fall on which side is decided here, once.
  const guidedSteps: number[] = [];
  const plainSteps: number[] = [];
  for (let step = 0; step < steps; step += 1) {
    const t = schedule[step]!;
    (t >= cfgMinT && t <= cfgMaxT ? guidedSteps : plainSteps).push(step);
  }
  const all = new Map<number, { steps: number[]; mod: ReturnType<typeof refineAll> }>();
  for (const [batch, list] of [
    [guidedBatch, guidedSteps],
    [1, plainSteps],
  ] as const) {
    if (list.length === 0) continue;
    all.set(batch, { steps: list, mod: refineAll(batch, list.length * batch, condFor(batch, list)) });
  }
  let x = noise.slice();

  for (let step = 0; step < steps; step += 1) {
    const t = schedule[step]!;
    const isGuided = t >= cfgMinT && t <= cfgMaxT;
    const batch = isGuided ? guidedBatch : 1;
    onStep?.(step, t, isGuided);

    const stacked = new Float32Array(batch * width);
    for (let b = 0; b < batch; b += 1) stacked.set(x, b * width);

    // This step's rows of the precomputed modulation, bound as windows.
    const prepared_ = all.get(batch)!;
    const row = prepared_.steps.indexOf(step) * batch * dim;
    const modulation: { attention: Modulation; mlp: Modulation }[] = prepared_.mod.map((block) => ({
      attention: {
        scale: gpu.view(block.attention.scale, row, batch * dim),
        shift: gpu.view(block.attention.shift, row, batch * dim),
        gate: gpu.view(block.attention.gate, row, batch * dim),
      },
      mlp: {
        scale: gpu.view(block.mlp.scale, row, batch * dim),
        shift: gpu.view(block.mlp.shift, row, batch * dim),
        gate: gpu.view(block.mlp.gate, row, batch * dim),
      },
    }));

    const out = await gpu.read(
      velocityGpu(prepared.get(batch)!, gpu.writeInto(`x.${batch}`, stacked), modulation),
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
