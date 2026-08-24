import { ACTIVATION } from "web-xpu-ops/ops/activation";

import { ROPE_THETA } from "./blocks.js";
import { filled, halveFor as halve, repeatRowsFor as repeatRows } from "./dit-gpu.js";
import type { Gpu, Tensor } from "./gpu.js";
import type { ModelWeights } from "./model-weights.js";

/**
 * `ReferenceLatentEncoder` on the device — the same graph as
 * `speaker-encoder.ts`.
 *
 * That file is the definition of correct; this one exists because swapping the
 * voice is the point of the demo and the CPU reference took **six seconds** for
 * a twenty-second clip. Waiting six seconds to hear a different speaker is not
 * a demonstration of zero-shot cloning, it is a demonstration of patience.
 *
 * ## `sigmoid` is folded into two weights, again
 *
 * `SelfAttention` ends in `wo(y * sigmoid(gate(x)))` and web-xpu-ops has `tanh`
 * but not `sigmoid`. With `sigmoid(g) = (tanh(g/2) + 1) / 2`, halving `gate`
 * and `wo` at upload removes every constant — the same trick `dit-gpu.ts` uses,
 * and the same warning: these two weights no longer match the checkpoint.
 *
 * ## The masking that could not be checked is still here
 *
 * The reference zeroes masked positions after every block. `spike/irodori`'s
 * notes say why that is unverified — one clip at batch 1 has no padding, so
 * removing it changes nothing and the check stays green. It is done here for
 * the same reason it is done there: the reference does it.
 */

const ADD = 0;
const MULTIPLY = 1;

export interface GpuSpeakerBlock {
  attentionNorm: Tensor;
  wq: Tensor;
  wk: Tensor;
  wv: Tensor;
  /** Halved — see the module note. */
  gate: Tensor;
  /** Halved. */
  wo: Tensor;
  qNorm: Tensor;
  kNorm: Tensor;
  mlpNorm: Tensor;
  w1: Tensor;
  w2: Tensor;
  w3: Tensor;
}

export interface GpuSpeaker {
  gpu: Gpu;
  blocks: GpuSpeakerBlock[];
  inProj: Tensor;
  /** `in_proj`'s bias, repeated per row so `elementwise` can add it. */
  inProjBias: Tensor;
  outNorm: Tensor;
  /** `[frames, dim]` of ones, for `tanh(g) + 1` and for the unweighted RMSNorm. */
  ones: Tensor;
  /** `[frames]` additive key bias. */
  keyBias: Tensor;
  /** `[frames, dim]` of 1/6, folded as a multiply rather than a constant. */
  sixth: Tensor;
  dim: number;
  heads: number;
  mlpHidden: number;
  eps: number;
  frames: number;
}

/**
 * Upload the weights once and the per-clip shapes per call.
 *
 * The weights go through `Gpu.weight`, so a second clip re-uses them; the
 * `[frames, dim]` helpers are written into named slots, so a second clip of a
 * different length gets its own buffer and one of the same length reuses it.
 */
export function prepareSpeaker(
  gpu: Gpu,
  weights: ModelWeights,
  keep: boolean[],
): GpuSpeaker {
  const { speaker, normEps } = weights;
  const frames = keep.length;
  const dim = speaker.dim;
  return {
    gpu,
    blocks: speaker.blocks.map((block) => ({
      attentionNorm: gpu.weight(block.attentionNorm),
      wq: gpu.weight(block.attention.wq),
      wk: gpu.weight(block.attention.wk),
      wv: gpu.weight(block.attention.wv),
      gate: gpu.weight(halve(block.attention.gate)),
      wo: gpu.weight(halve(block.attention.wo)),
      qNorm: gpu.weight(block.attention.qNorm),
      kNorm: gpu.weight(block.attention.kNorm),
      mlpNorm: gpu.weight(block.mlpNorm),
      w1: gpu.weight(block.w1),
      w2: gpu.weight(block.w2),
      w3: gpu.weight(block.w3),
    })),
    inProj: gpu.weight(speaker.inProjWeight),
    inProjBias: gpu.writeInto("spk.bias", repeatRows(speaker.inProjBias, frames)),
    outNorm: gpu.weight(speaker.outNorm),
    ones: gpu.writeInto("spk.ones", filled(1, frames * dim)),
    keyBias: gpu.writeInto("spk.mask", Float32Array.from(keep, (k) => (k ? 0 : -Infinity))),
    // `x = x / 6.0` in the reference, on its own line, with no comment. As a
    // tensor because `elementwise` has no scalar form.
    sixth: gpu.writeInto("spk.sixth", filled(1 / 6, frames * dim)),
    dim,
    heads: speaker.heads,
    mlpHidden: speaker.mlpHidden,
    eps: normEps,
    frames,
  };
}

/** `[frames, dim]` after the eight blocks and `speaker_norm`. */
export function runSpeakerGpu(speaker: GpuSpeaker, latent: Tensor, patchedDim: number): Tensor {
  const { gpu, dim, heads, mlpHidden, eps, frames } = speaker;
  const headDim = dim / heads;
  const at = (name: string) => `spk.${name}`;

  let x = gpu.elementwise(
    gpu.matmul(latent, speaker.inProj, frames, dim, patchedDim, at("in")),
    speaker.inProjBias,
    ADD,
    at("in.b"),
  );
  x = gpu.elementwise(x, speaker.sixth, MULTIPLY, at("in.6"));

  for (let index = 0; index < speaker.blocks.length; index += 1) {
    const block = speaker.blocks[index]!;

    const normed = gpu.rmsnorm(x, block.attentionNorm, frames, dim, eps, at("an"));
    const project = (weight: Tensor, slot: string) =>
      gpu.matmul(normed, weight, frames, dim, dim, at(slot));
    const rotate = (input: Tensor, slot: string) =>
      gpu.rope(input, frames, heads, headDim, ROPE_THETA, at(slot));

    const attended = gpu.attention({
      q: gpu.permute(
        rotate(gpu.rmsnorm(project(block.wq, "q"), block.qNorm, frames * heads, headDim, eps, at("qn"), heads), "qr"),
        frames,
        heads,
        headDim,
        at("qh"),
      ),
      k: gpu.permute(
        rotate(gpu.rmsnorm(project(block.wk, "k"), block.kNorm, frames * heads, headDim, eps, at("kn"), heads), "kr"),
        frames,
        heads,
        headDim,
        at("kh"),
      ),
      v: gpu.permute(project(block.wv, "v"), frames, heads, headDim, at("vh")),
      mask: speaker.keyBias,
      maskShape: [1, 1, 1],
      B: 1,
      H: heads,
      L: frames,
      S: frames,
      D: headDim,
      scale: 1 / Math.sqrt(headDim),
      slot: at("attn"),
    });

    // `y * sigmoid(gate)`, with the halves folded into `gate` and `wo`.
    const y = gpu.permute(attended, heads, frames, headDim, at("yh"));
    const gated = gpu.elementwise(
      y,
      gpu.elementwise(
        gpu.activation(project(block.gate, "g"), ACTIVATION.tanh, at("gt")),
        speaker.ones,
        ADD,
        at("g1"),
      ),
      MULTIPLY,
      at("gy"),
    );
    x = gpu.elementwise(x, gpu.matmul(gated, block.wo, frames, dim, dim, at("wo")), ADD, at("res1"));

    const mlpIn = gpu.rmsnorm(x, block.mlpNorm, frames, dim, eps, at("mn"));
    const up = gpu.elementwise(
      gpu.activation(gpu.matmul(mlpIn, block.w1, frames, mlpHidden, dim, at("w1")), ACTIVATION.silu, at("w1a")),
      gpu.matmul(mlpIn, block.w3, frames, mlpHidden, dim, at("w3")),
      MULTIPLY,
      at("swiglu"),
    );
    x = gpu.elementwise(x, gpu.matmul(up, block.w2, frames, dim, mlpHidden, at("w2")), ADD, at("res2"));

    // The block writes into pooled slots; its result is copied out before the
    // next block reuses them.
    const kept = gpu.scratch(`spk.layer.${index % 2}`, x.length);
    gpu.copy(x, 0, kept, 0, x.length * 4);
    x = kept;
  }

  return gpu.rmsnorm(x, speaker.outNorm, frames, dim, eps, at("final"));
}
