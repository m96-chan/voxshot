import {
  type TextBlockWeights,
  addInto,
  linear,
  textBlock,
  transpose,
} from "./blocks.js";
import type { ModelWeights } from "./model-weights.js";

/**
 * `ReferenceLatentEncoder` — what turns the reference clip into speaker
 * conditioning.
 *
 * Eight `TextBlock`s over the reference's DACVAE latents, patched 4 frames at a
 * time so 32 latent channels arrive as 128 features. Nothing is learned about
 * *which* speaker; the reference clip is the entire identity, which is what
 * makes this zero-shot.
 *
 * Two details are easy to lose and neither shows up as an error.
 *
 * **The input is divided by 6.** Right after `in_proj`, before any block. It is
 * a scale the training settled on, it is written as a bare `x = x / 6.0`, and
 * without it every block sees activations six times too large and the encoder
 * still produces a plausible-looking state.
 *
 * **Masked positions are forced to zero after every block**, not just at the
 * end. The residual path would otherwise carry padding forward: attention
 * output at a masked row is zero, but `x + 0` is still `x`.
 *
 * That last one is **unverified**, and saying so is more useful than the check
 * being green. A single reference clip at batch 1 has no padding — all 123
 * frames are kept — so removing the per-block zeroing changes nothing and
 * `check-speaker.ts` still passes. It is ported because the reference does it,
 * not because anything here has demonstrated it matters. Batching, or a
 * reference shorter than another in the same batch, would make it checkable.
 */

export interface SpeakerEncoderResult {
  /** `[frames, speakerDim]`. */
  state: Float32Array;
  captured: Map<string, Float32Array>;
}

export function runSpeakerEncoder({
  weights,
  latent,
  keep,
  capture,
}: {
  weights: ModelWeights;
  /** `[frames, patchedLatentDim]`, already patched. */
  latent: Float32Array;
  keep: boolean[];
  capture?: Iterable<string>;
}): SpeakerEncoderResult {
  const { speaker } = weights;
  const frames = keep.length;
  const dim = speaker.dim;
  const wanted = new Set(capture ?? []);
  const captured = new Map<string, Float32Array>();

  const zeroMasked = (x: Float32Array): Float32Array => {
    for (let frame = 0; frame < frames; frame += 1) {
      if (keep[frame]) continue;
      x.fill(0, frame * dim, (frame + 1) * dim);
    }
    return x;
  };

  let x = linear(latent, speaker.inProjWeight, frames, speaker.patchedLatentDim, dim, speaker.inProjBias);
  // `x = x / 6.0` in the reference, on its own line, with no comment.
  for (let i = 0; i < x.length; i += 1) x[i]! /= 6;
  zeroMasked(x);
  if (wanted.has("in_proj")) captured.set("in_proj", x.slice());

  for (let index = 0; index < speaker.blocks.length; index += 1) {
    x = zeroMasked(
      textBlock(x, speaker.blocks[index]!, keep, dim, speaker.heads, speaker.mlpHidden, weights.normEps),
    );
    if (wanted.has(`blocks.${index}`)) captured.set(`blocks.${index}`, x.slice());
  }

  return { state: x, captured };
}

/**
 * Patch the reference latent the way `_patch_reference` does.
 *
 * `[frames, latentDim]` to `[frames / patch, latentDim * patch]`, dropping the
 * remainder, with a frame kept only when **every** frame in its patch was.
 */
export function patchReference(
  latent: Float32Array,
  keep: boolean[],
  latentDim: number,
  patch: number,
): { latent: Float32Array; keep: boolean[] } {
  const usable = Math.floor(keep.length / patch) * patch;
  if (usable <= 0) throw new Error(`reference of ${keep.length} frames is shorter than one patch`);
  const patched = new Float32Array((usable / patch) * latentDim * patch);
  patched.set(latent.subarray(0, patched.length));
  const patchedKeep: boolean[] = [];
  for (let at = 0; at < usable; at += patch) {
    patchedKeep.push(keep.slice(at, at + patch).every(Boolean));
  }
  return { latent: patched, keep: patchedKeep };
}

/** Build the eight blocks' weights out of a flat tensor map. */
export function speakerBlocks(
  get: (name: string) => Float32Array,
  count: number,
  dim: number,
  hidden: number,
): TextBlockWeights[] {
  const blocks: TextBlockWeights[] = [];
  for (let index = 0; index < count; index += 1) {
    const at = `speaker_encoder.blocks.${index}`;
    blocks.push({
      attentionNorm: get(`${at}.attention_norm.weight`),
      attention: {
        wq: transpose(get(`${at}.attention.wq.weight`), dim, dim),
        wk: transpose(get(`${at}.attention.wk.weight`), dim, dim),
        wv: transpose(get(`${at}.attention.wv.weight`), dim, dim),
        wo: transpose(get(`${at}.attention.wo.weight`), dim, dim),
        gate: transpose(get(`${at}.attention.gate.weight`), dim, dim),
        qNorm: get(`${at}.attention.q_norm.weight`),
        kNorm: get(`${at}.attention.k_norm.weight`),
      },
      mlpNorm: get(`${at}.mlp_norm.weight`),
      w1: transpose(get(`${at}.mlp.w1.weight`), hidden, dim),
      w2: transpose(get(`${at}.mlp.w2.weight`), dim, hidden),
      w3: transpose(get(`${at}.mlp.w3.weight`), hidden, dim),
    });
  }
  return blocks;
}

export { addInto };
