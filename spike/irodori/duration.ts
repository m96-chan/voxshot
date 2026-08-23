import { ACTIVATION, activation } from "web-xpu-ops/ops/activation";
import { rmsnorm } from "web-xpu-ops/ops/rmsnorm";

import { linear, swiglu, transpose } from "./blocks.js";
import type { ModelWeights } from "./model-weights.js";

/**
 * How long the audio will be, decided before a single latent frame is sampled.
 *
 * This is what a non-autoregressive model needs and an autoregressive one does
 * not: MioTTS stops when it emits an end token, but rectified flow integrates a
 * fixed-size tensor, so the size has to be known first. Get it wrong and the
 * speech is not mispronounced — it is stretched or clipped.
 *
 * The checkpoint's architecture is `token_sum_dual_adarn_zero_no_aux`, and
 * every word in that name is a branch the reference could have taken:
 *
 * **token_sum** — a frame count is predicted *per text token* and summed, not
 * pooled into one number. `softplus` keeps each contribution positive, the text
 * mask drops the padding, and the result is returned as `log1p` of the total.
 *
 * **dual_adarn_zero** — the speaker and the caption each modulate every block,
 * through their own zero-initialised `Linear`, and the two modulations are
 * *added* before being applied. Not concatenated, not applied in sequence.
 *
 * **no_aux** — `aux_features` is an argument the reference still accepts, still
 * validates, and on this branch never reads. Fourteen numbers that go nowhere.
 */

export interface DurationBlockWeights {
  norm: Float32Array;
  w1: Float32Array;
  w2: Float32Array;
  w3: Float32Array;
  modulation: Float32Array;
  modulationBias: Float32Array;
  captionModulation: Float32Array;
  captionModulationBias: Float32Array;
}

export interface DurationWeights {
  dim: number;
  hidden: number;
  inProj: Float32Array;
  inProjBias: Float32Array;
  blocks: DurationBlockWeights[];
  outNorm: Float32Array;
  outProj: Float32Array;
  outProjBias: Float32Array;
  nullSpeaker: Float32Array;
  nullCaption: Float32Array;
}

/** `modulation(silu(cond))`, split into shift, scale and gate. */
function modulate(cond: Float32Array, weight: Float32Array, bias: Float32Array, dim: number) {
  const out = linear(activation({ input: cond.slice(), kind: ACTIVATION.silu }), weight, 1, cond.length, 3 * dim, bias);
  return { shift: out.subarray(0, dim), scale: out.subarray(dim, 2 * dim), gate: out.subarray(2 * dim, 3 * dim) };
}

/**
 * Predict the total latent-frame count for one text state.
 *
 * `speakerVec` is the speaker state's **first** token — the masked-mean summary
 * `prependMeanToken` put there — not a pooling of its own. `captionVec` is a
 * masked mean over the caption, or `null_caption` when there is none.
 */
export function predictDuration({
  weights,
  textState,
  textKeep,
  speakerVec,
  captionVec,
}: {
  weights: ModelWeights;
  /** `[tokens, textDim]`. */
  textState: Float32Array;
  textKeep: boolean[];
  /** `[speakerDim]`, or null to use `null_speaker`. */
  speakerVec: Float32Array | null;
  /** `[captionDim]`, or null to use `null_caption`. */
  captionVec: Float32Array | null;
}): number {
  const w = weights.duration;
  const { normEps, config } = weights;
  const tokens = textKeep.length;
  const dim = w.dim;

  const speaker = speakerVec ?? w.nullSpeaker;
  const caption = captionVec ?? w.nullCaption;

  let h = linear(textState, w.inProj, tokens, config.text_dim, dim, w.inProjBias);
  for (const block of w.blocks) {
    const normed = rmsnorm({ input: h, weight: block.norm, N: tokens, D: dim, eps: normEps });
    const speakerMod = modulate(speaker, block.modulation, block.modulationBias, dim);
    const captionMod = modulate(caption, block.captionModulation, block.captionModulationBias, dim);

    for (let token = 0; token < tokens; token += 1) {
      const at = token * dim;
      for (let d = 0; d < dim; d += 1) {
        // The two modulations add before they are applied.
        const scale = speakerMod.scale[d]! + captionMod.scale[d]!;
        const shift = speakerMod.shift[d]! + captionMod.shift[d]!;
        normed[at + d] = normed[at + d]! * (1 + scale) + shift;
      }
    }

    const mlp = swiglu(normed, block.w1, block.w2, block.w3, tokens, dim, w.hidden);
    for (let token = 0; token < tokens; token += 1) {
      const at = token * dim;
      for (let d = 0; d < dim; d += 1) {
        h[at + d]! += Math.tanh(speakerMod.gate[d]! + captionMod.gate[d]!) * mlp[at + d]!;
      }
    }
  }

  const normed = rmsnorm({ input: h, weight: w.outNorm, N: tokens, D: dim, eps: normEps });
  const logits = linear(normed, w.outProj, tokens, dim, 1, w.outProjBias);

  let total = 0;
  for (let token = 0; token < tokens; token += 1) {
    if (!textKeep[token]) continue;
    // softplus, guarded the way torch guards it: for large x, `log1p(exp(x))`
    // overflows and the answer is x.
    const value = logits[token]!;
    total += value > 20 ? value : Math.log1p(Math.exp(value));
  }
  return Math.log1p(Math.max(total, 0));
}

/** How many latent frames the prediction means. */
export function framesFrom(logDuration: number): number {
  return Math.expm1(logDuration);
}

export function loadDurationWeights(
  raw: (name: string) => Float32Array,
  shapeOf: (name: string) => number[],
  textDim: number,
): DurationWeights {
  const at = (name: string) => `duration_predictor.${name}`;
  const dim = shapeOf(at("token_input_proj.bias"))[0]!;
  const hidden = shapeOf(at("token_blocks.0.mlp.w1.weight"))[0]!;
  const speakerDim = shapeOf(at("null_speaker"))[0]!;
  const captionDim = shapeOf(at("null_caption"))[0]!;

  const blocks: DurationBlockWeights[] = [];
  for (let index = 0; ; index += 1) {
    const block = at(`token_blocks.${index}`);
    try {
      shapeOf(`${block}.norm.weight`);
    } catch {
      break;
    }
    blocks.push({
      norm: raw(`${block}.norm.weight`),
      w1: transpose(raw(`${block}.mlp.w1.weight`), hidden, dim),
      w2: transpose(raw(`${block}.mlp.w2.weight`), dim, hidden),
      w3: transpose(raw(`${block}.mlp.w3.weight`), hidden, dim),
      modulation: transpose(raw(`${block}.modulation.weight`), 3 * dim, speakerDim),
      modulationBias: raw(`${block}.modulation.bias`),
      captionModulation: transpose(raw(`${block}.caption_modulation.weight`), 3 * dim, captionDim),
      captionModulationBias: raw(`${block}.caption_modulation.bias`),
    });
  }

  return {
    dim,
    hidden,
    inProj: transpose(raw(at("token_input_proj.weight")), dim, textDim),
    inProjBias: raw(at("token_input_proj.bias")),
    blocks,
    outNorm: raw(at("token_out_norm.weight")),
    outProj: transpose(raw(at("token_out_proj.weight")), 1, dim),
    outProjBias: raw(at("token_out_proj.bias")),
    nullSpeaker: raw(at("null_speaker")),
    nullCaption: raw(at("null_caption")),
  };
}
