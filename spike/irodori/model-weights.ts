import { readFileSync } from "node:fs";
import { join } from "node:path";

import type { TextBlockWeights } from "./blocks.js";
import { speakerBlocks } from "./speaker-encoder.js";

/**
 * The checkpoint's non-backbone weights, as `dump_model.py` wrote them.
 *
 * Sizes come from the checkpoint's own `config_json`, which travels in the
 * safetensors metadata. That matters more here than it did for ModernBERT: the
 * reference's `ModelConfig` has branches for a projector that is `linear` or
 * `residual_mlp`, six duration architectures, optional caption and speaker
 * conditioning. Reading the config says which one this checkpoint is; reading
 * the code only says which ones exist.
 */

export interface ModelConfig {
  latent_dim: number;
  model_dim: number;
  num_layers: number;
  num_heads: number;
  mlp_ratio: number;
  text_dim: number;
  speaker_dim: number;
  speaker_layers: number;
  speaker_heads: number;
  speaker_patch_size: number;
  speaker_mlp_ratio: number;
  timestep_embed_dim: number;
  adaln_rank: number;
  norm_eps: number;
  pretrained_projector_type: string;
  pretrained_projector_hidden_ratio: number;
  duration_architecture: string;
  use_caption_condition: boolean;
  use_speaker_condition: boolean;
  max_text_len: number;
}

export interface SpeakerWeights {
  dim: number;
  heads: number;
  layers: number;
  patch: number;
  patchedLatentDim: number;
  mlpHidden: number;
  inProjWeight: Float32Array;
  inProjBias: Float32Array;
  blocks: TextBlockWeights[];
  /** `speaker_norm`, applied to the encoder's output one level up. */
  outNorm: Float32Array;
}

export interface ModelWeights {
  config: ModelConfig;
  normEps: number;
  speaker: SpeakerWeights;
  /** Every tensor, for the parts not yet given a shaped view. */
  raw: (name: string) => Float32Array;
}

interface Index {
  config: ModelConfig;
  tensors: Record<string, { shape: number[]; bytes: number }>;
}

export function loadModelWeights(dir: string): ModelWeights {
  const index = JSON.parse(readFileSync(join(dir, "index.json"), "utf8")) as Index;
  const cache = new Map<string, Float32Array>();

  const raw = (name: string): Float32Array => {
    const found = cache.get(name);
    if (found) return found;
    const entry = index.tensors[name];
    if (!entry) throw new Error(`${name} is not in golden/model/index.json`);
    const bytes = readFileSync(join(dir, `${name}.f32`));
    if (bytes.byteLength !== entry.bytes) {
      throw new Error(`${name}.f32 is ${bytes.byteLength} bytes, index says ${entry.bytes}`);
    }
    // `Buffer` views a pooled `ArrayBuffer` at an arbitrary offset; `Float32Array`
    // needs 4-byte alignment, so the range is copied out.
    const copy = new Uint8Array(bytes.byteLength);
    copy.set(bytes);
    const array = new Float32Array(copy.buffer);
    cache.set(name, array);
    return array;
  };

  const config = index.config;
  const speakerDim = config.speaker_dim;
  const speaker: SpeakerWeights = {
    dim: speakerDim,
    heads: config.speaker_heads,
    layers: config.speaker_layers,
    patch: config.speaker_patch_size,
    patchedLatentDim: config.latent_dim * config.speaker_patch_size,
    // `int(dim * ratio)` in the reference — truncation, not rounding. 768 *
    // 2.6 is 1996.79..., and the checkpoint's w1 is [1996, 768].
    mlpHidden: Math.trunc(speakerDim * config.speaker_mlp_ratio),
    inProjWeight: transposeOf(raw("speaker_encoder.in_proj.weight"), speakerDim, config.latent_dim * config.speaker_patch_size),
    inProjBias: raw("speaker_encoder.in_proj.bias"),
    blocks: [],
    outNorm: raw("speaker_norm.weight"),
  };
  speaker.blocks = speakerBlocks(raw, speaker.layers, speaker.dim, speaker.mlpHidden);

  return { config, normEps: config.norm_eps, speaker, raw };
}

function transposeOf(source: Float32Array, out: number, inn: number): Float32Array {
  const result = new Float32Array(source.length);
  for (let o = 0; o < out; o += 1) {
    const from = o * inn;
    for (let i = 0; i < inn; i += 1) result[i * out + o] = source[from + i]!;
  }
  return result;
}
