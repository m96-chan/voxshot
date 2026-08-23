import { readFileSync } from "node:fs";
import { join } from "node:path";

import type { TextBlockWeights } from "./blocks.js";
import type { DitBlockWeights, DitShape } from "./dit.js";
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
  caption_dim: number | null;
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

export interface DitWeights {
  shape: DitShape;
  blocks: DitBlockWeights[];
  /** `out_norm` is an RMSNorm; `out_proj` takes model_dim back to the latent. */
  outNorm: Float32Array;
  outProjWeight: Float32Array;
  outProjBias: Float32Array;
  inProjWeight: Float32Array;
  inProjBias: Float32Array;
}

export interface ModelWeights {
  config: ModelConfig;
  normEps: number;
  speaker: SpeakerWeights;
  dit: DitWeights;
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

  const dim = config.model_dim;
  const shape: DitShape = {
    dim,
    heads: config.num_heads,
    // `int(model_dim * mlp_ratio)` — truncation. 1280 * 2.875 is exact at 3680.
    mlpHidden: Math.trunc(dim * config.mlp_ratio),
    rank: Math.max(1, Math.min(config.adaln_rank, dim)),
    eps: config.norm_eps,
  };

  const adaLn = (at: string) => ({
    shiftDown: transposeOf(raw(`${at}.shift_down.weight`), shape.rank, dim),
    scaleDown: transposeOf(raw(`${at}.scale_down.weight`), shape.rank, dim),
    gateDown: transposeOf(raw(`${at}.gate_down.weight`), shape.rank, dim),
    shiftUp: transposeOf(raw(`${at}.shift_up.weight`), dim, shape.rank),
    scaleUp: transposeOf(raw(`${at}.scale_up.weight`), dim, shape.rank),
    gateUp: transposeOf(raw(`${at}.gate_up.weight`), dim, shape.rank),
    shiftBias: raw(`${at}.shift_up.bias`),
    scaleBias: raw(`${at}.scale_up.bias`),
    gateBias: raw(`${at}.gate_up.bias`),
  });

  const blocks: DitBlockWeights[] = [];
  for (let index = 0; index < config.num_layers; index += 1) {
    const at = `blocks.${index}`;
    // Insertion order is the reference's concat order in `JointAttention`:
    // text, then speaker, then caption. A Map would say so more loudly, but the
    // key order of an object literal is specified and this is read in one place.
    const contexts: DitBlockWeights["contexts"] = {
      text: {
        wk: transposeOf(raw(`${at}.attention.wk_text.weight`), dim, config.text_dim),
        wv: transposeOf(raw(`${at}.attention.wv_text.weight`), dim, config.text_dim),
        dim: config.text_dim,
      },
    };
    if (config.use_speaker_condition) {
      contexts.speaker = {
        wk: transposeOf(raw(`${at}.attention.wk_speaker.weight`), dim, config.speaker_dim),
        wv: transposeOf(raw(`${at}.attention.wv_speaker.weight`), dim, config.speaker_dim),
        dim: config.speaker_dim,
      };
    }
    if (config.use_caption_condition) {
      const captionDim = config.caption_dim ?? config.text_dim;
      contexts.caption = {
        wk: transposeOf(raw(`${at}.attention.wk_caption.weight`), dim, captionDim),
        wv: transposeOf(raw(`${at}.attention.wv_caption.weight`), dim, captionDim),
        dim: captionDim,
      };
    }
    blocks.push({
      wq: transposeOf(raw(`${at}.attention.wq.weight`), dim, dim),
      wk: transposeOf(raw(`${at}.attention.wk.weight`), dim, dim),
      wv: transposeOf(raw(`${at}.attention.wv.weight`), dim, dim),
      gate: transposeOf(raw(`${at}.attention.gate.weight`), dim, dim),
      wo: transposeOf(raw(`${at}.attention.wo.weight`), dim, dim),
      qNorm: raw(`${at}.attention.q_norm.weight`),
      kNorm: raw(`${at}.attention.k_norm.weight`),
      contexts,
      w1: transposeOf(raw(`${at}.mlp.w1.weight`), shape.mlpHidden, dim),
      w2: transposeOf(raw(`${at}.mlp.w2.weight`), dim, shape.mlpHidden),
      w3: transposeOf(raw(`${at}.mlp.w3.weight`), shape.mlpHidden, dim),
      attentionAdaLn: adaLn(`${at}.attention_adaln`),
      mlpAdaLn: adaLn(`${at}.mlp_adaln`),
    });
  }

  const dit: DitWeights = {
    shape,
    blocks,
    outNorm: raw("out_norm.weight"),
    outProjWeight: transposeOf(raw("out_proj.weight"), config.latent_dim, dim),
    outProjBias: raw("out_proj.bias"),
    inProjWeight: transposeOf(raw("in_proj.weight"), dim, config.latent_dim),
    inProjBias: raw("in_proj.bias"),
  };

  return { config, normEps: config.norm_eps, speaker, dit, raw };
}

function transposeOf(source: Float32Array, out: number, inn: number): Float32Array {
  const result = new Float32Array(source.length);
  for (let o = 0; o < out; o += 1) {
    const from = o * inn;
    for (let i = 0; i < inn; i += 1) result[i * out + o] = source[from + i]!;
  }
  return result;
}
