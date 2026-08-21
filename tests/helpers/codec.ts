import type { DecoderConfig } from "../../src/engine/miotts/codec/decoder.js";
import { buildSafetensors, pseudoRandom, type SyntheticTensor } from "./safetensors.js";

/**
 * A MioCodec decoder checkpoint small enough for CI.
 *
 * The real one is 523 MB and not in git, and the numbers it produces are
 * `npm run test:models`' business. What a CI test can still check is that the
 * graph is wired: that every stage feeds the next at the shape it expects,
 * that the lengths follow the 25 Hz token rate, and that a run produces finite
 * audio. Driving a two-dimension codebook and an 8-channel model through
 * `cpuBackend` does exactly that, in milliseconds and with no download.
 *
 * The dimensions below are deliberately small AND deliberately unequal where
 * the real model's are unequal, so a transposed or swapped axis fails rather
 * than happening to line up.
 */

/** A decoder config with the real one's structure and none of its size. */
export const TINY_DECODER: DecoderConfig = {
  nFft: 16,
  hopLength: 4,
  sampleRate: 24_000,
  waveResnetNumBlocks: 1,
  waveResnetKernelSize: 3,
  waveResnetNumGroups: 4,
  fsqLevels: [4, 4],
  // `outputDim` has to match the decoder's `dim`: the prenet's output is what
  // the resnet stack and the decoder transformer both run on, and the real
  // config has them equal (512) for the same reason.
  prenet: { dim: 8, layers: 1, heads: 2, windowSize: 5, ropeTheta: 10_000, outputDim: 8 },
  // 128 like the real model, not scaled down with everything else: the
  // speaker embedding's width is a contract `assertUsableVoice` enforces, so
  // a tiny one here would make this checkpoint unusable through the engine.
  decoder: { dim: 8, layers: 1, heads: 2, windowSize: 5, ropeTheta: 10_000, adaLnConditionDim: 128 },
};

/** Width of the SwiGLU hidden layer; fixed by the w1/w2/w3 shapes alone. */
const FFN_HIDDEN = 16;

/** `wave_conv_upsample`'s kernel. With stride 2 it doubles the token count exactly. */
const UPSAMPLE_KERNEL = 2;

let seedCounter = 0;

/** `nn.Linear`-shaped weight: `[out, in]`. */
function linear(out: number, inDim: number): SyntheticTensor {
  seedCounter += 1;
  // Scaled down by fan-in: an untrained-looking model whose activations do not
  // explode through eight stages, so a NaN in the output means a real defect.
  return { shape: [out, inDim], data: pseudoRandom(out * inDim, seedCounter, 1 / Math.sqrt(inDim)) };
}

function vector(n: number, fill?: number): SyntheticTensor {
  seedCounter += 1;
  const data =
    fill === undefined ? pseudoRandom(n, seedCounter, 0.1) : new Float32Array(n).fill(fill);
  return { shape: [n], data };
}

function conv(cOut: number, cIn: number, k: number): SyntheticTensor {
  seedCounter += 1;
  return { shape: [cOut, cIn, k], data: pseudoRandom(cOut * cIn * k, seedCounter, 1 / Math.sqrt(cIn * k)) };
}

/** One transformer block's tensors, under `prefix.layers.i`. */
function transformerLayer(
  into: Record<string, SyntheticTensor>,
  prefix: string,
  dim: number,
  conditionDim: number | undefined,
  zero = false,
): void {
  const projection = (out: number, inDim: number): SyntheticTensor =>
    zero ? { shape: [out, inDim], data: new Float32Array(out * inDim) } : linear(out, inDim);
  for (const norm of ["attention_norm", "ffn_norm"] as const) {
    if (conditionDim === undefined) {
      into[`${prefix}.${norm}.weight`] = vector(dim, 1);
      into[`${prefix}.${norm}.bias`] = vector(dim, 0);
    } else {
      // AdaLN-Zero: silu(condition) -> one Linear producing [shift, scale, gate].
      into[`${prefix}.${norm}.condition_proj.1.weight`] = projection(3 * dim, conditionDim);
      into[`${prefix}.${norm}.condition_proj.1.bias`] = vector(3 * dim, 0);
    }
  }
  for (const name of ["wq", "wk", "wv", "wo"] as const) {
    into[`${prefix}.attention.${name}.weight`] = projection(dim, dim);
  }
  into[`${prefix}.feed_forward.w1.weight`] = projection(FFN_HIDDEN, dim);
  into[`${prefix}.feed_forward.w3.weight`] = projection(FFN_HIDDEN, dim);
  into[`${prefix}.feed_forward.w2.weight`] = projection(dim, FFN_HIDDEN);
}

/** A whole transformer stack, including its final norm and optional projection. */
function transformerStack(
  into: Record<string, SyntheticTensor>,
  prefix: string,
  config: { dim: number; layers: number; outputDim?: number; adaLnConditionDim?: number },
  zero = false,
): void {
  for (let layer = 0; layer < config.layers; layer += 1) {
    transformerLayer(into, `${prefix}.layers.${layer}`, config.dim, config.adaLnConditionDim, zero);
  }
  if (config.adaLnConditionDim === undefined) {
    into[`${prefix}.norm.weight`] = vector(config.dim, 1);
    into[`${prefix}.norm.bias`] = vector(config.dim, 0);
  } else {
    // The final norm has no gate, so its projection is [shift, scale] only.
    into[`${prefix}.norm.condition_proj.1.weight`] = zero
      ? { shape: [2 * config.dim, config.adaLnConditionDim], data: new Float32Array(2 * config.dim * config.adaLnConditionDim) }
      : linear(2 * config.dim, config.adaLnConditionDim);
    into[`${prefix}.norm.condition_proj.1.bias`] = vector(2 * config.dim, 0);
  }
  if (config.outputDim !== undefined) {
    into[`${prefix}.output_proj.weight`] = linear(config.outputDim, config.dim);
    into[`${prefix}.output_proj.bias`] = vector(config.outputDim, 0);
  }
}

/** A ResNet stack's two convolutions and two group norms per block. */
function resnetStack(
  into: Record<string, SyntheticTensor>,
  prefix: string,
  channels: number,
  config: DecoderConfig,
): void {
  for (let block = 0; block < config.waveResnetNumBlocks; block += 1) {
    for (const [norm, convolution] of [
      ["norm1", "conv1"],
      ["norm2", "conv2"],
    ] as const) {
      into[`${prefix}.blocks.${block}.${norm}.weight`] = vector(channels, 1);
      into[`${prefix}.blocks.${block}.${norm}.bias`] = vector(channels, 0);
      into[`${prefix}.blocks.${block}.${convolution}.weight`] = conv(
        channels,
        channels,
        config.waveResnetKernelSize,
      );
      into[`${prefix}.blocks.${block}.${convolution}.bias`] = vector(channels, 0);
    }
  }
}

export interface TinyCheckpointOptions {
  /**
   * Make `wave_conv_upsample` a per-channel identity: each channel's value is
   * repeated into both of its upsampled positions, and channels do not mix.
   *
   * That turns an otherwise opaque stage into one whose output can be stated
   * in terms of its input, which is what lets a test observe the transpose
   * feeding it. With random weights every axis ordering produces numbers of
   * the same shape, and a swap is invisible.
   */
  identityUpsample?: boolean;
  /**
   * Zero every attention and feed-forward weight, and every AdaLN condition
   * projection.
   *
   * Both residual branches then contribute nothing, so a transformer stack
   * reduces to its final norm — and AdaLN-Zero with a zero projection reduces
   * to a plain LayerNorm, which is the property that makes an untrained model
   * the identity rather than a model that scales everything to zero.
   */
  zeroResidualBranches?: boolean;
}

/** Every tensor `decode()` reads, at {@link TINY_DECODER}'s dimensions. */
export function buildTinyDecoderCheckpoint(
  config: DecoderConfig = TINY_DECODER,
  options: TinyCheckpointOptions = {},
): ArrayBuffer {
  seedCounter = 0;
  const tensors: Record<string, SyntheticTensor> = {};
  const prenetOut = config.prenet.outputDim ?? config.prenet.dim;
  const dim = config.decoder.dim;
  const bins = config.nFft / 2 + 1;

  tensors["local_quantizer.proj_out.weight"] = linear(config.prenet.dim, config.fsqLevels.length);
  tensors["local_quantizer.proj_out.bias"] = vector(config.prenet.dim, 0);

  transformerStack(tensors, "wave_prenet", config.prenet, options.zeroResidualBranches);

  // ConvTranspose1d stores `[in, out/groups, K]`, unlike Conv1d's `[out, in, K]`.
  seedCounter += 1;
  const upsample = new Float32Array(prenetOut * prenetOut * UPSAMPLE_KERNEL);
  if (options.identityUpsample) {
    for (let channel = 0; channel < prenetOut; channel += 1) {
      for (let tap = 0; tap < UPSAMPLE_KERNEL; tap += 1) {
        upsample[(channel * prenetOut + channel) * UPSAMPLE_KERNEL + tap] = 1;
      }
    }
  } else {
    upsample.set(
      pseudoRandom(upsample.length, seedCounter, 1 / Math.sqrt(prenetOut)),
    );
  }
  tensors["wave_conv_upsample.weight"] = {
    shape: [prenetOut, prenetOut, UPSAMPLE_KERNEL],
    data: upsample,
  };
  tensors["wave_conv_upsample.bias"] = vector(prenetOut, 0);

  resnetStack(tensors, "wave_prior_net", dim, config);
  transformerStack(tensors, "wave_decoder", config.decoder, options.zeroResidualBranches);
  resnetStack(tensors, "wave_post_net", dim, config);

  // The head emits log-magnitude and phase side by side, hence 2 * bins.
  tensors["istft_head.out.weight"] = linear(2 * bins, dim);
  tensors["istft_head.out.bias"] = vector(2 * bins, 0);

  return buildSafetensors(tensors);
}

/** A speaker embedding of the width the decoder's AdaLN conditioning expects. */
export function tinyGlobalEmbedding(config: DecoderConfig = TINY_DECODER): Float32Array {
  return pseudoRandom(config.decoder.adaLnConditionDim ?? 0, 4242, 0.5);
}

/** Codebook indices inside `fsqLevels`' range. */
export function tinyTokens(count: number, config: DecoderConfig = TINY_DECODER): Float32Array {
  const size = config.fsqLevels.reduce((a, b) => a * b, 1);
  return Float32Array.from({ length: count }, (_, i) => (i * 7 + 3) % size);
}
