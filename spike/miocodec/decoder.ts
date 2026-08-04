import { activation, ACTIVATION } from "web-xpu-ops/ops/activation";
import { attention } from "web-xpu-ops/ops/attention";
import { conv1d } from "web-xpu-ops/ops/conv";
import { convTranspose1d } from "web-xpu-ops/ops/conv_transpose";
import { groupNorm } from "web-xpu-ops/ops/group_norm";
import { layernorm } from "web-xpu-ops/ops/layernorm";
import { matmul } from "web-xpu-ops/ops/matmul";
import { rope } from "web-xpu-ops/ops/rope";
import { hannWindow, istft } from "web-xpu-ops/ops/stft";
import type { Safetensors } from "./safetensors.js";

/**
 * MioCodec's decoder — content tokens and a speaker embedding to a waveform —
 * expressed entirely in web-xpu-ops.
 *
 * A port of `miocodec.model.MioCodecModel.forward_wave` for the 24 kHz rung.
 * The graph, and where each piece comes from:
 *
 *   tokens ─▶ FSQ decode          arithmetic + `matmul`
 *          ─▶ wave_prenet         Transformer 768, 6L x 12H, window 65 → 512
 *          ─▶ wave_conv_upsample  `convTranspose1d` k=2 s=2   (25 Hz → 50 Hz)
 *          ─▶ interpolate         linear resample to the STFT length
 *          ─▶ wave_prior_net      ResNet x2: `groupNorm` → silu → `conv1d`
 *          ─▶ wave_decoder        Transformer 512, 8L x 8H, window 65, AdaLN-Zero
 *          ─▶ wave_post_net       ResNet x2
 *          ─▶ istft_head          `matmul` [512 → 1922] ─▶ `istft` "same"
 *          ─▶ waveform, 24 kHz
 *
 * This runs on the **reference** implementations, which are deliberately the
 * slowest expression of each op — it exists to be checked against the golden
 * stage by stage, not to be fast. `gpu.ts` is the same graph on the kernels.
 */

export interface Tensor {
  data: Float32Array;
  shape: number[];
}

/**
 * The three ops that carry the arithmetic, behind a seam.
 *
 * Everything else in this graph is linear in its data and costs nothing beside
 * these: `matmul`, `conv1d` and the inverse transform are where the work is. So
 * they are the ones a backend replaces, and the rest of the file is the same
 * code whichever backend is in use — which is also what lets both be compared
 * against the same golden rather than against each other.
 *
 * Async because a GPU readback is, and the alternative — two copies of the
 * graph, one sync and one not — is two places for a port to be subtly wrong.
 */
export interface Backend {
  readonly name: string;
  matmul(a: Float32Array, b: Float32Array, M: number, N: number, K: number): Promise<Float32Array>;
  conv1d(
    input: Float32Array,
    weight: Float32Array,
    bias: Float32Array | null,
    Cin: number,
    Cout: number,
    L: number,
    K: number,
    padding: number,
  ): Promise<Float32Array>;
  /** Always `"same"` here; the backend resolves what that means for its own API. */
  istft(
    real: Float32Array,
    imag: Float32Array,
    window: Float32Array,
    frames: number,
    nFft: number,
    hop: number,
  ): Promise<Float32Array>;
}

/** The reference implementations — the definition of correct, and the slowest. */
export const cpuBackend: Backend = {
  name: "reference (CPU)",
  async matmul(a, b, M, N, K) {
    return matmul({ a, b, M, N, K });
  },
  async conv1d(input, weight, bias, Cin, Cout, L, K, padding) {
    return conv1d({ input, weight, bias: bias ?? undefined, N: 1, Cin, Cout, L, K, padding });
  },
  async istft(real, imag, window, frames, nFft, hop) {
    return istft({ real, imag, frames, nFft, hop, window, padding: "same" });
  },
};

const NORM_EPS = 1e-5;
/** `ResNetBlock` passes this explicitly; it is **not** torch's 1e-5 default. */
const GROUP_NORM_EPS = 1e-6;

export interface DecoderConfig {
  nFft: number;
  hopLength: number;
  sampleRate: number;
  waveResnetNumBlocks: number;
  waveResnetKernelSize: number;
  waveResnetNumGroups: number;
  fsqLevels: number[];
  prenet: TransformerConfig;
  decoder: TransformerConfig;
}

export interface TransformerConfig {
  dim: number;
  layers: number;
  heads: number;
  windowSize: number;
  ropeTheta: number;
  outputDim?: number;
  adaLnConditionDim?: number;
}

/** The 24 kHz rung, from its `config.yaml`. */
export const MIOCODEC_24K: DecoderConfig = {
  nFft: 1920,
  hopLength: 480,
  sampleRate: 24000,
  waveResnetNumBlocks: 2,
  waveResnetKernelSize: 3,
  waveResnetNumGroups: 32,
  fsqLevels: [8, 8, 8, 5, 5],
  prenet: { dim: 768, layers: 6, heads: 12, windowSize: 65, ropeTheta: 10000, outputDim: 512 },
  decoder: {
    dim: 512,
    layers: 8,
    heads: 8,
    windowSize: 65,
    ropeTheta: 10000,
    adaLnConditionDim: 128,
  },
};

/* -------------------------------------------------------------------------- *
 * Small helpers
 * -------------------------------------------------------------------------- */

/**
 * `y = x @ W^T + b`, the shape `nn.Linear` stores its weight in.
 *
 * `matmul` is `torch.mm` — `[M, K] x [K, N]` — and a Linear's weight is
 * `[out, in]`, so one of the two has to be turned. Transposing the weight here,
 * every call, is the slow and obvious thing; the GPU path hoists it to load
 * time. Correctness first (rule 8).
 */
const transposed = new WeakMap<Float32Array, Float32Array>();

async function linear(
  x: Tensor,
  weight: Tensor,
  bias: Tensor | null,
  backend: Backend,
): Promise<Tensor> {
  const [outFeatures, inFeatures] = weight.shape as [number, number];
  const rows = x.data.length / inFeatures;
  // Turned once per weight, not once per call. `matmul` is `torch.mm` and a
  // Linear's weight is `[out, in]`, so one of the two has to be transposed —
  // but the weight does not change between calls, and the decoder makes
  // hundreds of them. Keyed on the array's identity, which `Weights` now keeps
  // stable; a `WeakMap` so nothing is held alive once the checkpoint is
  // dropped.
  let b = transposed.get(weight.data);
  if (!b) {
    b = new Float32Array(inFeatures * outFeatures);
    for (let o = 0; o < outFeatures; o += 1) {
      for (let i = 0; i < inFeatures; i += 1) {
        b[i * outFeatures + o] = weight.data[o * inFeatures + i]!;
      }
    }
    transposed.set(weight.data, b);
  }
  const out = await backend.matmul(x.data, b, rows, outFeatures, inFeatures);
  if (bias) {
    for (let r = 0; r < rows; r += 1) {
      for (let o = 0; o < outFeatures; o += 1) {
        out[r * outFeatures + o] = out[r * outFeatures + o]! + bias.data[o]!;
      }
    }
  }
  return { data: out, shape: [...x.shape.slice(0, -1), outFeatures] };
}

function silu(x: Float32Array): Float32Array {
  return activation({ input: x, kind: ACTIVATION.silu });
}

function addInPlace(a: Float32Array, b: Float32Array): Float32Array {
  for (let i = 0; i < a.length; i += 1) a[i] = a[i]! + b[i]!;
  return a;
}

/** `[L, C]` to `[C, L]`, the axis swap that separates conv stages from attention ones. */
function transpose2d(data: Float32Array, rows: number, cols: number): Float32Array {
  const out = new Float32Array(data.length);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) out[c * rows + r] = data[r * cols + c]!;
  }
  return out;
}

/**
 * `F.interpolate(mode="linear", align_corners=False)` over `[C, L]`.
 *
 * The default `align_corners=False` is what torch uses when the argument is
 * omitted, and it is the one that matters here: it samples at
 * `(i + 0.5) · scale − 0.5` and clamps, where `align_corners=True` would spread
 * the endpoints to the corners. The two agree only when the length is
 * unchanged — which is exactly the case the golden's `aligned` run exercises,
 * and why there is a `resampled` case that does not.
 */
function interpolateLinear(input: Tensor, channels: number, targetLength: number): Tensor {
  const sourceLength = input.data.length / channels;
  if (sourceLength === targetLength) return input;
  const out = new Float32Array(channels * targetLength);
  const scale = sourceLength / targetLength;
  for (let t = 0; t < targetLength; t += 1) {
    const position = Math.max(0, (t + 0.5) * scale - 0.5);
    const left = Math.min(Math.floor(position), sourceLength - 1);
    const right = Math.min(left + 1, sourceLength - 1);
    const weight = position - left;
    for (let c = 0; c < channels; c += 1) {
      const a = input.data[c * sourceLength + left]!;
      const b = input.data[c * sourceLength + right]!;
      out[c * targetLength + t] = a + (b - a) * weight;
    }
  }
  return { data: out, shape: [channels, targetLength] };
}

/**
 * The local-attention mask, as an additive bias.
 *
 * `Attention.create_mask` builds `triu(diagonal=-w) & tril(diagonal=+w)` with
 * `w = window_size // 2`, so position `i` sees `j` when `|i - j| <= w`. A
 * boolean there, `-Infinity` here — `ops/attention` takes torch's *float*
 * `attn_mask`, which composes by addition and is the form `alibi` already
 * produces.
 *
 * Shape `[1, 1, L]` per `maskShape`'s contract: batch and head broadcast, the
 * rows are real, and the last axis is always the keys.
 */
function windowMask(length: number, windowSize: number): Float32Array {
  const perSide = Math.floor(windowSize / 2);
  const mask = new Float32Array(length * length);
  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      mask[i * length + j] = Math.abs(i - j) <= perSide ? 0 : -Infinity;
    }
  }
  return mask;
}

/* -------------------------------------------------------------------------- *
 * Weights
 * -------------------------------------------------------------------------- */

export class Weights {
  // `Safetensors.tensor` copies out of the checkpoint on every call, by design
  // — a view would pin the whole file. That makes it the wrong thing to call
  // per layer per forward, which is what an uncached `get` does: the decoder
  // asks for the same two hundred tensors on every decode. Memoised by name, so
  // each is copied once and every later reader gets the same array — which is
  // also what makes the transpose cache below able to key on identity.
  private readonly cache = new Map<string, Tensor>();

  constructor(private readonly file: Safetensors) {}

  get(name: string): Tensor {
    let tensor = this.cache.get(name);
    if (!tensor) {
      const view = this.file.tensor(name);
      tensor = { data: view.data, shape: [...view.shape] };
      this.cache.set(name, tensor);
    }
    return tensor;
  }

  maybe(name: string): Tensor | null {
    return this.file.has(name) ? this.get(name) : null;
  }
}

/* -------------------------------------------------------------------------- *
 * Stages
 * -------------------------------------------------------------------------- */

/**
 * FSQ decode: a flat codebook index to a continuous embedding.
 *
 * `indices_to_codes` is `(index // basis) % levels` per dimension, then
 * `(code - half_width) / half_width`, then the `proj_out` Linear. The basis is
 * `cumprod([1] + levels[:-1])`, so the first dimension varies fastest.
 */
export async function fsqDecode(
  tokens: Float32Array,
  levels: number[],
  weights: Weights,
  backend: Backend,
): Promise<Tensor> {
  const basis: number[] = [];
  let running = 1;
  for (let i = 0; i < levels.length; i += 1) {
    basis.push(running);
    running *= levels[i]!;
  }
  const codes = new Float32Array(tokens.length * levels.length);
  for (let t = 0; t < tokens.length; t += 1) {
    for (let d = 0; d < levels.length; d += 1) {
      const halfWidth = Math.floor(levels[d]! / 2);
      const code = Math.floor(tokens[t]! / basis[d]!) % levels[d]!;
      codes[t * levels.length + d] = (code - halfWidth) / halfWidth;
    }
  }
  return await linear(
    { data: codes, shape: [tokens.length, levels.length] },
    weights.get("local_quantizer.proj_out.weight"),
    weights.maybe("local_quantizer.proj_out.bias"),
    backend,
  );
}

/** `LayerNorm(dim)` with learned scale and shift, over the last axis. */
function layerNorm(x: Tensor, weight: Tensor, bias: Tensor, dim: number): Tensor {
  return {
    data: layernorm({
      input: x.data,
      weight: weight.data,
      bias: bias.data,
      N: x.data.length / dim,
      D: dim,
      eps: NORM_EPS,
    }),
    shape: [...x.shape],
  };
}

/**
 * AdaLN-Zero: LayerNorm with **no** learnable affine, modulated by a projection
 * of the conditioning vector.
 *
 * `silu` then one Linear producing `[shift, scale, gate]` (or `[shift, scale]`
 * for the final norm, which has no gate), then `x_norm * (1 + scale) + shift`.
 * The `1 +` matters: the projection is zero-initialised so an untrained model
 * is the identity, and dropping it would scale everything to zero instead.
 */
async function adaLnZero(
  x: Tensor,
  condition: Float32Array,
  dim: number,
  prefix: string,
  weights: Weights,
  withGate: boolean,
  backend: Backend,
): Promise<{ modulated: Tensor; gate: Float32Array | null }> {
  const rows = x.data.length / dim;
  const normed = layernorm({
    input: x.data,
    // elementwise_affine=False upstream, so the identity affine here rather
    // than weights that do not exist in the checkpoint.
    weight: new Float32Array(dim).fill(1),
    bias: new Float32Array(dim),
    N: rows,
    D: dim,
    eps: NORM_EPS,
  });

  const projected = await linear(
    { data: silu(condition), shape: [1, condition.length] },
    weights.get(`${prefix}.condition_proj.1.weight`),
    weights.maybe(`${prefix}.condition_proj.1.bias`),
    backend,
  );

  const parts = withGate ? 3 : 2;
  const shift = projected.data.subarray(0, dim);
  const scale = projected.data.subarray(dim, 2 * dim);
  const gate = withGate ? projected.data.slice(2 * dim, 3 * dim) : null;
  if (projected.data.length !== parts * dim) {
    throw new Error(`${prefix} projected ${projected.data.length}, expected ${parts * dim}`);
  }

  const out = new Float32Array(normed.length);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < dim; c += 1) {
      const i = r * dim + c;
      out[i] = normed[i]! * (1 + scale[c]!) + shift[c]!;
    }
  }
  return { modulated: { data: out, shape: [...x.shape] }, gate };
}

/** One attention block: qkv, RoPE on both q and k, windowed SDPA, output projection. */
async function selfAttention(
  x: Tensor,
  config: TransformerConfig,
  prefix: string,
  weights: Weights,
  mask: Float32Array,
  length: number,
  backend: Backend,
): Promise<Tensor> {
  const { dim, heads, ropeTheta } = config;
  const headDim = dim / heads;

  const q = await linear(x, weights.get(`${prefix}.wq.weight`), weights.maybe(`${prefix}.wq.bias`), backend);
  const k = await linear(x, weights.get(`${prefix}.wk.weight`), weights.maybe(`${prefix}.wk.bias`), backend);
  const v = await linear(x, weights.get(`${prefix}.wv.weight`), weights.maybe(`${prefix}.wv.bias`), backend);

  // `[L, heads, headDim]` is what `rope` wants and what the qkv projections
  // already produce — `x.view(bsz, seqlen, n_heads, head_dim)` upstream is a
  // reshape, not a permutation.
  const roped = (t: Tensor) => ({
    data: rope({
      input: t.data,
      N: length,
      numHeads: heads,
      headDim,
      posOffset: 0,
      thetaBase: ropeTheta,
    }),
    shape: t.shape,
  });

  // `attention` wants `[B, H, L, D]`; the projections are `[L, H, D]`.
  const toHeadMajor = (t: Tensor) => {
    const out = new Float32Array(t.data.length);
    for (let l = 0; l < length; l += 1) {
      for (let h = 0; h < heads; h += 1) {
        for (let d = 0; d < headDim; d += 1) {
          out[(h * length + l) * headDim + d] = t.data[(l * heads + h) * headDim + d]!;
        }
      }
    }
    return out;
  };

  const { output: scores } = attention({
    q: toHeadMajor(roped(q)),
    k: toHeadMajor(roped(k)),
    v: toHeadMajor(v),
    B: 1,
    H: heads,
    L: length,
    S: length,
    D: headDim,
    Dv: headDim,
    // `Attention.scale` is `head_dim ** -0.5`, which is also this op's default;
    // passed anyway so the two cannot drift apart silently.
    scale: 1 / Math.sqrt(headDim),
    mask,
    maskShape: [1, 1, length],
  });

  // Back to `[L, H * D]`, which is `output.transpose(1, 2).contiguous().view`.
  const merged = new Float32Array(scores.length);
  for (let l = 0; l < length; l += 1) {
    for (let h = 0; h < heads; h += 1) {
      for (let d = 0; d < headDim; d += 1) {
        merged[(l * heads + h) * headDim + d] = scores[(h * length + l) * headDim + d]!;
      }
    }
  }

  return await linear(
    { data: merged, shape: [length, dim] },
    weights.get(`${prefix}.wo.weight`),
    weights.maybe(`${prefix}.wo.bias`),
    backend,
  );
}

/** SwiGLU: `w2(silu(w1(x)) * w3(x))`. */
async function feedForward(
  x: Tensor,
  prefix: string,
  weights: Weights,
  backend: Backend,
): Promise<Tensor> {
  const gate = await linear(x, weights.get(`${prefix}.w1.weight`), null, backend);
  const up = await linear(x, weights.get(`${prefix}.w3.weight`), null, backend);
  const activated = silu(gate.data);
  for (let i = 0; i < activated.length; i += 1) activated[i] = activated[i]! * up.data[i]!;
  return await linear(
    { data: activated, shape: gate.shape },
    weights.get(`${prefix}.w2.weight`),
    null,
    backend,
  );
}

export async function transformer(
  input: Tensor,
  config: TransformerConfig,
  prefix: string,
  weights: Weights,
  condition: Float32Array | null,
  backend: Backend,
): Promise<Tensor> {
  const { dim, layers, windowSize } = config;
  const length = input.data.length / dim;
  const mask = windowMask(length, windowSize);
  const useAdaLn = config.adaLnConditionDim !== undefined;
  if (useAdaLn && !condition) throw new Error(`${prefix} needs a condition`);

  let x: Tensor = { data: Float32Array.from(input.data), shape: [length, dim] };

  for (let layer = 0; layer < layers; layer += 1) {
    const layerPrefix = `${prefix}.layers.${layer}`;

    let normed: Tensor;
    let attnGate: Float32Array | null = null;
    if (useAdaLn) {
      const result = await adaLnZero(
        x,
        condition!,
        dim,
        `${layerPrefix}.attention_norm`,
        weights,
        true,
        backend,
      );
      normed = result.modulated;
      attnGate = result.gate;
    } else {
      normed = layerNorm(
        x,
        weights.get(`${layerPrefix}.attention_norm.weight`),
        weights.get(`${layerPrefix}.attention_norm.bias`),
        dim,
      );
    }

    const attended = await selfAttention(
      normed,
      config,
      `${layerPrefix}.attention`,
      weights,
      mask,
      length,
      backend,
    );
    applyGated(x.data, attended.data, attnGate, dim);

    let ffnNormed: Tensor;
    let ffnGate: Float32Array | null = null;
    if (useAdaLn) {
      const result = await adaLnZero(
        x,
        condition!,
        dim,
        `${layerPrefix}.ffn_norm`,
        weights,
        true,
        backend,
      );
      ffnNormed = result.modulated;
      ffnGate = result.gate;
    } else {
      ffnNormed = layerNorm(
        x,
        weights.get(`${layerPrefix}.ffn_norm.weight`),
        weights.get(`${layerPrefix}.ffn_norm.bias`),
        dim,
      );
    }

    const forwarded = await feedForward(ffnNormed, `${layerPrefix}.feed_forward`, weights, backend);
    applyGated(x.data, forwarded.data, ffnGate, dim);
  }

  // Final norm, then the optional output projection.
  let out: Tensor;
  if (useAdaLn) {
    out = (await adaLnZero(x, condition!, dim, `${prefix}.norm`, weights, false, backend)).modulated;
  } else {
    out = layerNorm(x, weights.get(`${prefix}.norm.weight`), weights.get(`${prefix}.norm.bias`), dim);
  }

  const projWeight = weights.maybe(`${prefix}.output_proj.weight`);
  if (projWeight) {
    out = await linear(out, projWeight, weights.maybe(`${prefix}.output_proj.bias`), backend);
  }
  return out;
}

/** `x += gate * y`, or `x += y` when the block has no gate. */
function applyGated(x: Float32Array, y: Float32Array, gate: Float32Array | null, dim: number): void {
  if (!gate) {
    addInPlace(x, y);
    return;
  }
  const rows = x.length / dim;
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < dim; c += 1) {
      x[r * dim + c] = x[r * dim + c]! + gate[c]! * y[r * dim + c]!;
    }
  }
}

/**
 * `ResNetStack`: blocks of `GroupNorm → SiLU → Conv1d` twice, plus a skip.
 *
 * Channel-major `[C, L]` throughout, which is what both `groupNorm` and `conv1d`
 * assume and what the reference carries between these stages.
 */
export async function resnetStack(
  input: Tensor,
  channels: number,
  length: number,
  config: DecoderConfig,
  prefix: string,
  weights: Weights,
  backend: Backend,
): Promise<Tensor> {
  const kernel = config.waveResnetKernelSize;
  const padding = (kernel - 1) >> 1;
  // Annotated: `Float32Array.from` infers the buffer-typed form, which TS 5.7
  // will not assign a kernel's `ArrayBufferLike` result back into.
  let x: Float32Array = Float32Array.from(input.data);

  for (let block = 0; block < config.waveResnetNumBlocks; block += 1) {
    const blockPrefix = `${prefix}.blocks.${block}`;
    const residual = Float32Array.from(x);

    for (const [normName, convName] of [
      ["norm1", "conv1"],
      ["norm2", "conv2"],
    ] as const) {
      const normed = groupNorm({
        input: x,
        weight: weights.get(`${blockPrefix}.${normName}.weight`).data,
        bias: weights.get(`${blockPrefix}.${normName}.bias`).data,
        N: 1,
        C: channels,
        L: length,
        G: config.waveResnetNumGroups,
        eps: GROUP_NORM_EPS,
      });
      const weight = weights.get(`${blockPrefix}.${convName}.weight`);
      x = await backend.conv1d(
        silu(normed),
        weight.data,
        weights.maybe(`${blockPrefix}.${convName}.bias`)?.data ?? null,
        channels,
        channels,
        length,
        kernel,
        padding,
      );
    }
    addInPlace(x, residual);
  }
  return { data: x, shape: [channels, length] };
}

/* -------------------------------------------------------------------------- *
 * The whole decoder
 * -------------------------------------------------------------------------- */

export interface DecodeResult {
  waveform: Float32Array;
  stages: Record<string, Tensor>;
}

export async function decode(
  tokens: Float32Array,
  globalEmbedding: Float32Array,
  stftLength: number,
  config: DecoderConfig,
  weights: Weights,
  backend: Backend = cpuBackend,
): Promise<DecodeResult> {
  const stages: Record<string, Tensor> = {};

  const contentEmbedding = await fsqDecode(tokens, config.fsqLevels, weights, backend);
  stages.content_embedding = contentEmbedding;

  const prenetOut = await transformer(
    contentEmbedding,
    config.prenet,
    "wave_prenet",
    weights,
    null,
    backend,
  );
  stages.after_prenet = prenetOut;

  // `[L, C]` to `[C, L]`: the conv stages are channel-major, the attention
  // stages are length-major, and upstream transposes between them too.
  const prenetDim = config.prenet.outputDim ?? config.prenet.dim;
  const upsampleWeight = weights.get("wave_conv_upsample.weight");
  const upsampled = convTranspose1d({
    input: transpose2d(prenetOut.data, tokens.length, prenetDim),
    weight: upsampleWeight.data,
    bias: weights.maybe("wave_conv_upsample.bias")?.data,
    N: 1,
    Cin: prenetDim,
    Cout: prenetDim,
    L: tokens.length,
    K: upsampleWeight.shape[2]!,
    stride: 2,
  });
  const upsampledLength = upsampled.length / prenetDim;
  stages.after_conv_upsample = { data: upsampled, shape: [prenetDim, upsampledLength] };

  const interpolated = interpolateLinear(stages.after_conv_upsample, prenetDim, stftLength);
  stages.after_interpolate = interpolated;

  const dim = config.decoder.dim;
  stages.after_prior_net = await resnetStack(
    interpolated,
    dim,
    stftLength,
    config,
    "wave_prior_net",
    weights,
    backend,
  );

  const decoderInput: Tensor = {
    data: transpose2d(stages.after_prior_net.data, dim, stftLength),
    shape: [stftLength, dim],
  };
  const decoded = await transformer(
    decoderInput,
    config.decoder,
    "wave_decoder",
    weights,
    globalEmbedding,
    backend,
  );
  stages.after_decoder = decoded;

  stages.after_post_net = await resnetStack(
    { data: transpose2d(decoded.data, stftLength, dim), shape: [dim, stftLength] },
    dim,
    stftLength,
    config,
    "wave_post_net",
    weights,
    backend,
  );

  const headInput: Tensor = {
    data: transpose2d(stages.after_post_net.data, dim, stftLength),
    shape: [stftLength, dim],
  };
  const projected = await linear(
    headInput,
    weights.get("istft_head.out.weight"),
    weights.maybe("istft_head.out.bias"),
    backend,
  );
  stages.istft_linear = projected;

  // `chunk(2, dim=1)` on `[B, out_dim, L]`: the first half is log-magnitude and
  // the second is phase. Here the layout is `[L, out_dim]`, so the split is
  // within each row rather than across the buffer.
  const bins = config.nFft / 2 + 1;
  const real = new Float32Array(stftLength * bins);
  const imag = new Float32Array(stftLength * bins);
  for (let t = 0; t < stftLength; t += 1) {
    for (let bin = 0; bin < bins; bin += 1) {
      // `exp` then clamp at 1e2 — upstream's safeguard, at upstream's value.
      const magnitude = Math.min(Math.exp(projected.data[t * 2 * bins + bin]!), 1e2);
      const phase = projected.data[t * 2 * bins + bins + bin]!;
      real[t * bins + bin] = magnitude * Math.cos(phase);
      imag[t * bins + bin] = magnitude * Math.sin(phase);
    }
  }
  stages.spec_real = { data: real, shape: [stftLength, bins] };
  stages.spec_imag = { data: imag, shape: [stftLength, bins] };

  const waveform = await backend.istft(
    real,
    imag,
    hannWindow(config.nFft),
    stftLength,
    config.nFft,
    config.hopLength,
  );
  stages.waveform = { data: waveform, shape: [waveform.length] };

  return { waveform, stages };
}
