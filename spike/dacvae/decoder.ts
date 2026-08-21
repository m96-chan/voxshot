import { conv1d, conv1dOutputLength } from "web-xpu-ops/ops/conv";
import { convTranspose1d } from "web-xpu-ops/ops/conv_transpose";
import { snake } from "web-xpu-ops/ops/snake";

import { convWeights, has, tensor, weightIndex } from "./weights.js";

/**
 * DACVAE's decoder: a 32-dimension latent at 25 Hz to a 48 kHz waveform.
 *
 * The port of `dacvae/model/dacvae.py`'s decode path, and the reason ISSUE #138
 * exists: `snake` is the one operation in web-xpu-ops that no voxshot model had
 * used, so nothing had ever checked our understanding of its contract against
 * a real checkpoint. Upstream's tests say the op computes what it claims; they
 * cannot say we are calling it correctly.
 *
 * ## The graph
 *
 * ```
 * latent [32, T]
 *   -> quantizer.out_proj            32 -> 1024
 *   -> decoder.model.0               conv 1024 -> 1536, k=7
 *   -> DecoderBlock x4               rates 12, 10, 8, 2   (1536 -> 768 -> 384 -> 192 -> 96)
 *   -> tail                          snake -> conv 96 -> 1 -> tanh
 * waveform [1, T * 1920]
 * ```
 *
 * One block is
 *
 * ```
 * snake -> convTranspose(stride=rate, k=2*rate) -> res(d=1) -> res(d=3) -> res(d=9)
 * ```
 *
 * and one residual unit is `snake -> conv(k=7, d) -> snake -> conv(k=1) -> + x`.
 *
 * ## What is deliberately absent
 *
 * Half of what the checkpoint holds. `DecoderBlock` interleaves the main path
 * and a watermark path in one `ModuleList`, and `forward()` walks only the even
 * chunks; the odd ones are reached solely by `upsample_group()`, which only the
 * real `watermark()` calls, and Irodori replaces that method wholesale. The
 * encoder is not here either — decoding never touches it.
 *
 * What Irodori's replacement keeps IS here, though, and it is easy to miss:
 * the output tail lives inside `wm_model.encoder_block.pre`, so "skip the
 * watermark model" would leave 96 channels where a waveform belongs.
 *
 * ## No normalisation layers
 *
 * There are none in the main path. The `weight_norm` the checkpoint carries is
 * a way of storing a filter, not a step in the graph, and `weights.ts` folds it
 * at load. That leaves exactly three operations on the run path — conv,
 * convTranspose and snake — which makes this the simplest graph voxshot has
 * ported.
 */

export interface DecoderConfig {
  sampleRate: number;
  hopLength: number;
  latentDim: number;
  decoderDim: number;
  decoderRates: number[];
}

export function decoderConfig(): DecoderConfig {
  const { config } = weightIndex();
  return {
    sampleRate: config.sample_rate,
    hopLength: config.hop_length,
    latentDim: config.latent_dim,
    decoderDim: config.decoder_dim,
    decoderRates: config.decoder_rates,
  };
}

/** A `[C, L]` activation. Channel-major throughout, as the reference is. */
export interface Signal {
  data: Float32Array;
  channels: number;
  length: number;
}

/**
 * `NormConv1d`'s padding rule, copied from the reference rather than derived:
 *
 *     pad = (kernel_size - stride) * dilation // 2      when pad_mode == "none"
 *
 * Every convolution on the main path uses `pad_mode="none"`; the `"auto"` (zero
 * padding, causal) variant belongs to the watermark path, which is not here.
 */
function convPadding(kernel: number, stride: number, dilation: number): number {
  return Math.floor(((kernel - stride) * dilation) / 2);
}

function applyConv(input: Signal, prefix: string, dilation = 1, stride = 1): Signal {
  const { weight, bias, shape } = convWeights(prefix);
  const [cOut, cIn, k] = shape as [number, number, number];
  if (cIn !== input.channels) {
    throw new Error(`${prefix}: expects ${cIn} channels, got ${input.channels}`);
  }
  const padding = convPadding(k, stride, dilation);
  const out = conv1d({
    input: input.data,
    weight,
    ...(bias ? { bias } : {}),
    N: 1,
    Cin: cIn,
    Cout: cOut,
    L: input.length,
    K: k,
    stride,
    padding,
    dilation,
  });
  return {
    data: out,
    channels: cOut,
    length: conv1dOutputLength({ L: input.length, K: k, stride, padding, dilation }),
  };
}

/**
 * `NormConvTranspose1d`'s padding rule, again copied:
 *
 *     padding        = (stride + 1) // 2
 *     output_padding = 1 if stride % 2 else 0
 *
 * with `kernel_size = 2 * stride`, which is what makes each block multiply the
 * length by exactly its rate.
 *
 * **The odd-stride half of that is unverified.** This checkpoint's rates are
 * [12, 10, 8, 2] — every one even — so `output_padding` is always 0 here and
 * forcing it to 0 changes nothing. Measured, by doing exactly that: every stage
 * still agreed. It is kept because it is the reference's rule and a rung with
 * an odd rate would need it, but nothing here has checked it, and that is worth
 * knowing before trusting this file against a different checkpoint.
 */
function applyConvTranspose(input: Signal, prefix: string, stride: number): Signal {
  const { weight, bias, shape } = convWeights(prefix);
  // Transposed convolutions store `[Cin, Cout, K]`, not `[Cout, Cin, K]`.
  const [cIn, cOut, k] = shape as [number, number, number];
  if (cIn !== input.channels) {
    throw new Error(`${prefix}: expects ${cIn} channels, got ${input.channels}`);
  }
  const padding = Math.floor((stride + 1) / 2);
  const outputPadding = stride % 2 === 1 ? 1 : 0;
  const out = convTranspose1d({
    input: input.data,
    weight,
    ...(bias ? { bias } : {}),
    N: 1,
    Cin: cIn,
    Cout: cOut,
    L: input.length,
    K: k,
    stride,
    padding,
    outputPadding,
  });
  return { data: out, channels: cOut, length: out.length / cOut };
}

function applySnake(input: Signal, prefix: string): Signal {
  // `alpha` is stored as [1, C, 1] — one learned value per channel.
  const alpha = tensor(`${prefix}.alpha`);
  if (alpha.length !== input.channels) {
    throw new Error(`${prefix}: alpha has ${alpha.length} channels, input has ${input.channels}`);
  }
  return {
    data: snake({ input: input.data, alpha, N: 1, C: input.channels, L: input.length }),
    channels: input.channels,
    length: input.length,
  };
}

/**
 * `ResidualUnit.forward`: `y = block(x); return y + shortcut(x, y)`.
 *
 * `shortcut` centre-crops `x` when the block shortened it. Here it never does —
 * `pad = (k - 1) * d / 2` makes every main-path convolution length-preserving —
 * so the crop is a no-op. It is implemented anyway, because "it happens to be
 * zero for these numbers" is a property of the config, not of the code, and the
 * next config would break silently.
 */
function residualUnit(input: Signal, prefix: string, dilation: number): Signal {
  let x = applySnake(input, `${prefix}.block.0`);
  x = applyConv(x, `${prefix}.block.1`, dilation);
  x = applySnake(x, `${prefix}.block.2`);
  x = applyConv(x, `${prefix}.block.3`, 1);

  const crop = (input.length - x.length) / 2;
  if (!Number.isInteger(crop) || crop < 0) {
    throw new Error(`${prefix}: cannot centre-crop ${input.length} to ${x.length}`);
  }
  const out = new Float32Array(x.data.length);
  for (let channel = 0; channel < x.channels; channel += 1) {
    const from = channel * input.length + crop;
    const to = channel * x.length;
    for (let i = 0; i < x.length; i += 1) out[to + i] = x.data[to + i]! + input.data[from + i]!;
  }
  return { data: out, channels: x.channels, length: x.length };
}

/** One upsampling block: snake, transposed conv, then three residual units. */
function decoderBlock(input: Signal, index: number, rate: number): Signal {
  const prefix = `decoder.model.${index}`;
  let x = applySnake(input, `${prefix}.block.0`);
  x = applyConvTranspose(x, `${prefix}.block.1`, rate);
  x = residualUnit(x, `${prefix}.block.4`, 1);
  x = residualUnit(x, `${prefix}.block.5`, 3);
  x = residualUnit(x, `${prefix}.block.8`, 9);
  // `block.9` is `nn.Identity()` unless the block was built with
  // `last_kernel_size`, in which case it is one more residual unit. This
  // checkpoint has none; the check is here so a checkpoint that does is a
  // failure rather than a silently shorter graph.
  if (has(`${prefix}.block.9.block.0.alpha`)) {
    throw new Error(`${prefix}.block.9 is a residual unit in this checkpoint, which is unhandled`);
  }
  return x;
}

/** Decode a `[latentDim, frames]` latent into a `[1, frames * hop]` waveform. */
export function decode(latent: Signal, options: { trace?: (stage: string, signal: Signal) => void } = {}): Signal {
  const config = decoderConfig();
  const trace = options.trace;
  if (latent.channels !== config.latentDim) {
    throw new Error(`latent has ${latent.channels} channels, expected ${config.latentDim}`);
  }

  let x = applyConv(latent, "quantizer.out_proj");
  trace?.("after_out_proj", x);

  x = applyConv(x, "decoder.model.0");
  trace?.("decoder_model_0", x);

  config.decoderRates.forEach((rate, i) => {
    x = decoderBlock(x, i + 1, rate);
    trace?.(`decoder_model_${i + 1}`, x);
  });

  // The tail, which lives inside `wm_model` — see the module doc.
  const tail = "decoder.wm_model.encoder_block.pre";
  x = applySnake(x, `${tail}.0`);
  trace?.("tail_0_snake1d", x);
  x = applyConv(x, `${tail}.1`);
  trace?.("tail_1_normconv1d", x);
  x = { ...x, data: Float32Array.from(x.data, Math.tanh) };
  trace?.("tail_2_tanh", x);

  return x;
}
