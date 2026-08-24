import { cpuBackend, type Backend } from "./backend.js";
import { type Signal } from "./decoder.js";
import { convWeights, tensor, weightIndex } from "./weights.js";

/**
 * DACVAE's encoder: a 48 kHz waveform to a 32-dimension latent at 25 Hz.
 *
 * The decoder mirrored, and added for `spike/irodori`: cloning a voice means
 * turning a reference clip into a latent, and a text-to-speech pipeline that
 * can only speak in one recorded voice is a partial pipeline.
 *
 * ```
 * waveform [1, T]
 *   -> encoder.block.0               conv 1 -> 64, k=7
 *   -> EncoderBlock x4               rates 2, 8, 10, 12   (64 -> 128 -> 256 -> 512 -> 1024)
 *   -> encoder.block.5               snake
 *   -> encoder.block.6               conv 1024 -> 1024, k=3
 *   -> quantizer.in_proj             1024 -> 64, k=1
 *   -> take the first 32 channels
 * latent [32, T / 1920]
 * ```
 *
 * One block is three residual units then `snake -> conv(stride=rate,
 * k=2*rate)` — the exact reverse of the decoder's `snake -> convTranspose ->
 * three residual units`, and the residual unit itself is identical.
 *
 * ## The last step is a VAE, not a slice
 *
 * `quantizer.in_proj` produces **64** channels for a 32-dimension latent,
 * because they are the mean and the log-variance of a Gaussian. Encoding
 * deterministically — which is what `codec_deterministic_encode=True` asks for,
 * and what a reference clip should be — takes the mean and discards the rest.
 * Reading all 64 as the latent, or averaging the two halves, both produce a
 * tensor of the right shape that conditions the model on nonsense.
 *
 * ## One new argument, no new operation
 *
 * The encoder downsamples with strided `conv1d` where the decoder upsamples
 * with `convTranspose1d`. web-xpu-ops' reference and its WGSL kernel both took
 * a stride already; only this spike's `Backend` seam and GPU host code assumed
 * one.
 *
 * ## What comes before this, and is not here
 *
 * `encode_waveform` normalises the clip to **-16 dB LUFS** before encoding —
 * a K-weighted, gated loudness measurement and a gain, which is a filter design
 * problem rather than a tensor one. It is not ported. `check-encode.ts` reads
 * the waveform the reference actually handed the encoder, so this file is
 * checked against real input rather than against a clip that skipped a step,
 * and the gap is one named thing rather than a diffuse disagreement.
 */

export interface EncoderConfig {
  sampleRate: number;
  hopLength: number;
  latentDim: number;
  encoderDim: number;
  encoderRates: number[];
}

export function encoderConfig(): EncoderConfig {
  const { config } = weightIndex();
  return {
    sampleRate: config.sample_rate,
    hopLength: config.hop_length,
    latentDim: config.latent_dim,
    encoderDim: config.encoder_dim,
    encoderRates: config.encoder_rates,
  };
}

/** Same rule as the decoder's, from `NormConv1d` with `pad_mode="none"`. */
function convPadding(kernel: number, stride: number, dilation: number): number {
  return Math.floor(((kernel - stride) * dilation) / 2);
}

async function applyConv(
  backend: Backend,
  input: Signal,
  prefix: string,
  dilation = 1,
  stride = 1,
): Promise<Signal> {
  const { weight, bias, shape } = convWeights(prefix);
  const [cOut, cIn, k] = shape as [number, number, number];
  if (cIn !== input.channels) {
    throw new Error(`${prefix}: expects ${cIn} channels, got ${input.channels}`);
  }
  const out = await backend.conv1d({
    input: input.data,
    weight,
    bias,
    Cin: cIn,
    Cout: cOut,
    L: input.length,
    K: k,
    padding: convPadding(k, stride, dilation),
    dilation,
    stride,
  });
  return { data: out, channels: cOut, length: out.length / cOut };
}

async function applySnake(backend: Backend, input: Signal, prefix: string): Promise<Signal> {
  const alpha = tensor(`${prefix}.alpha`);
  if (alpha.length !== input.channels) {
    throw new Error(`${prefix}: alpha has ${alpha.length} channels, input has ${input.channels}`);
  }
  return {
    data: await backend.snake({ input: input.data, alpha, C: input.channels, L: input.length }),
    channels: input.channels,
    length: input.length,
  };
}

/** Identical to the decoder's — `snake -> conv(k=7, d) -> snake -> conv(k=1) -> + x`. */
async function residualUnit(
  backend: Backend,
  input: Signal,
  prefix: string,
  dilation: number,
): Promise<Signal> {
  let x = await applySnake(backend, input, `${prefix}.block.0`);
  x = await applyConv(backend, x, `${prefix}.block.1`, dilation);
  x = await applySnake(backend, x, `${prefix}.block.2`);
  x = await applyConv(backend, x, `${prefix}.block.3`, 1);

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

/** Three residual units, then snake and a strided convolution. */
async function encoderBlock(
  backend: Backend,
  input: Signal,
  index: number,
  rate: number,
): Promise<Signal> {
  const prefix = `encoder.block.${index}`;
  let x = await residualUnit(backend, input, `${prefix}.block.0`, 1);
  x = await residualUnit(backend, x, `${prefix}.block.1`, 3);
  x = await residualUnit(backend, x, `${prefix}.block.2`, 9);
  x = await applySnake(backend, x, `${prefix}.block.3`);
  return applyConv(backend, x, `${prefix}.block.4`, 1, rate);
}

/** Encode a `[1, samples]` waveform into a `[latentDim, frames]` latent. */
export async function encode(
  waveform: Signal,
  options: { trace?: (stage: string, signal: Signal) => void; backend?: Backend } = {},
): Promise<Signal> {
  const backend = options.backend ?? cpuBackend;
  const config = encoderConfig();
  const trace = options.trace;
  if (waveform.channels !== 1) {
    throw new Error(`waveform has ${waveform.channels} channels, expected mono`);
  }

  // `_pad`: reflect-pad on the right to a whole number of hops. Reflect, not
  // zero — a zero tail is a transient the encoder hears.
  const remainder = waveform.length % config.hopLength;
  if (remainder !== 0) {
    const extra = config.hopLength - remainder;
    if (extra > waveform.length - 1) {
      throw new Error(`clip of ${waveform.length} samples is too short to reflect-pad by ${extra}`);
    }
    const padded = new Float32Array(waveform.length + extra);
    padded.set(waveform.data);
    for (let i = 0; i < extra; i += 1) padded[waveform.length + i] = waveform.data[waveform.length - 2 - i]!;
    waveform = { data: padded, channels: 1, length: padded.length };
  }

  let x = await applyConv(backend, waveform, "encoder.block.0");
  trace?.("encoder_block_0", x);

  for (const [index, rate] of config.encoderRates.entries()) {
    x = await encoderBlock(backend, x, index + 1, rate);
    trace?.(`encoder_block_${index + 1}`, x);
  }

  const tailIndex = config.encoderRates.length + 1;
  x = await applySnake(backend, x, `encoder.block.${tailIndex}`);
  x = await applyConv(backend, x, `encoder.block.${tailIndex + 1}`);
  trace?.("encoder_out", x);

  const projected = await applyConv(backend, x, "quantizer.in_proj");
  trace?.("quantizer_in_proj", projected);

  // Mean and log-variance, interleaved as channels. Deterministic encoding is
  // the mean; the second half is the distribution's width and is dropped.
  if (projected.channels !== 2 * config.latentDim) {
    throw new Error(
      `quantizer.in_proj gave ${projected.channels} channels, expected ${2 * config.latentDim} ` +
        `(mean and log-variance for a ${config.latentDim}-dimension latent)`,
    );
  }
  const mean = projected.data.subarray(0, config.latentDim * projected.length);
  return { data: new Float32Array(mean), channels: config.latentDim, length: projected.length };
}
