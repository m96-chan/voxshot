import { encoderConfig } from "./encoder.js";
import type { ResidentGpu, Tensor } from "./gpu-resident.js";
import { convWeights, tensor } from "./weights.js";

/**
 * The encode graph on {@link ResidentGpu} — the same graph as `encoder.ts`.
 *
 * `encoder.ts` is the definition of correct and `check-encode.ts` compares it
 * against the reference block by block. This one exists for the same reason
 * `decode-gpu.ts` does, and it matters more: swapping the reference voice is
 * the point of the demo, and every swap pays this. Nineteen seconds of audio
 * took **4.6 s** through the readback path.
 *
 * The mirror of the decoder — three residual units then a strided convolution
 * where the decoder does a transposed convolution then three residual units.
 */

function convPadding(kernel: number, stride: number, dilation: number): number {
  return Math.floor(((kernel - stride) * dilation) / 2);
}

interface Shaped {
  tensor: Tensor;
  channels: number;
  length: number;
}

function applyConv(
  gpu: ResidentGpu,
  input: Shaped,
  prefix: string,
  slot: string,
  dilation = 1,
  stride = 1,
): Shaped {
  const { weight, bias, shape } = convWeights(prefix);
  const [cOut, cIn, k] = shape as [number, number, number];
  if (cIn !== input.channels) {
    throw new Error(`${prefix}: expects ${cIn} channels, got ${input.channels}`);
  }
  const out = gpu.conv1d({
    input: input.tensor,
    weight: gpu.weight(weight),
    bias: gpu.weight(bias ?? new Float32Array(cOut)),
    Cin: cIn,
    Cout: cOut,
    L: input.length,
    K: k,
    padding: convPadding(k, stride, dilation),
    dilation,
    stride,
    slot,
  });
  return { tensor: out, channels: cOut, length: out.length / cOut };
}

function applySnake(gpu: ResidentGpu, input: Shaped, prefix: string, slot: string): Shaped {
  const alpha = tensor(`${prefix}.alpha`);
  if (alpha.length !== input.channels) {
    throw new Error(`${prefix}: alpha has ${alpha.length} channels, input has ${input.channels}`);
  }
  return {
    tensor: gpu.snake(input.tensor, alpha, input.channels, input.length, slot),
    channels: input.channels,
    length: input.length,
  };
}

/** Identical to the decoder's. */
function residualUnit(
  gpu: ResidentGpu,
  input: Shaped,
  prefix: string,
  slot: string,
  dilation: number,
): Shaped {
  let x = applySnake(gpu, input, `${prefix}.block.0`, `${slot}.s0`);
  x = applyConv(gpu, x, `${prefix}.block.1`, `${slot}.c1`, dilation);
  x = applySnake(gpu, x, `${prefix}.block.2`, `${slot}.s2`);
  x = applyConv(gpu, x, `${prefix}.block.3`, `${slot}.c3`, 1);
  if (x.length !== input.length) {
    throw new Error(`${prefix}: the shortcut needs a centre crop, which this path does not implement`);
  }
  return { tensor: gpu.add(x.tensor, input.tensor, `${slot}.add`), channels: x.channels, length: x.length };
}

/** Three residual units, then snake and a strided convolution. */
function encoderBlock(gpu: ResidentGpu, input: Shaped, index: number, rate: number): Shaped {
  const prefix = `encoder.block.${index}`;
  const slot = `e${index}`;
  let x = residualUnit(gpu, input, `${prefix}.block.0`, `${slot}.r1`, 1);
  x = residualUnit(gpu, x, `${prefix}.block.1`, `${slot}.r3`, 3);
  x = residualUnit(gpu, x, `${prefix}.block.2`, `${slot}.r9`, 9);
  x = applySnake(gpu, x, `${prefix}.block.3`, `${slot}.s`);
  return applyConv(gpu, x, `${prefix}.block.4`, `${slot}.c`, 1, rate);
}

/**
 * Reflect-pad a waveform to a whole number of hops.
 *
 * On the host, before the upload: it is one pass over the samples and the
 * device would need a kernel for it.
 */
export function padForHop(samples: Float32Array, hop: number): Float32Array {
  const remainder = samples.length % hop;
  if (remainder === 0) return samples;
  const extra = hop - remainder;
  if (extra > samples.length - 1) {
    throw new Error(`clip of ${samples.length} samples is too short to reflect-pad by ${extra}`);
  }
  const padded = new Float32Array(samples.length + extra);
  padded.set(samples);
  for (let i = 0; i < extra; i += 1) padded[samples.length + i] = samples[samples.length - 2 - i]!;
  return padded;
}

/**
 * A `[1, samples]` waveform to a `[latentDim, frames]` latent, on the device.
 *
 * `samples` must already be reflect-padded — {@link padForHop} — and
 * loudness-normalised, which `loudness.ts` does on the host.
 */
export function encodeGpu(gpu: ResidentGpu, waveform: Tensor, samples: number): Shaped {
  const config = encoderConfig();
  let x: Shaped = { tensor: waveform, channels: 1, length: samples };

  x = applyConv(gpu, x, "encoder.block.0", "e0");
  for (const [index, rate] of config.encoderRates.entries()) {
    x = encoderBlock(gpu, x, index + 1, rate);
  }
  const tail = config.encoderRates.length + 1;
  x = applySnake(gpu, x, `encoder.block.${tail}`, "etail.s");
  x = applyConv(gpu, x, `encoder.block.${tail + 1}`, "etail.c");

  const projected = applyConv(gpu, x, "quantizer.in_proj", "in_proj");
  if (projected.channels !== 2 * config.latentDim) {
    throw new Error(
      `quantizer.in_proj gave ${projected.channels} channels, expected ${2 * config.latentDim} ` +
        `(mean and log-variance for a ${config.latentDim}-dimension latent)`,
    );
  }
  // Deterministic encoding is the mean — the first half. The second half is the
  // distribution's width and is dropped.
  const mean = gpu.view(projected.tensor, 0, config.latentDim * projected.length);
  if (!mean) throw new Error("the mean half is not bindable, which cannot happen at offset 0");
  return { tensor: mean, channels: config.latentDim, length: projected.length };
}
