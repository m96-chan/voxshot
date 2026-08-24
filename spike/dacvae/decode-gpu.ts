import { decoderConfig } from "./decoder.js";
import type { ResidentGpu, Tensor } from "./gpu-resident.js";
import { convWeights, has, tensor } from "./weights.js";

/**
 * The decode graph on {@link ResidentGpu} — the same graph as `decoder.ts`.
 *
 * That file is the definition of correct and `check.ts` compares it against the
 * reference stage by stage. This one exists because reading every intermediate
 * back cost four seconds of a five-second request:
 *
 * ```
 *   decoder_model_3   2030 ms   127.5 MB
 *   decoder_model_4   1859 ms   127.5 MB
 * ```
 *
 * Nine operations each, over a tensor that was copied down and back up between
 * every one of them.
 *
 * ## One difference, and it is not in the arithmetic
 *
 * `residualUnit` in `decoder.ts` centre-crops the shortcut before adding it.
 * The crop is always zero for this checkpoint — every main-path convolution is
 * length-preserving — and that file implements it anyway, because "it happens
 * to be zero for these numbers" is a property of the config rather than of the
 * code. Here the same reasoning gives the same guard, and the add is a kernel
 * rather than a loop.
 *
 * ## Slots
 *
 * Scratch is pooled by name, so two tensors live at once must not share one.
 * The shapes change between blocks, and `scratch` keys on `(slot, length)`, so
 * a name reused at a different width is a different buffer — which is why the
 * names below carry the block index rather than being reused across it.
 */

/** `NormConv1d`'s padding rule with `pad_mode="none"`, copied not derived. */
function convPadding(kernel: number, stride: number, dilation: number): number {
  return Math.floor(((kernel - stride) * dilation) / 2);
}

/**
 * `NormConvTranspose1d`'s, likewise — written the way the reference writes it.
 *
 * The odd-stride half is unverified here for the same reason `decoder.ts` gives:
 * this checkpoint's rates are all even, so `output_padding` is always 0.
 */
function transposePadding(stride: number): { padding: number; outputPadding: number } {
  return { padding: Math.floor((stride + 1) / 2), outputPadding: stride % 2 === 1 ? 1 : 0 };
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
): Shaped {
  const { weight, bias, shape } = convWeights(prefix);
  const [cOut, cIn, k] = shape as [number, number, number];
  if (cIn !== input.channels) {
    throw new Error(`${prefix}: expects ${cIn} channels, got ${input.channels}`);
  }
  const padding = convPadding(k, 1, dilation);
  const out = gpu.conv1d({
    input: input.tensor,
    weight: gpu.weight(weight),
    bias: gpu.weight(bias ?? new Float32Array(cOut)),
    Cin: cIn,
    Cout: cOut,
    L: input.length,
    K: k,
    padding,
    dilation,
    slot,
  });
  return { tensor: out, channels: cOut, length: out.length / cOut };
}

function applyTranspose(gpu: ResidentGpu, input: Shaped, prefix: string, slot: string, stride: number): Shaped {
  const { weight, bias, shape } = convWeights(prefix);
  const [cIn, cOut, k] = shape as [number, number, number];
  if (cIn !== input.channels) {
    throw new Error(`${prefix}: expects ${cIn} channels, got ${input.channels}`);
  }
  const { padding, outputPadding } = transposePadding(stride);
  const out = gpu.convTranspose1d({
    input: input.tensor,
    weight: gpu.weight(weight),
    bias: gpu.weight(bias ?? new Float32Array(cOut)),
    Cin: cIn,
    Cout: cOut,
    L: input.length,
    K: k,
    stride,
    padding,
    outputPadding,
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

/**
 * `snake -> conv(k=7, d) -> snake -> conv(k=1) -> + x`.
 *
 * The four intermediates are strictly sequential — each reads only the one
 * before it — so they share two slots that alternate, and the three residual
 * units of a block share the same two. Giving each its own name was 5 slots per
 * unit and 15 per block; at 92 MB apiece in the last two blocks that was
 * **1.4 GB a block** of buffers that are never live at the same moment.
 *
 * The two are shared across blocks as well as across units, because a block's
 * working buffers do not outlive it. `input` must not be one of them: the
 * shortcut is read at the end, which is why the caller alternates the unit
 * outputs.
 */
function residualUnit(
  gpu: ResidentGpu,
  input: Shaped,
  prefix: string,
  slot: string,
  dilation: number,
): Shaped {
  let x = applySnake(gpu, input, `${prefix}.block.0`, "work.a");
  x = applyConv(gpu, x, `${prefix}.block.1`, "work.b", dilation);
  x = applySnake(gpu, x, `${prefix}.block.2`, "work.a");
  x = applyConv(gpu, x, `${prefix}.block.3`, "work.b", 1);
  if (x.length !== input.length) {
    throw new Error(
      `${prefix}: the shortcut needs a centre crop from ${input.length} to ${x.length}, ` +
        `which this path does not implement — see decoder.ts, where it is a no-op for this config`,
    );
  }
  return { tensor: gpu.add(x.tensor, input.tensor, slot), channels: x.channels, length: x.length };
}

function decoderBlock(gpu: ResidentGpu, input: Shaped, index: number, rate: number): Shaped {
  const prefix = `decoder.model.${index}`;
  const slot = `d${index}`;
  let x = applySnake(gpu, input, `${prefix}.block.0`, `${slot}.s`);
  // The transposed convolution's output *is* the first unit's input, so it goes
  // straight into one of the two alternating unit slots rather than a third.
  x = applyTranspose(gpu, x, `${prefix}.block.1`, `${slot}.u1`, rate);
  // Unit outputs alternate: a unit's shortcut is its input, so it cannot write
  // into the slot it is reading from.
  x = residualUnit(gpu, x, `${prefix}.block.4`, `${slot}.u0`, 1);
  x = residualUnit(gpu, x, `${prefix}.block.5`, `${slot}.u1`, 3);
  x = residualUnit(gpu, x, `${prefix}.block.8`, `${slot}.u0`, 9);
  if (has(`${prefix}.block.9.block.0.alpha`)) {
    throw new Error(`${prefix}.block.9 is a residual unit in this checkpoint, which is unhandled`);
  }
  return x;
}

/** A `[latentDim, frames]` latent to a `[1, frames * hop]` waveform, on the device. */
export function decodeGpu(gpu: ResidentGpu, latent: Tensor, frames: number): Shaped {
  const config = decoderConfig();
  let x: Shaped = { tensor: latent, channels: config.latentDim, length: frames };

  x = applyConv(gpu, x, "quantizer.out_proj", "out_proj");
  x = applyConv(gpu, x, "decoder.model.0", "d0");
  for (const [index, rate] of config.decoderRates.entries()) {
    x = decoderBlock(gpu, x, index + 1, rate);
  }

  // The output tail lives inside `wm_model.encoder_block.pre` — skipping the
  // watermark model wholesale would leave 96 channels where a waveform belongs.
  x = applySnake(gpu, x, "decoder.wm_model.encoder_block.pre.0", "tail.s");
  x = applyConv(gpu, x, "decoder.wm_model.encoder_block.pre.1", "tail.c");
  return { tensor: gpu.tanh(x.tensor, "tail.t"), channels: x.channels, length: x.length };
}
