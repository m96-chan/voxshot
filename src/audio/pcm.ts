import { InvalidInputError } from "../errors.js";

/**
 * Down-mix interleaved channel data to a single mono channel by averaging.
 *
 * Channels of differing lengths are truncated to the shortest one, which is
 * what browsers hand back for malformed files.
 */
export function toMono(channels: readonly Float32Array[]): Float32Array {
  if (channels.length === 0) {
    throw new InvalidInputError("At least one audio channel is required.");
  }
  if (channels.length === 1) {
    return Float32Array.from(channels[0] as Float32Array);
  }

  const length = channels.reduce((min, channel) => Math.min(min, channel.length), Infinity);
  const mono = new Float32Array(length);

  for (const channel of channels) {
    for (let index = 0; index < length; index += 1) {
      mono[index] += channel[index] as number;
    }
  }
  for (let index = 0; index < length; index += 1) {
    mono[index] /= channels.length;
  }
  return mono;
}

/**
 * Resample a mono signal with linear interpolation.
 *
 * Linear interpolation is intentionally simple: reference audio is band
 * limited by the model front-end anyway, and this keeps the library free of
 * a polyphase filter implementation for v0.1.
 */
export function resample(samples: Float32Array, fromRate: number, toRate: number): Float32Array {
  assertPositiveRate(fromRate, "fromRate");
  assertPositiveRate(toRate, "toRate");

  if (fromRate === toRate) {
    return Float32Array.from(samples);
  }
  if (samples.length === 0) {
    return new Float32Array(0);
  }

  const ratio = toRate / fromRate;
  const length = Math.max(1, Math.round(samples.length * ratio));
  const output = new Float32Array(length);
  const lastIndex = samples.length - 1;

  for (let index = 0; index < length; index += 1) {
    const position = index / ratio;
    const left = Math.min(Math.floor(position), lastIndex);
    const right = Math.min(left + 1, lastIndex);
    const fraction = position - left;
    const a = samples[left] as number;
    const b = samples[right] as number;
    output[index] = a + (b - a) * fraction;
  }
  return output;
}

/**
 * Scale a signal so that its loudest sample sits exactly at `targetPeak`.
 * Silent input is returned unchanged instead of being amplified into noise.
 */
export function normalizePeak(samples: Float32Array, targetPeak = 0.95): Float32Array {
  if (!(targetPeak > 0) || !Number.isFinite(targetPeak)) {
    throw new InvalidInputError("targetPeak must be a positive finite number.");
  }

  let peak = 0;
  for (const sample of samples) {
    const magnitude = Math.abs(sample);
    if (magnitude > peak) {
      peak = magnitude;
    }
  }
  if (peak === 0) {
    return Float32Array.from(samples);
  }

  const gain = targetPeak / peak;
  const output = new Float32Array(samples.length);
  for (let index = 0; index < samples.length; index += 1) {
    output[index] = (samples[index] as number) * gain;
  }
  return output;
}

/**
 * Drop leading and trailing samples whose magnitude stays below `threshold`.
 * Quiet passages inside the signal are preserved.
 */
export function trimSilence(samples: Float32Array, threshold = 0.01): Float32Array {
  let start = 0;
  let end = samples.length - 1;

  while (start <= end && Math.abs(samples[start] as number) < threshold) {
    start += 1;
  }
  while (end >= start && Math.abs(samples[end] as number) < threshold) {
    end -= 1;
  }
  if (start > end) {
    return new Float32Array(0);
  }
  return samples.slice(start, end + 1);
}

function assertPositiveRate(rate: number, name: string): void {
  if (!(rate > 0) || !Number.isFinite(rate)) {
    throw new InvalidInputError(`${name} must be a positive finite number.`);
  }
}
