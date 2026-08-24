/**
 * ITU-R BS.1770-4 loudness, and the normalisation Irodori applies before
 * encoding a reference clip.
 *
 * `encode_waveform` normalises to **-16 dB LUFS** by default, and skipping it
 * is not cosmetic: for `samples/reference-voice.wav` it is a gain of 3.36x, and
 * encoding the raw clip instead moves the latent by **50% of peak**. A speaker
 * condition that far off is a different voice, so this had to be ported rather
 * than noted.
 *
 * ## What the reference actually runs
 *
 * `audiotools`' `AudioSignal.normalize()`, whose meter has two paths: an FIR
 * approximation on GPU and true IIR biquads on CPU. Irodori's codec loads on
 * whatever device it was given, and the goldens here were dumped on CPU, so
 * this ports the **IIR** path. They are not the same function — the FIR is
 * truncated at 512 taps — and porting the wrong one would leave a small,
 * permanent disagreement with no obvious cause.
 *
 * The coefficients below were printed from a live `Meter(48000)` rather than
 * recomputed from the shelf and high-pass designs. They depend on the sample
 * rate, which is why {@link normalizeLoudness} refuses a rate it has no
 * coefficients for instead of quietly using 48 kHz's.
 */

/** `[b, a]` for one biquad, as `scipy.signal.lfilter` takes them. */
type Biquad = { b: [number, number, number]; a: [number, number, number] };

/**
 * K-weighting at 48 kHz: a high-frequency shelf for the head's response, then a
 * high-pass. Both `passband_gain` are 1.0 in this build, so neither is applied.
 */
const K_WEIGHTING_48K: Biquad[] = [
  {
    b: [1.5351828863637502, -2.691804030199196, 1.198426263333146],
    a: [1.0, -1.6906995865986896, 0.7325047060963897],
  },
  {
    b: [0.9950442970178917, -1.9900885940357833, 0.9950442970178917],
    a: [1.0, -1.990076284018423, 0.9901009040531438],
  },
];

const BLOCK_SECONDS = 0.4;
const OVERLAP = 0.75;
/** -70 LKFS, the absolute gate. */
const GAMMA_A = -70;

/**
 * A direct-form-I biquad, zero initial state — `lfilter`'s default.
 *
 * `clamp=False` in the reference, so nothing is clipped between stages.
 */
function biquad(input: Float32Array, { b, a }: Biquad): Float32Array {
  const out = new Float32Array(input.length);
  let x1 = 0;
  let x2 = 0;
  let y1 = 0;
  let y2 = 0;
  for (let n = 0; n < input.length; n += 1) {
    const x0 = input[n]!;
    const y0 = b[0] * x0 + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2;
    x2 = x1;
    x1 = x0;
    y2 = y1;
    y1 = y0;
    out[n] = y0;
  }
  return out;
}

/**
 * Integrated loudness in LUFS, mono.
 *
 * The two-pass gate is the part that is easy to write as one pass and get
 * wrong: block powers above the absolute gate give a provisional loudness, the
 * relative gate is that minus 10 LU, and the answer is the mean power of the
 * blocks above *both*. A single absolute gate reads a few LU high on anything
 * with silence in it.
 */
export function integratedLoudness(waveform: Float32Array, sampleRate: number): number {
  if (sampleRate !== 48000) {
    throw new Error(`no K-weighting coefficients for ${sampleRate} Hz — only 48000`);
  }
  let filtered = waveform;
  for (const stage of K_WEIGHTING_48K) filtered = biquad(filtered, stage);

  const blockSize = Math.floor(BLOCK_SECONDS * sampleRate);
  const stride = Math.floor(BLOCK_SECONDS * sampleRate * (1 - OVERLAP));
  if (filtered.length < blockSize) {
    throw new Error(`clip of ${filtered.length} samples is shorter than one ${blockSize}-sample block`);
  }

  // Mean square per block. Mono, so the channel gain G is 1 and drops out.
  //
  // The block count follows `julius.core.unfold`, which **zero-pads the tail**
  // so every sample is covered by at least one frame:
  // `frames = ceil((T - K) / stride) + 1`. Stopping at the last whole block
  // instead drops one partial block, and since that block is part zeros its
  // power is low — leaving out the one thing that would have pulled the mean
  // down. On a 19.8 s clip that was 0.011 LUFS, which is small, real, and not
  // something a waveform comparison can see.
  const frames = Math.ceil((Math.max(filtered.length, blockSize) - blockSize) / stride) + 1;
  const power: number[] = [];
  for (let frame = 0; frame < frames; frame += 1) {
    const start = frame * stride;
    let sum = 0;
    for (let i = start; i < start + blockSize; i += 1) {
      const value = i < filtered.length ? filtered[i]! : 0;
      sum += value * value;
    }
    power.push(sum / blockSize);
  }
  const loudnessOf = (z: number) => -0.691 + 10 * Math.log10(z);

  let sum = 0;
  let count = 0;
  for (const z of power) {
    if (loudnessOf(z) <= GAMMA_A) continue;
    sum += z;
    count += 1;
  }
  if (count === 0) return -Infinity;
  const relative = loudnessOf(sum / count) - 10;

  sum = 0;
  count = 0;
  for (const z of power) {
    const l = loudnessOf(z);
    if (l <= GAMMA_A || l <= relative) continue;
    sum += z;
    count += 1;
  }
  if (count === 0) return -Infinity;
  return loudnessOf(sum / count);
}

/**
 * `signal.normalize(target)` then `ensure_max_of_audio()`.
 *
 * The peak clamp after the gain is not decoration — a quiet clip gained up to
 * -16 LUFS can exceed full scale, and the reference scales it back down. For
 * `samples/reference-voice.wav` it does exactly that: the normalised waveform's
 * peak is 1.000, not something below it.
 */
export function normalizeLoudness(
  waveform: Float32Array,
  sampleRate: number,
  targetDb: number,
): { data: Float32Array; measured: number; gain: number } {
  const measured = integratedLoudness(waveform, sampleRate);
  // `gain = exp((target - measured) * ln(10) / 20)`, which is the usual
  // `10 ** (delta / 20)` written the way `audiotools` writes it.
  const gain = Math.exp(((targetDb - measured) * Math.log(10)) / 20);

  const out = new Float32Array(waveform.length);
  let peak = 0;
  for (let i = 0; i < waveform.length; i += 1) {
    const value = waveform[i]! * gain;
    out[i] = value;
    peak = Math.max(peak, Math.abs(value));
  }
  if (peak > 1) {
    for (let i = 0; i < out.length; i += 1) out[i]! /= peak;
  }
  return { data: out, measured, gain };
}
