import { describe, expect, it } from "vitest";
import { hannWindow, istft, istftLength } from "web-xpu-ops/ops/stft";
import { GoldenCase, loadCase, loadIndex, toFrameMajor, worstDifference } from "./golden.js";

/**
 * The decoder's last stage, against the reference implementation's own output.
 *
 * This is the stage the spike was blocked on: MioCodec synthesises with
 * `padding="same"` (X-Codec-2.0's convention), which web-xpu-ops gained in #92.
 * Everything before it is a matter of wiring existing ops together; this is the
 * one that had no expression at all, so it is worth checking first and on its
 * own — it needs no weights, only the spectrogram the golden already carries.
 */

const index = loadIndex();
const { n_fft: N_FFT, hop_length: HOP } = index.config;

function synthesise(golden: GoldenCase): Float32Array {
  const real = golden.tensor("spec_real");
  const imag = golden.tensor("spec_imag");
  // [1, bins, frames] — batch of one, torch's bin-major layout.
  const [, bins, frames] = real.shape as [number, number, number];
  return istft({
    real: toFrameMajor(real.data, bins, frames),
    imag: toFrameMajor(imag.data, bins, frames),
    frames,
    nFft: N_FFT,
    hop: HOP,
    window: hannWindow(N_FFT),
    padding: "same",
  });
}

describe("istft / same padding against the reference decoder", () => {
  it("agrees with torch's config on what this model is", () => {
    // Not ceremony. Every number below is derived from these, and a golden
    // regenerated against a different rung would otherwise fail somewhere far
    // less obvious.
    expect(index.config.istft_padding).toBe("same");
    expect([N_FFT, HOP]).toEqual([1920, 480]);
    expect(index.config.sample_rate).toBe(24000);
  });

  for (const name of Object.keys(index.cases)) {
    const golden = loadCase(name);
    const frames = golden.manifest.stft_length;

    it(`[${name}] returns hop x frames samples, which no other mode does`, () => {
      // The whole reason "same" had to exist. At nFft 1920 / hop 480 the three
      // modes crop 960, 0 and 720 per end, so they disagree about the length by
      // a hop or by a window — a waveform that is a little too long still
      // sounds like audio, which is why this is asserted rather than eyeballed.
      expect(istftLength(N_FFT, HOP, frames, "same")).toBe(HOP * frames);
      expect(istftLength(N_FFT, HOP, frames, "center")).toBe(HOP * (frames - 1));
      expect(istftLength(N_FFT, HOP, frames, "none")).toBe(N_FFT + HOP * (frames - 1));
      expect(golden.manifest.tensors.waveform!.shape).toEqual([HOP * frames]);
    });

    it(`[${name}] reproduces the reference waveform`, () => {
      const expected = golden.tensor("waveform").data;
      const actual = synthesise(golden);
      expect(actual.length).toBe(expected.length);

      const worst = worstDifference(actual, expected);
      // Relative to the waveform's own peak, not to the element: a waveform
      // crosses zero constantly, so per-element relative error is meaningless
      // near a crossing, and the peak is the scale a listener hears.
      //
      // 1e-6 is set from measurement, not from taste. Against a zero tolerance
      // the worst element in each case was:
      //
      //   aligned    abs 8.94e-8   rel 2.65e-7
      //   resampled  abs 2.98e-7   rel 2.39e-7
      //   windowed   abs 1.79e-7   rel 2.85e-7
      //
      // That is f32 round-off — an ulp at these magnitudes is around 6e-8 —
      // between a float64 JavaScript reference and a float32 torch tensor. It
      // leaves no room for a real disagreement to hide: the two other padding
      // modes miss by a whole hop, and a wrong window or a dropped envelope
      // division is off by percent, not by parts per ten million.
      expect(worst.rel, `worst: ${JSON.stringify(worst)}`).toBeLessThan(1e-6);
    });
  }
});
