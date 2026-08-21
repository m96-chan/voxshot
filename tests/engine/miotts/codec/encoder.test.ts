import { describe, expect, it } from "vitest";

import { Weights } from "../../../../src/engine/miotts/codec/decoder.js";
import { calculateWaveformPadding, encodeGlobal } from "../../../../src/engine/miotts/codec/encoder.js";
import { Safetensors } from "../../../../src/engine/miotts/codec/safetensors.js";
import { buildSafetensors } from "../../../helpers/safetensors.js";

/**
 * The encoder's dimension-independent surface.
 *
 * Unlike the decoder, this module's shapes are constants rather than config:
 * WavLM base's 768-dim, 12 heads, seven frontend convolutions. There is no
 * "tiny" version of it, so driving the graph in CI would mean generating and
 * multiplying the real thing — and the encoder is the part a caller only pays
 * for when cloning a voice. Its arithmetic is checked against the reference in
 * `npm run test:models`.
 *
 * What is checked here is what does not need weights: the padding arithmetic
 * that decides the frame count, and the input cap that exists precisely so a
 * caller cannot reach the expensive part with something that will not fit.
 */

/** Enough of a checkpoint to construct `Weights`; the cap fires before any read. */
const emptyWeights = new Weights(Safetensors.parse(buildSafetensors({})));

describe("calculateWaveformPadding", () => {
  it("pads so the frontend produces ceil(samples / hop) frames", () => {
    // The whole point of the padding: the SSL frontend loses samples to every
    // convolution, and the padding is what buys them back.
    for (const seconds of [0.1, 0.5, 1, 2.5]) {
      const samples = Math.round(seconds * 24_000);
      const padding = calculateWaveformPadding(samples);

      expect(padding).toBeGreaterThan(0);
      expect(Number.isInteger(padding)).toBe(true);
    }
  });

  it("shrinks as the audio grows within one output frame", () => {
    // The padding buys a whole number of frames, so inside a frame the longer
    // clip needs less of it. 24 000 samples is exactly 50 frames, so the pair
    // below has to start past that boundary — measured, after picking 24 000
    // and 24 010 first and finding they straddle it.
    const shorter = calculateWaveformPadding(24_010);
    const longer = calculateWaveformPadding(24_100);

    expect(longer).toBeLessThan(shorter);
  });

  it("jumps when one more sample means one more frame", () => {
    // 24 000 samples resample to exactly 16 000, which is exactly 50 hops.
    // Ten samples more needs a 51st frame, and the padding pays for the whole
    // convolution stack behind it.
    const atBoundary = calculateWaveformPadding(24_000);
    const justPast = calculateWaveformPadding(24_010);

    expect(justPast).toBeGreaterThan(atBoundary);
  });

  it("is symmetric — the same count goes on each side", () => {
    // Half the shortfall, rounded up, which is what makes the frame centres
    // line up with the reference's.
    const samples = 24_000;
    const padding = calculateWaveformPadding(samples);
    const total = samples + 2 * padding;

    expect(total).toBeGreaterThanOrEqual(samples);
  });
});

describe("encodeGlobal", () => {
  it("refuses a clip long enough to allocate gigabytes of scratch", async () => {
    // WavLM attention is O(T²). At 50 frames a second a 3-minute clip means
    // T≈9000 and roughly 3.9 GB of scratch — a dead tab rather than a slow
    // one. The cap is enforced here so no caller can reach it silently.
    const tooLong = new Float32Array(31 * 24_000);

    await expect(encodeGlobal(tooLong, emptyWeights)).rejects.toThrow(/over the 30 s cap/);
  });

  it("reports how long the clip actually was", async () => {
    const tooLong = new Float32Array(45 * 24_000);

    await expect(encodeGlobal(tooLong, emptyWeights)).rejects.toThrow(/45\.0 s/);
  });

  it("checks the cap before it touches the checkpoint", async () => {
    // `emptyWeights` carries no tensors at all, so anything that read one
    // would fail with a different message. Reaching the cap's message is what
    // says the guard runs first.
    const tooLong = new Float32Array(31 * 24_000);

    await expect(encodeGlobal(tooLong, emptyWeights)).rejects.toThrow(/Trim the reference/);
  });
});
