import { describe, expect, it } from "vitest";

import { normalizePeak, resample, toMono, trimSilence } from "../../src/audio/pcm.js";
import { InvalidInputError } from "../../src/errors.js";

describe("toMono", () => {
  it("returns the single channel as is", () => {
    const left = new Float32Array([0.1, -0.2, 0.3]);

    expect(Array.from(toMono([left]))).toEqual([
      expect.closeTo(0.1, 6),
      expect.closeTo(-0.2, 6),
      expect.closeTo(0.3, 6),
    ]);
  });

  it("averages every channel sample by sample", () => {
    const left = new Float32Array([1, 0, -1]);
    const right = new Float32Array([0, 1, 1]);

    expect(Array.from(toMono([left, right]))).toEqual([0.5, 0.5, 0]);
  });

  it("uses the shortest channel length", () => {
    const left = new Float32Array([1, 1, 1, 1]);
    const right = new Float32Array([1, 1]);

    expect(toMono([left, right]).length).toBe(2);
  });

  it("rejects an empty channel list", () => {
    expect(() => toMono([])).toThrow(InvalidInputError);
  });
});

describe("resample", () => {
  it("returns a copy when the rate is unchanged", () => {
    const input = new Float32Array([0, 0.5, 1]);
    const output = resample(input, 16_000, 16_000);

    expect(Array.from(output)).toEqual([0, 0.5, 1]);
    expect(output).not.toBe(input);
  });

  it("halves the sample count when downsampling by two", () => {
    const input = new Float32Array(100).fill(0.25);

    expect(resample(input, 32_000, 16_000).length).toBe(50);
  });

  it("doubles the sample count when upsampling by two", () => {
    const input = new Float32Array([0, 1]);
    const output = resample(input, 8_000, 16_000);

    expect(output.length).toBe(4);
  });

  it("interpolates linearly between neighbouring samples", () => {
    const input = new Float32Array([0, 1]);
    const output = resample(input, 1, 2);

    expect(output[0]).toBeCloseTo(0, 5);
    expect(output[1]).toBeCloseTo(0.5, 5);
  });

  it("preserves a constant signal", () => {
    const input = new Float32Array(64).fill(0.4);

    for (const sample of resample(input, 44_100, 24_000)) {
      expect(sample).toBeCloseTo(0.4, 5);
    }
  });

  it("returns an empty array for empty input", () => {
    expect(resample(new Float32Array(0), 44_100, 16_000).length).toBe(0);
  });

  it("rejects non positive sample rates", () => {
    const input = new Float32Array([1]);

    expect(() => resample(input, 0, 16_000)).toThrow(InvalidInputError);
    expect(() => resample(input, 16_000, -1)).toThrow(InvalidInputError);
  });
});

describe("normalizePeak", () => {
  it("scales the loudest sample to the target peak", () => {
    const output = normalizePeak(new Float32Array([0.1, -0.25, 0.05]), 0.5);

    expect(output[1]).toBeCloseTo(-0.5, 5);
    expect(output[0]).toBeCloseTo(0.2, 5);
  });

  it("leaves a silent signal untouched", () => {
    const output = normalizePeak(new Float32Array([0, 0, 0]), 0.9);

    expect(Array.from(output)).toEqual([0, 0, 0]);
  });

  it("attenuates a signal that clips", () => {
    const output = normalizePeak(new Float32Array([2, -4]), 1);

    expect(output[0]).toBeCloseTo(0.5, 5);
    expect(output[1]).toBeCloseTo(-1, 5);
  });

  it("rejects a non positive target peak", () => {
    expect(() => normalizePeak(new Float32Array([1]), 0)).toThrow(InvalidInputError);
  });
});

describe("trimSilence", () => {
  it("removes leading and trailing silence", () => {
    const input = new Float32Array([0, 0, 0.5, -0.5, 0, 0]);

    expect(Array.from(trimSilence(input, 0.01))).toEqual([0.5, -0.5]);
  });

  it("keeps quiet samples between loud ones", () => {
    const input = new Float32Array([0.5, 0, 0, 0.5]);

    expect(trimSilence(input, 0.01).length).toBe(4);
  });

  it("returns an empty array when everything is below the threshold", () => {
    expect(trimSilence(new Float32Array([0.001, -0.002]), 0.01).length).toBe(0);
  });

  it("returns an empty array for empty input", () => {
    expect(trimSilence(new Float32Array(0), 0.01).length).toBe(0);
  });
});
