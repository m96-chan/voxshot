import { describe, expect, it } from "vitest";

import * as zerovox from "../src/index.js";

describe("public API", () => {
  it("exports the facade and its building blocks", () => {
    expect(typeof zerovox.ZeroVox.create).toBe("function");
    expect(typeof zerovox.SynthesizedAudio).toBe("function");
    expect(typeof zerovox.PlaceholderEngine).toBe("function");
    expect(typeof zerovox.MemoryVoiceStore).toBe("function");
    expect(typeof zerovox.IndexedDbVoiceStore).toBe("function");
    expect(typeof zerovox.createBrowserPlatform).toBe("function");
    expect(typeof zerovox.resolveDevice).toBe("function");
  });

  it("exports the text helpers", () => {
    expect(zerovox.normalizeText("  a  b ")).toBe("a b");
    expect(zerovox.splitSentences("a. b.")).toEqual(["a.", "b."]);
  });

  it("exports the audio helpers", () => {
    expect(zerovox.encodeWav(new Float32Array([0]), 16_000).byteLength).toBe(46);
    expect(Array.from(zerovox.toMono([new Float32Array([1]), new Float32Array([0])]))).toEqual([
      0.5,
    ]);
    expect(zerovox.resample(new Float32Array([0, 1]), 8_000, 16_000)).toHaveLength(4);
    expect(zerovox.normalizePeak(new Float32Array([0.5]), 1)[0]).toBeCloseTo(1, 6);
    expect(zerovox.trimSilence(new Float32Array([0, 1, 0]))).toHaveLength(1);
  });

  it("exports the error types", () => {
    expect(zerovox.isZeroVoxError(new zerovox.NoVoiceError())).toBe(true);
    expect(new zerovox.ZeroVoxError("x").code).toBe("UNKNOWN");
    expect(typeof zerovox.DeviceUnavailableError).toBe("function");
    expect(typeof zerovox.VoiceNotFoundError).toBe("function");
    expect(typeof zerovox.InvalidInputError).toBe("function");
    expect(typeof zerovox.AudioDecodeError).toBe("function");
    expect(typeof zerovox.DisposedError).toBe("function");
  });

  it("exports the embedding size of the built in engine", () => {
    expect(zerovox.VOICE_EMBEDDING_SIZE).toBe(4);
  });

  it("exports the Chatterbox engine and its load planning", () => {
    expect(typeof zerovox.ChatterboxEngine).toBe("function");
    expect(zerovox.CHATTERBOX_SAMPLE_RATE).toBe(24_000);
    expect(zerovox.buildLoadPlans("wasm")).toHaveLength(1);
  });

  it("exports the worker transport", () => {
    expect(typeof zerovox.WorkerSynthesisEngine).toBe("function");
    expect(typeof zerovox.exposeEngine).toBe("function");
    expect(zerovox.PROTOCOL_VERSION).toBe(1);
  });
});
