import { describe, expect, it } from "vitest";

import * as voxshot from "../src/index.js";

describe("public API", () => {
  it("exports the facade and its building blocks", () => {
    expect(typeof voxshot.VoxShot.create).toBe("function");
    expect(typeof voxshot.SynthesizedAudio).toBe("function");
    expect(typeof voxshot.PlaceholderEngine).toBe("function");
    expect(typeof voxshot.MemoryVoiceStore).toBe("function");
    expect(typeof voxshot.IndexedDbVoiceStore).toBe("function");
    expect(typeof voxshot.createBrowserPlatform).toBe("function");
    expect(typeof voxshot.resolveDevice).toBe("function");
  });

  it("exports the text helpers", () => {
    expect(voxshot.normalizeText("  a  b ")).toBe("a b");
    expect(voxshot.splitSentences("a. b.")).toEqual(["a.", "b."]);
  });

  it("exports the audio helpers", () => {
    expect(voxshot.encodeWav(new Float32Array([0]), 16_000).byteLength).toBe(46);
    expect(Array.from(voxshot.toMono([new Float32Array([1]), new Float32Array([0])]))).toEqual([
      0.5,
    ]);
    expect(voxshot.resample(new Float32Array([0, 1]), 8_000, 16_000)).toHaveLength(4);
    expect(voxshot.normalizePeak(new Float32Array([0.5]), 1)[0]).toBeCloseTo(1, 6);
    expect(voxshot.trimSilence(new Float32Array([0, 1, 0]))).toHaveLength(1);
  });

  it("exports the error types", () => {
    expect(voxshot.isVoxShotError(new voxshot.NoVoiceError())).toBe(true);
    expect(new voxshot.VoxShotError("x").code).toBe("UNKNOWN");
    expect(typeof voxshot.DeviceUnavailableError).toBe("function");
    expect(typeof voxshot.VoiceNotFoundError).toBe("function");
    expect(typeof voxshot.InvalidInputError).toBe("function");
    expect(typeof voxshot.AudioDecodeError).toBe("function");
    expect(typeof voxshot.DisposedError).toBe("function");
    expect(new voxshot.LoadStalledError(1000).code).toBe("LOAD_STALLED");
  });

  it("exports the embedding size of the built in engine", () => {
    expect(voxshot.VOICE_EMBEDDING_SIZE).toBe(4);
  });

  it("exports the Chatterbox engine and its load planning", () => {
    expect(typeof voxshot.ChatterboxEngine).toBe("function");
    expect(voxshot.CHATTERBOX_SAMPLE_RATE).toBe(24_000);
    expect(voxshot.buildLoadPlans("wasm")).toHaveLength(1);
  });

  it("exports the worker transport", () => {
    expect(typeof voxshot.WorkerSynthesisEngine).toBe("function");
    expect(typeof voxshot.exposeEngine).toBe("function");
    expect(voxshot.PROTOCOL_VERSION).toBe(1);
  });
});
