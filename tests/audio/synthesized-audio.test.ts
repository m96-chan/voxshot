import { describe, expect, it, vi } from "vitest";

import { SynthesizedAudio } from "../../src/audio/synthesized-audio.js";
import { InvalidInputError } from "../../src/errors.js";
import type { AudioPlayer } from "../../src/platform.js";

describe("SynthesizedAudio", () => {
  it("exposes its samples and sample rate", () => {
    const samples = new Float32Array([0, 0.5, -0.5, 0]);
    const audio = new SynthesizedAudio(samples, 16_000);

    expect(audio.samples).toBe(samples);
    expect(audio.sampleRate).toBe(16_000);
  });

  it("reports the duration in seconds", () => {
    const audio = new SynthesizedAudio(new Float32Array(8_000), 16_000);

    expect(audio.duration).toBeCloseTo(0.5, 6);
  });

  it("reports a zero duration for an empty signal", () => {
    expect(new SynthesizedAudio(new Float32Array(0), 16_000).duration).toBe(0);
  });

  it("rejects a non positive sample rate", () => {
    expect(() => new SynthesizedAudio(new Float32Array(1), 0)).toThrow(InvalidInputError);
  });

  it("encodes itself as a WAV buffer", () => {
    const audio = new SynthesizedAudio(new Float32Array([0, 1]), 16_000);

    expect(audio.toWav().byteLength).toBe(44 + 4);
  });

  it("wraps the WAV buffer in an audio/wav blob", async () => {
    const audio = new SynthesizedAudio(new Float32Array([0, 1]), 16_000);
    const blob = audio.toBlob();

    expect(blob.type).toBe("audio/wav");
    expect(blob.size).toBe(44 + 4);
  });

  it("delegates playback to the injected player", async () => {
    const player: AudioPlayer = { play: vi.fn(async () => {}) };
    const samples = new Float32Array([0, 1]);
    const audio = new SynthesizedAudio(samples, 16_000, player);

    await audio.play();

    expect(player.play).toHaveBeenCalledWith({ samples, sampleRate: 16_000 });
  });

  it("throws when played without a player", async () => {
    const audio = new SynthesizedAudio(new Float32Array([0, 1]), 16_000);

    await expect(audio.play()).rejects.toThrow(/player/i);
  });

  describe("concat", () => {
    it("joins chunks that share a sample rate", () => {
      const first = new SynthesizedAudio(new Float32Array([0, 1]), 16_000);
      const second = new SynthesizedAudio(new Float32Array([-1]), 16_000);

      const joined = SynthesizedAudio.concat([first, second]);

      expect(Array.from(joined.samples)).toEqual([0, 1, -1]);
      expect(joined.sampleRate).toBe(16_000);
    });

    it("keeps the player of the first chunk", async () => {
      const player: AudioPlayer = { play: vi.fn(async () => {}) };
      const first = new SynthesizedAudio(new Float32Array([0]), 16_000, player);
      const second = new SynthesizedAudio(new Float32Array([1]), 16_000);

      await SynthesizedAudio.concat([first, second]).play();

      expect(player.play).toHaveBeenCalledOnce();
    });

    it("rejects an empty list", () => {
      expect(() => SynthesizedAudio.concat([])).toThrow(InvalidInputError);
    });

    it("rejects mixed sample rates", () => {
      const first = new SynthesizedAudio(new Float32Array([0]), 16_000);
      const second = new SynthesizedAudio(new Float32Array([0]), 24_000);

      expect(() => SynthesizedAudio.concat([first, second])).toThrow(/sample rate/i);
    });
  });
});
