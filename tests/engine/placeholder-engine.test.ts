import { beforeEach, describe, expect, it } from "vitest";

import { PlaceholderEngine, VOICE_EMBEDDING_SIZE } from "../../src/engine/placeholder-engine.js";
import { InvalidInputError } from "../../src/errors.js";
import type { PcmAudio } from "../../src/platform.js";
import type { VoiceEmbedding } from "../../src/voice/types.js";

function tone(frequency: number, seconds = 0.5, sampleRate = 16_000, amplitude = 0.8): PcmAudio {
  const samples = new Float32Array(Math.round(seconds * sampleRate));
  for (let index = 0; index < samples.length; index += 1) {
    samples[index] = amplitude * Math.sin((2 * Math.PI * frequency * index) / sampleRate);
  }
  return { samples, sampleRate };
}

function voice(vector: Float32Array): VoiceEmbedding {
  return { vector, sampleRate: 16_000, createdAt: 0 };
}

describe("PlaceholderEngine", () => {
  let engine: PlaceholderEngine;

  beforeEach(() => {
    engine = new PlaceholderEngine();
  });

  describe("metadata", () => {
    it("names itself and reports its output sample rate", () => {
      expect(engine.name).toBe("placeholder");
      expect(engine.sampleRate).toBe(24_000);
    });

    it("accepts a custom output sample rate", () => {
      expect(new PlaceholderEngine({ sampleRate: 16_000 }).sampleRate).toBe(16_000);
    });

    it("rejects a non positive sample rate", () => {
      expect(() => new PlaceholderEngine({ sampleRate: 0 })).toThrow(InvalidInputError);
    });
  });

  describe("load", () => {
    it("loads on wasm and remembers the device", async () => {
      await engine.load("wasm");

      expect(engine.device).toBe("wasm");
    });

    it("loads on webgpu", async () => {
      await engine.load("webgpu");

      expect(engine.device).toBe("webgpu");
    });

    it("is idempotent", async () => {
      await engine.load("wasm");
      await engine.load("wasm");

      expect(engine.device).toBe("wasm");
    });
  });

  describe("embed", () => {
    it("produces a fixed size vector", async () => {
      const vector = await engine.embed(tone(220));

      expect(vector).toHaveLength(VOICE_EMBEDDING_SIZE);
    });

    it("is deterministic for identical audio", async () => {
      const first = await engine.embed(tone(220));
      const second = await engine.embed(tone(220));

      expect(Array.from(first)).toEqual(Array.from(second));
    });

    it("distinguishes a low pitched from a high pitched reference", async () => {
      const low = await engine.embed(tone(110));
      const high = await engine.embed(tone(440));

      expect(high[0]).toBeGreaterThan(low[0] as number);
    });

    it("distinguishes a quiet from a loud reference", async () => {
      const quiet = await engine.embed(tone(220, 0.5, 16_000, 0.1));
      const loud = await engine.embed(tone(220, 0.5, 16_000, 0.9));

      expect(Array.from(quiet)).not.toEqual(Array.from(loud));
    });

    it("produces finite values only", async () => {
      for (const value of await engine.embed(tone(220))) {
        expect(Number.isFinite(value)).toBe(true);
      }
    });

    it("rejects empty reference audio", async () => {
      await expect(
        engine.embed({ samples: new Float32Array(0), sampleRate: 16_000 }),
      ).rejects.toBeInstanceOf(InvalidInputError);
    });

    it("rejects reference audio that is entirely silent", async () => {
      await expect(
        engine.embed({ samples: new Float32Array(1_000), sampleRate: 16_000 }),
      ).rejects.toBeInstanceOf(InvalidInputError);
    });
  });

  describe("synthesize", () => {
    let speaker: VoiceEmbedding;

    beforeEach(async () => {
      speaker = voice(await engine.embed(tone(220)));
    });

    it("returns audio roughly proportional to the text length", async () => {
      const short = await engine.synthesize({ text: "aa", voice: speaker, speed: 1 });
      const long = await engine.synthesize({ text: "aaaaaaaa", voice: speaker, speed: 1 });

      expect(long.length).toBeGreaterThan(short.length * 3);
    });

    it("is deterministic", async () => {
      const first = await engine.synthesize({ text: "hello", voice: speaker, speed: 1 });
      const second = await engine.synthesize({ text: "hello", voice: speaker, speed: 1 });

      expect(Array.from(first)).toEqual(Array.from(second));
    });

    it("renders different text differently", async () => {
      const first = await engine.synthesize({ text: "hello", voice: speaker, speed: 1 });
      const second = await engine.synthesize({ text: "world", voice: speaker, speed: 1 });

      expect(Array.from(first)).not.toEqual(Array.from(second));
    });

    it("renders the same text differently for a different voice", async () => {
      const other = voice(await engine.embed(tone(440)));

      const first = await engine.synthesize({ text: "hello", voice: speaker, speed: 1 });
      const second = await engine.synthesize({ text: "hello", voice: other, speed: 1 });

      expect(Array.from(first)).not.toEqual(Array.from(second));
    });

    it("shortens the audio as speed increases", async () => {
      const normal = await engine.synthesize({ text: "hello there", voice: speaker, speed: 1 });
      const fast = await engine.synthesize({ text: "hello there", voice: speaker, speed: 2 });

      expect(fast.length).toBeCloseTo(normal.length / 2, -2);
    });

    it("stays inside the -1..1 range", async () => {
      for (const sample of await engine.synthesize({
        text: "hello there, this is VoxShot.",
        voice: speaker,
        speed: 1,
      })) {
        expect(Math.abs(sample)).toBeLessThanOrEqual(1);
      }
    });

    it("produces finite samples only", async () => {
      for (const sample of await engine.synthesize({ text: "hi", voice: speaker, speed: 1 })) {
        expect(Number.isFinite(sample)).toBe(true);
      }
    });

    it("returns nothing for empty text", async () => {
      const output = await engine.synthesize({ text: "", voice: speaker, speed: 1 });

      expect(output).toHaveLength(0);
    });

    it("renders silence for whitespace", async () => {
      const output = await engine.synthesize({ text: "   ", voice: speaker, speed: 1 });

      expect(output.length).toBeGreaterThan(0);
      expect(Math.max(...output)).toBe(0);
    });

    it("rejects a speed outside the supported range", async () => {
      await expect(
        engine.synthesize({ text: "hi", voice: speaker, speed: 0 }),
      ).rejects.toBeInstanceOf(InvalidInputError);
      await expect(
        engine.synthesize({ text: "hi", voice: speaker, speed: 10 }),
      ).rejects.toBeInstanceOf(InvalidInputError);
    });

    it("rejects an embedding of the wrong size", async () => {
      await expect(
        engine.synthesize({ text: "hi", voice: voice(new Float32Array(2)), speed: 1 }),
      ).rejects.toBeInstanceOf(InvalidInputError);
    });
  });

  describe("dispose", () => {
    it("forgets the loaded device", async () => {
      await engine.load("webgpu");

      await engine.dispose();

      expect(engine.device).toBeUndefined();
    });
  });
});
