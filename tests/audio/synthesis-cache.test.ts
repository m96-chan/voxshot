import { describe, expect, it } from "vitest";

import { SynthesisCache } from "../../src/audio/synthesis-cache.js";
import { InvalidInputError } from "../../src/errors.js";
import type { VoiceEmbedding } from "../../src/voice/types.js";

const voice = (): VoiceEmbedding => ({
  vector: Float32Array.from([1, 2, 3]),
  sampleRate: 24_000,
  createdAt: 0,
});

const samples = (value: number) => Float32Array.from([value]);

describe("SynthesisCache", () => {
  it("returns undefined for a miss", () => {
    const cache = new SynthesisCache();

    expect(cache.get(voice(), "hello", 1)).toBeUndefined();
  });

  it("returns what was stored for the same voice, text and speed", () => {
    const cache = new SynthesisCache();
    const speaker = voice();
    const audio = samples(0.5);

    cache.set(speaker, "hello", 1, audio);

    expect(cache.get(speaker, "hello", 1)).toBe(audio);
  });

  it("misses when the text differs", () => {
    const cache = new SynthesisCache();
    const speaker = voice();

    cache.set(speaker, "hello", 1, samples(1));

    expect(cache.get(speaker, "bye", 1)).toBeUndefined();
  });

  it("misses when the speed differs", () => {
    const cache = new SynthesisCache();
    const speaker = voice();

    cache.set(speaker, "hello", 1, samples(1));

    expect(cache.get(speaker, "hello", 1.5)).toBeUndefined();
  });

  it("keeps voices isolated by identity", () => {
    const cache = new SynthesisCache();
    const alice = voice();
    const bob = voice();

    cache.set(alice, "hello", 1, samples(1));

    expect(cache.get(bob, "hello", 1)).toBeUndefined();
  });

  it("evicts the least recently used entry beyond maxEntriesPerVoice", () => {
    const cache = new SynthesisCache({ maxEntriesPerVoice: 2 });
    const speaker = voice();

    cache.set(speaker, "a", 1, samples(1));
    cache.set(speaker, "b", 1, samples(2));
    cache.get(speaker, "a", 1); // refresh "a"
    cache.set(speaker, "c", 1, samples(3)); // evicts "b"

    expect(cache.get(speaker, "a", 1)).toBeDefined();
    expect(cache.get(speaker, "b", 1)).toBeUndefined();
    expect(cache.get(speaker, "c", 1)).toBeDefined();
  });

  it("overwriting an existing key does not grow the cache", () => {
    const cache = new SynthesisCache({ maxEntriesPerVoice: 2 });
    const speaker = voice();

    cache.set(speaker, "a", 1, samples(1));
    cache.set(speaker, "a", 1, samples(2));
    cache.set(speaker, "b", 1, samples(3));

    expect(cache.get(speaker, "a", 1)?.[0]).toBe(2);
    expect(cache.get(speaker, "b", 1)).toBeDefined();
  });

  it("rejects a non-positive maxEntriesPerVoice", () => {
    expect(() => new SynthesisCache({ maxEntriesPerVoice: 0 })).toThrow(InvalidInputError);
    expect(() => new SynthesisCache({ maxEntriesPerVoice: -1 })).toThrow(InvalidInputError);
  });
});

/**
 * #69: expressiveness varies per utterance, so it has to take part in the key.
 * Without it, asking for the same line at a different expressiveness would be
 * answered from the cache and the setting would look silently ignored.
 */
describe("expressiveness in the cache key", () => {
  const voice = { vector: Float32Array.from([1]), sampleRate: 24_000, createdAt: 0 };
  const audioOf = (v: number) => Float32Array.from([v]);

  it("keeps renderings at different expressiveness apart", () => {
    const cache = new SynthesisCache();

    cache.set(voice, "hello", 1, audioOf(1), 0.2);
    cache.set(voice, "hello", 1, audioOf(2), 0.9);

    expect(cache.get(voice, "hello", 1, 0.2)?.[0]).toBe(1);
    expect(cache.get(voice, "hello", 1, 0.9)?.[0]).toBe(2);
  });

  it("treats an unspecified expressiveness as its own key", () => {
    const cache = new SynthesisCache();

    cache.set(voice, "hello", 1, audioOf(1));
    cache.set(voice, "hello", 1, audioOf(2), 0.5);

    // "whatever the engine defaults to" is not knowable here, so it must not
    // collide with an explicit value that happens to match it.
    expect(cache.get(voice, "hello", 1)?.[0]).toBe(1);
    expect(cache.get(voice, "hello", 1, 0.5)?.[0]).toBe(2);
  });

  it("still returns a hit when expressiveness matches", () => {
    const cache = new SynthesisCache();

    cache.set(voice, "hello", 1, audioOf(7), 0.3);

    expect(cache.get(voice, "hello", 1, 0.3)?.[0]).toBe(7);
  });
});
