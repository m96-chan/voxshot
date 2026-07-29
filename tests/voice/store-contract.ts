import { describe, expect, it } from "vitest";

import { InvalidInputError } from "../../src/errors.js";
import type { VoiceEmbedding, VoiceStore } from "../../src/voice/types.js";
import { toArray } from "../helpers/tensor.js";

export function embedding(values: number[], sampleRate = 16_000): VoiceEmbedding {
  return { vector: Float32Array.from(values), sampleRate, createdAt: 1_700_000_000_000 };
}

/**
 * Behaviour every {@link VoiceStore} implementation must share, so the
 * in-memory and IndexedDB stores stay interchangeable.
 */
export function describeVoiceStoreContract(name: string, createStore: () => VoiceStore): void {
  describe(`${name} (VoiceStore contract)`, () => {
    it("returns undefined for an unknown voice", async () => {
      await expect(createStore().load("nobody")).resolves.toBeUndefined();
    });

    it("round trips a saved voice", async () => {
      const store = createStore();
      await store.save("alice", embedding([0.1, 0.2, 0.3], 24_000));

      const loaded = await store.load("alice");

      expect(loaded?.sampleRate).toBe(24_000);
      expect(loaded?.createdAt).toBe(1_700_000_000_000);
      expect(Array.from(loaded?.vector ?? [])).toEqual([
        expect.closeTo(0.1, 6),
        expect.closeTo(0.2, 6),
        expect.closeTo(0.3, 6),
      ]);
    });

    it("round trips engine specific tensors", async () => {
      const store = createStore();
      await store.save("alice", {
        ...embedding([0.1]),
        engine: "chatterbox",
        tensors: {
          speaker_embeddings: {
            type: "float32",
            dims: [1, 2],
            data: Float32Array.from([0.25, -0.75]),
          },
          audio_tokens: { type: "int64", dims: [1, 2], data: BigInt64Array.from([7n, 9n]) },
        },
      });

      const loaded = await store.load("alice");

      expect(loaded?.engine).toBe("chatterbox");
      expect(loaded?.tensors?.speaker_embeddings?.dims).toEqual([1, 2]);
      expect(toArray(loaded?.tensors?.speaker_embeddings?.data)).toEqual([
        expect.closeTo(0.25, 6),
        expect.closeTo(-0.75, 6),
      ]);
      expect(loaded?.tensors?.audio_tokens?.type).toBe("int64");
      expect(toArray(loaded?.tensors?.audio_tokens?.data)).toEqual([7n, 9n]);
    });

    it("does not alias engine specific tensors", async () => {
      const store = createStore();
      const data = Float32Array.from([0.5]);
      await store.save("alice", {
        ...embedding([0.1]),
        tensors: { speaker_embeddings: { type: "float32", dims: [1], data } },
      });

      data[0] = 9;

      const loaded = await store.load("alice");
      expect(loaded?.tensors?.speaker_embeddings?.data[0]).toBeCloseTo(0.5, 6);
    });

    it("overwrites a voice saved under the same name", async () => {
      const store = createStore();
      await store.save("alice", embedding([1]));
      await store.save("alice", embedding([2]));

      expect(Array.from((await store.load("alice"))?.vector ?? [])).toEqual([2]);
      expect(await store.list()).toEqual(["alice"]);
    });

    it("lists saved voices in alphabetical order", async () => {
      const store = createStore();
      await store.save("carol", embedding([1]));
      await store.save("alice", embedding([1]));
      await store.save("bob", embedding([1]));

      expect(await store.list()).toEqual(["alice", "bob", "carol"]);
    });

    it("lists nothing when empty", async () => {
      await expect(createStore().list()).resolves.toEqual([]);
    });

    it("reports whether a delete removed anything", async () => {
      const store = createStore();
      await store.save("alice", embedding([1]));

      await expect(store.delete("alice")).resolves.toBe(true);
      await expect(store.delete("alice")).resolves.toBe(false);
      await expect(store.load("alice")).resolves.toBeUndefined();
    });

    it("removes every voice on clear", async () => {
      const store = createStore();
      await store.save("alice", embedding([1]));
      await store.save("bob", embedding([1]));

      await store.clear();

      expect(await store.list()).toEqual([]);
    });

    it("does not alias the caller's vector", async () => {
      const store = createStore();
      const original = embedding([0.5]);
      await store.save("alice", original);

      original.vector[0] = 9;

      expect((await store.load("alice"))?.vector[0]).toBeCloseTo(0.5, 6);
    });

    it("rejects a blank voice name", async () => {
      const store = createStore();

      await expect(store.save("  ", embedding([1]))).rejects.toBeInstanceOf(InvalidInputError);
      await expect(store.load("")).rejects.toBeInstanceOf(InvalidInputError);
      await expect(store.delete("")).rejects.toBeInstanceOf(InvalidInputError);
    });

    it("rejects an empty embedding vector", async () => {
      await expect(createStore().save("alice", embedding([]))).rejects.toBeInstanceOf(
        InvalidInputError,
      );
    });
  });
}
