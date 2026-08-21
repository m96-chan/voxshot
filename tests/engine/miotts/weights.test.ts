import { describe, expect, it, vi } from "vitest";

import {
  MIOTTS_WEIGHT_PARTS,
  loadPart,
  readJsonPart,
  type MioTtsWeightPart,
  type MioTtsWeightSource,
} from "../../../src/engine/miotts/weights.js";
import { InvalidInputError } from "../../../src/errors.js";

/** A source that answers every part from a fixed table. */
function sourceOf(table: Partial<Record<MioTtsWeightPart, ArrayBuffer | ArrayBufferView>>) {
  return {
    load: vi.fn(async (part: MioTtsWeightPart) => {
      const bytes = table[part];
      if (!bytes) throw new Error(`no fixture for ${part}`);
      return bytes;
    }),
  } satisfies MioTtsWeightSource;
}

function bytesOf(...values: number[]): ArrayBuffer {
  return Uint8Array.from(values).buffer;
}

function jsonBytes(value: unknown): ArrayBuffer {
  return new TextEncoder().encode(JSON.stringify(value)).buffer as ArrayBuffer;
}

describe("MIOTTS_WEIGHT_PARTS", () => {
  it("names every part the engine can ask for", () => {
    // The list is public API: a caller writes a switch over it to map part
    // names onto URLs, so it is spelled out here rather than derived.
    expect([...MIOTTS_WEIGHT_PARTS]).toEqual([
      "tokenizer",
      "lm-manifest",
      "lm-codes",
      "lm-scales",
      "lm-norms",
      "codec-decoder",
      "codec-encoder",
    ]);
  });
});

describe("loadPart", () => {
  it("asks the source for the named part and hands back its bytes", async () => {
    const source = sourceOf({ "lm-codes": bytesOf(1, 2, 3) });

    const buffer = await loadPart(source, "lm-codes");

    expect(source.load).toHaveBeenCalledWith("lm-codes");
    expect(new Uint8Array(buffer)).toEqual(Uint8Array.from([1, 2, 3]));
  });

  it("accepts a view and copies out exactly its window", async () => {
    // `readFileSync` hands back a Buffer, which is a Uint8Array over a pooled
    // ArrayBuffer that is usually far larger than the file. Taking `.buffer`
    // naively would read a neighbouring file's bytes as weights.
    const pool = Uint8Array.from([9, 9, 1, 2, 3, 9]);
    const source = sourceOf({ "lm-norms": pool.subarray(2, 5) });

    const buffer = await loadPart(source, "lm-norms");

    expect(new Uint8Array(buffer)).toEqual(Uint8Array.from([1, 2, 3]));
  });

  it("reports the part when the source rejects, keeping the original as the cause", async () => {
    const failure = new Error("404");
    const source: MioTtsWeightSource = { load: async () => Promise.reject(failure) };

    const error = await loadPart(source, "codec-encoder").catch((thrown: unknown) => thrown);

    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message).toContain("codec-encoder");
    expect((error as Error).cause).toBe(failure);
  });

  it("rejects a source that answers with something other than bytes", async () => {
    // The likely caller bug is returning the `Response` instead of awaiting
    // `.arrayBuffer()`; failing here names the part and the type, rather than
    // surfacing later as a safetensors parse error on garbage.
    const source = { load: async () => "not bytes" } as unknown as MioTtsWeightSource;

    await expect(loadPart(source, "tokenizer")).rejects.toThrow(InvalidInputError);
    await expect(loadPart(source, "tokenizer")).rejects.toThrow(/tokenizer/);
  });

  it("rejects a source whose load forgot to return anything", async () => {
    // A `load` written as a switch with a missing branch resolves to
    // `undefined`; without this it would surface as a TypeError deep inside a
    // parser instead of naming the part that was never answered.
    const source = { load: async () => undefined } as unknown as MioTtsWeightSource;

    await expect(loadPart(source, "lm-scales")).rejects.toThrow(/lm-scales/);
  });

  it("rejects a source that answers with null", async () => {
    // A cache lookup handed straight through resolves to null on a miss.
    const source = { load: async () => null } as unknown as MioTtsWeightSource;

    await expect(loadPart(source, "codec-decoder")).rejects.toThrow(/null/);
  });

  it("rejects a source that answers with a parsed object", async () => {
    // The manifest is the one part a caller is tempted to hand over already
    // parsed, but the engine reads it from bytes like everything else.
    const source = { load: async () => ({ ropePermuted: true }) } as unknown as MioTtsWeightSource;

    await expect(loadPart(source, "lm-manifest")).rejects.toThrow(InvalidInputError);
  });
});

describe("readJsonPart", () => {
  it("decodes a part as UTF-8 JSON", async () => {
    const source = sourceOf({ "lm-manifest": jsonBytes({ ropePermuted: true }) });

    await expect(readJsonPart(source, "lm-manifest")).resolves.toEqual({ ropePermuted: true });
  });

  it("names the part when the bytes are not JSON", async () => {
    const source = sourceOf({ tokenizer: bytesOf(0x7b, 0x7b, 0x7b) });

    const error = await readJsonPart(source, "tokenizer").catch((thrown: unknown) => thrown);

    expect(error).toBeInstanceOf(InvalidInputError);
    expect((error as Error).message).toContain("tokenizer");
  });
});
