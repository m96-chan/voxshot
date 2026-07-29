import { describe, expect, it } from "vitest";

import { MemoryVoiceStore } from "../../src/voice/memory-store.js";
import { describeVoiceStoreContract, embedding } from "./store-contract.js";

describeVoiceStoreContract("MemoryVoiceStore", () => new MemoryVoiceStore());

describe("MemoryVoiceStore", () => {
  it("does not return a live reference to the stored vector", async () => {
    const store = new MemoryVoiceStore();
    await store.save("alice", embedding([0.5]));

    const first = await store.load("alice");
    (first as { vector: Float32Array }).vector[0] = 9;

    expect((await store.load("alice"))?.vector[0]).toBeCloseTo(0.5, 6);
  });

  it("trims surrounding whitespace from voice names", async () => {
    const store = new MemoryVoiceStore();
    await store.save("  alice  ", embedding([1]));

    expect(await store.list()).toEqual(["alice"]);
    expect(await store.load("alice")).toBeDefined();
  });
});
