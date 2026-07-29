import "fake-indexeddb/auto";

import { afterEach, describe, expect, it, vi } from "vitest";

import { ZeroVoxError } from "../../src/errors.js";
import { IndexedDbVoiceStore } from "../../src/voice/indexeddb-store.js";
import { describeVoiceStoreContract, embedding } from "./store-contract.js";

let counter = 0;
const uniqueDatabaseName = (): string => `zerovox-test-${(counter += 1)}`;

describeVoiceStoreContract(
  "IndexedDbVoiceStore",
  () => new IndexedDbVoiceStore({ databaseName: uniqueDatabaseName() }),
);

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("IndexedDbVoiceStore", () => {
  it("persists across store instances that share a database", async () => {
    const databaseName = uniqueDatabaseName();
    const writer = new IndexedDbVoiceStore({ databaseName });
    await writer.save("alice", embedding([0.25]));
    await writer.close();

    const reader = new IndexedDbVoiceStore({ databaseName });

    expect((await reader.load("alice"))?.vector[0]).toBeCloseTo(0.25, 6);
  });

  it("keeps separate databases isolated", async () => {
    const first = new IndexedDbVoiceStore({ databaseName: uniqueDatabaseName() });
    const second = new IndexedDbVoiceStore({ databaseName: uniqueDatabaseName() });
    await first.save("alice", embedding([1]));

    expect(await second.list()).toEqual([]);
  });

  it("reopens the database after being closed", async () => {
    const store = new IndexedDbVoiceStore({ databaseName: uniqueDatabaseName() });
    await store.save("alice", embedding([1]));
    await store.close();

    expect(await store.list()).toEqual(["alice"]);
  });

  it("tolerates close before any use", async () => {
    await expect(
      new IndexedDbVoiceStore({ databaseName: uniqueDatabaseName() }).close(),
    ).resolves.toBeUndefined();
  });

  it("reports availability of the IndexedDB API", () => {
    expect(IndexedDbVoiceStore.isSupported()).toBe(true);

    vi.stubGlobal("indexedDB", undefined);

    expect(IndexedDbVoiceStore.isSupported()).toBe(false);
  });

  it("throws a ZeroVoxError when IndexedDB is unavailable", async () => {
    vi.stubGlobal("indexedDB", undefined);

    await expect(
      new IndexedDbVoiceStore({ databaseName: uniqueDatabaseName() }).list(),
    ).rejects.toBeInstanceOf(ZeroVoxError);
  });

  it("treats a null indexedDB as unsupported", () => {
    vi.stubGlobal("indexedDB", null);

    expect(IndexedDbVoiceStore.isSupported()).toBe(false);
  });

  it("surfaces a failing request as a ZeroVoxError", async () => {
    const failingRequest: Record<string, unknown> = { error: new Error("read failed") };
    const database = {
      transaction: () => ({
        objectStore: () => ({
          getAllKeys: () => {
            queueMicrotask(() => (failingRequest.onerror as () => void)?.());
            return failingRequest;
          },
        }),
      }),
      close: () => {},
    };
    vi.stubGlobal("indexedDB", {
      open: () => {
        const request: Record<string, unknown> = { result: database };
        queueMicrotask(() => (request.onsuccess as () => void)?.());
        return request;
      },
    });

    await expect(
      new IndexedDbVoiceStore({ databaseName: uniqueDatabaseName() }).list(),
    ).rejects.toBeInstanceOf(ZeroVoxError);
  });

  it("surfaces an open failure as a ZeroVoxError", async () => {
    vi.stubGlobal("indexedDB", {
      open: () => {
        const request: Record<string, unknown> = { error: new Error("quota"), result: null };
        queueMicrotask(() => (request.onerror as () => void)?.());
        return request;
      },
    });

    await expect(
      new IndexedDbVoiceStore({ databaseName: uniqueDatabaseName() }).list(),
    ).rejects.toBeInstanceOf(ZeroVoxError);
  });
});
