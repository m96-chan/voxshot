import { describe, expect, it, vi } from "vitest";

import { MODEL_BASE, modelFiles, prewarmModelCache } from "../src/model-cache.js";

function fakeCache(overrides: Partial<Cache> = {}): Cache {
  return {
    match: vi.fn(async () => undefined),
    put: vi.fn(async () => undefined),
    ...overrides,
  } as unknown as Cache;
}

function fakeStorage(cache: Cache): CacheStorage {
  return { open: vi.fn(async () => cache) } as unknown as CacheStorage;
}

function okResponse(): Response {
  return new Response("data", { status: 200 });
}

describe("modelFiles", () => {
  it("includes the language model matching the requested dtype", () => {
    expect(modelFiles("q4f16")).toContain("onnx/language_model_q4f16.onnx");
    expect(modelFiles("q4")).toContain("onnx/language_model_q4.onnx");
    expect(modelFiles("q4")).not.toContain("onnx/language_model_q4f16.onnx");
  });

  it("includes the shared config and onnx session files", () => {
    const files = modelFiles("q4");
    expect(files).toContain("config.json");
    expect(files).toContain("tokenizer.json");
    expect(files).toContain("onnx/speech_encoder.onnx");
  });
});

describe("prewarmModelCache", () => {
  it("fetches every missing file with no-store and stores it in the cache", async () => {
    const cache = fakeCache();
    const fetchFn = vi.fn(async () => okResponse()) as unknown as typeof fetch;
    const warmed = await prewarmModelCache({
      dtype: "q4",
      log: () => {},
      cacheStorage: fakeStorage(cache),
      fetchFn,
    });
    expect(warmed).toBe(true);
    expect(fetchFn).toHaveBeenCalledTimes(modelFiles("q4").length);
    expect(fetchFn).toHaveBeenCalledWith(`${MODEL_BASE}config.json`, { cache: "no-store" });
    expect(cache.put).toHaveBeenCalledTimes(modelFiles("q4").length);
  });

  it("skips files that are already cached", async () => {
    const cache = fakeCache({ match: vi.fn(async () => okResponse()) });
    const fetchFn = vi.fn() as unknown as typeof fetch;
    const warmed = await prewarmModelCache({
      dtype: "q4",
      log: () => {},
      cacheStorage: fakeStorage(cache),
      fetchFn,
    });
    expect(warmed).toBe(true);
    expect(fetchFn).not.toHaveBeenCalled();
    expect(cache.put).not.toHaveBeenCalled();
  });

  it("returns false with a warning instead of throwing when caches.open fails", async () => {
    const storage = {
      open: vi.fn(async () => {
        throw new Error("Unexpected internal error");
      }),
    } as unknown as CacheStorage;
    const logs: string[] = [];
    const fetchFn = vi.fn() as unknown as typeof fetch;
    const warmed = await prewarmModelCache({
      dtype: "q4",
      log: (message) => logs.push(message),
      cacheStorage: storage,
      fetchFn,
    });
    expect(warmed).toBe(false);
    expect(fetchFn).not.toHaveBeenCalled();
    expect(logs.join("\n")).toContain("Unexpected internal error");
  });

  it("returns false with a warning when the Cache API is unavailable", async () => {
    const logs: string[] = [];
    const warmed = await prewarmModelCache({
      dtype: "q4",
      log: (message) => logs.push(message),
      cacheStorage: undefined,
      fetchFn: vi.fn() as unknown as typeof fetch,
    });
    expect(warmed).toBe(false);
    expect(logs.length).toBeGreaterThan(0);
  });

  it("stops prewarming and returns false when cache.put fails", async () => {
    const cache = fakeCache({
      put: vi.fn(async () => {
        throw new Error("Quota exceeded");
      }),
    });
    const logs: string[] = [];
    const fetchFn = vi.fn(async () => okResponse()) as unknown as typeof fetch;
    const warmed = await prewarmModelCache({
      dtype: "q4",
      log: (message) => logs.push(message),
      cacheStorage: fakeStorage(cache),
      fetchFn,
    });
    expect(warmed).toBe(false);
    expect(fetchFn).toHaveBeenCalledTimes(1);
    expect(logs.join("\n")).toContain("Quota exceeded");
  });

  it("treats a failing cache.match as a cache miss and keeps going", async () => {
    const cache = fakeCache({
      match: vi.fn(async () => {
        throw new Error("Unexpected internal error");
      }),
    });
    const fetchFn = vi.fn(async () => okResponse()) as unknown as typeof fetch;
    const warmed = await prewarmModelCache({
      dtype: "q4",
      log: () => {},
      cacheStorage: fakeStorage(cache),
      fetchFn,
    });
    expect(warmed).toBe(true);
    expect(fetchFn).toHaveBeenCalledTimes(modelFiles("q4").length);
  });

  it("still throws on HTTP errors: a failed download is not a cache problem", async () => {
    const cache = fakeCache();
    const fetchFn = vi.fn(async () => new Response("nope", { status: 404 })) as unknown as typeof fetch;
    await expect(
      prewarmModelCache({
        dtype: "q4",
        log: () => {},
        cacheStorage: fakeStorage(cache),
        fetchFn,
      }),
    ).rejects.toThrow(/HTTP 404/);
  });

  it("reports non-Error throw values in the warning", async () => {
    const storage = {
      open: vi.fn(async () => {
        throw "boom";
      }),
    } as unknown as CacheStorage;
    const logs: string[] = [];
    const warmed = await prewarmModelCache({
      dtype: "q4",
      log: (message) => logs.push(message),
      cacheStorage: storage,
      fetchFn: vi.fn() as unknown as typeof fetch,
    });
    expect(warmed).toBe(false);
    expect(logs.join("\n")).toContain("boom");
  });

  it("defaults to the global fetch when no fetchFn is injected", async () => {
    // cacheStorage: undefined bails out before any network access, so the
    // default is safe to take in a unit test.
    const warmed = await prewarmModelCache({
      dtype: "q4",
      log: () => {},
      cacheStorage: undefined,
    });
    expect(warmed).toBe(false);
  });

  it("stores body-less responses as-is without progress logging", async () => {
    const cache = fakeCache();
    const fetchFn = vi.fn(
      async () => new Response(null, { status: 200, headers: { "Content-Length": "8" } }),
    ) as unknown as typeof fetch;
    const logs: string[] = [];
    const warmed = await prewarmModelCache({
      dtype: "q4",
      log: (message) => logs.push(message),
      cacheStorage: fakeStorage(cache),
      fetchFn,
    });
    expect(warmed).toBe(true);
    expect(logs.join("\n")).not.toContain("%");
  });

  it("logs download progress for responses that declare a Content-Length", async () => {
    const cache = fakeCache();
    const body = new Blob(["x".repeat(64)]);
    const fetchFn = vi.fn(
      async () =>
        new Response(body, { status: 200, headers: { "Content-Length": "64" } }),
    ) as unknown as typeof fetch;
    const logs: string[] = [];
    const warmed = await prewarmModelCache({
      dtype: "q4",
      log: (message) => logs.push(message),
      cacheStorage: fakeStorage(cache),
      fetchFn,
    });
    expect(warmed).toBe(true);
    expect(logs.join("\n")).toContain("config.json: 100%");
  });
});
