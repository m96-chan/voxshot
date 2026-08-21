import { describe, expect, it, vi } from "vitest";

import {
  CLONING_PARTS,
  MIOTTS_ASSET_BYTES,
  MIOTTS_CACHE_NAME,
  SPEECH_PARTS,
  assetUrl,
  createHuggingFaceWeightSource,
  estimateDownload,
} from "../src/miotts-weights.js";

/**
 * The reference weight source.
 *
 * It is an example, not library code, and it is still worth testing: the
 * posture it demonstrates — a broken cache costs a re-download, never the page
 * — is the part somebody copying this file should be able to rely on, and it is
 * exactly the part that is easy to get wrong by letting a `CacheStorage`
 * rejection escape.
 */

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

const bytesResponse = (size = 8) =>
  new Response(new Uint8Array(size), { status: 200, headers: { "content-length": String(size) } });

describe("assetUrl", () => {
  it("resolves every part the engine can ask for", () => {
    for (const part of [...SPEECH_PARTS, ...CLONING_PARTS]) {
      expect(assetUrl(part)).toMatch(/^https:\/\/huggingface\.co\//);
    }
  });

  it("keeps the derived artifacts separate from upstream's", () => {
    // The int8 model and the voice encoder are produced by this repository's
    // scripts; the tokenizer and the codec are upstream's, verbatim.
    expect(assetUrl("tokenizer")).toContain("Aratako/MioTTS-0.6B");
    expect(assetUrl("codec-decoder")).toContain("Aratako/MioCodec");
    expect(assetUrl("lm-codes")).toContain("m96-chan/MioTTS-0.6B-q8-webgpu");
    expect(assetUrl("codec-encoder")).toContain("m96-chan/MioCodec-encoder-webgpu");
  });
});

describe("the parts split", () => {
  it("keeps the voice encoder out of what text-to-speech needs", () => {
    // The whole reason the engine asks part by part: 117 MB nobody who uses a
    // saved voice should pay for.
    expect(SPEECH_PARTS).not.toContain("codec-encoder");
    expect(CLONING_PARTS).toEqual(["codec-encoder"]);
  });

  it("covers every part between them", () => {
    expect([...SPEECH_PARTS, ...CLONING_PARTS].sort()).toEqual(
      Object.keys(MIOTTS_ASSET_BYTES).sort(),
    );
  });
});

describe("createHuggingFaceWeightSource", () => {
  it("fetches a part and hands back its bytes", async () => {
    const fetchFn = vi.fn(async () => bytesResponse(16));
    const source = createHuggingFaceWeightSource({ fetchFn: fetchFn as unknown as typeof fetch });

    const buffer = await source.load("lm-norms");

    expect(buffer.byteLength).toBe(16);
    expect(fetchFn).toHaveBeenCalledWith(assetUrl("lm-norms"));
  });

  it("stores what it fetched under the part's URL", async () => {
    const cache = fakeCache();
    const source = createHuggingFaceWeightSource({
      cacheStorage: fakeStorage(cache),
      fetchFn: (async () => bytesResponse()) as unknown as typeof fetch,
    });

    await source.load("lm-scales");

    expect(cache.put).toHaveBeenCalledWith(assetUrl("lm-scales"), expect.anything());
  });

  it("serves a cached part without touching the network", async () => {
    // The point of the cache: a second visit costs nothing.
    const cache = fakeCache({ match: vi.fn(async () => bytesResponse(32)) });
    const fetchFn = vi.fn(async () => bytesResponse());
    const source = createHuggingFaceWeightSource({
      cacheStorage: fakeStorage(cache),
      fetchFn: fetchFn as unknown as typeof fetch,
    });

    const buffer = await source.load("codec-decoder");

    expect(buffer.byteLength).toBe(32);
    expect(fetchFn).not.toHaveBeenCalled();
  });

  it("uses the name that will not collide with Transformers.js' bucket", async () => {
    // Sharing a cache would let either demo's eviction take out the other's
    // gigabyte.
    const storage = fakeStorage(fakeCache());
    const source = createHuggingFaceWeightSource({
      cacheStorage: storage,
      fetchFn: (async () => bytesResponse()) as unknown as typeof fetch,
    });

    await source.load("tokenizer");

    expect(storage.open).toHaveBeenCalledWith(MIOTTS_CACHE_NAME);
  });

  it("downloads anyway when the cache cannot be opened", async () => {
    // A broken cache costs a re-download, not the page.
    const storage = {
      open: vi.fn(async () => {
        throw new Error("quota");
      }),
    } as unknown as CacheStorage;
    const warnings: string[] = [];
    const source = createHuggingFaceWeightSource({
      cacheStorage: storage,
      fetchFn: (async () => bytesResponse(4)) as unknown as typeof fetch,
      onCacheProblem: (message) => warnings.push(message),
    });

    await expect(source.load("lm-manifest")).resolves.toHaveProperty("byteLength", 4);
    expect(warnings).toHaveLength(1);
  });

  it("downloads anyway when reading from the cache throws", async () => {
    const cache = fakeCache({
      match: vi.fn(async () => {
        throw new Error("corrupt");
      }),
    });
    const warnings: string[] = [];
    const source = createHuggingFaceWeightSource({
      cacheStorage: fakeStorage(cache),
      fetchFn: (async () => bytesResponse(4)) as unknown as typeof fetch,
      onCacheProblem: (message) => warnings.push(message),
    });

    await expect(source.load("lm-codes")).resolves.toHaveProperty("byteLength", 4);
    expect(warnings[0]).toContain("lm-codes");
  });

  it("returns the bytes even when storing them fails", async () => {
    // Out of quota is the common case, and it must not fail a load that has
    // already succeeded.
    const cache = fakeCache({
      put: vi.fn(async () => {
        throw new Error("QuotaExceededError");
      }),
    });
    const warnings: string[] = [];
    const source = createHuggingFaceWeightSource({
      cacheStorage: fakeStorage(cache),
      fetchFn: (async () => bytesResponse(4)) as unknown as typeof fetch,
      onCacheProblem: (message) => warnings.push(message),
    });

    await expect(source.load("lm-codes")).resolves.toHaveProperty("byteLength", 4);
    expect(warnings).toHaveLength(1);
  });

  it("reports a cache problem to the console when nothing else is listening", async () => {
    // The default has to be a report, not silence: a cache that quietly stopped
    // working looks exactly like a slow network, and the visitor re-downloads a
    // gigabyte every visit without ever being told why.
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const storage = {
      open: vi.fn(async () => {
        throw new Error("quota");
      }),
    } as unknown as CacheStorage;

    await createHuggingFaceWeightSource({
      cacheStorage: storage,
      fetchFn: (async () => bytesResponse(4)) as unknown as typeof fetch,
    }).load("lm-manifest");

    expect(warn).toHaveBeenCalledWith(expect.stringContaining("could not be opened"));
    warn.mockRestore();
  });

  it("reports the failing status rather than a parse error later", async () => {
    const source = createHuggingFaceWeightSource({
      fetchFn: (async () => new Response("nope", { status: 404, statusText: "Not Found" })) as unknown as typeof fetch,
    });

    await expect(source.load("codec-encoder")).rejects.toThrow(/404 Not Found/);
  });

  it("reports progress against the published size while the body arrives", async () => {
    // 1.1 GB with no feedback reads as a hung page. The declared size is used
    // when the server gives one, and the published size when it does not.
    const seen: { loaded: number; total?: number }[] = [];
    const source = createHuggingFaceWeightSource({
      fetchFn: (async () => new Response(new Uint8Array(64), { status: 200 })) as unknown as typeof fetch,
      onProgress: (_part, detail) => seen.push(detail),
    });

    await source.load("lm-codes");

    expect(seen.length).toBeGreaterThan(0);
    expect(seen.at(-1)?.loaded).toBe(64);
    expect(seen.at(-1)?.total).toBe(MIOTTS_ASSET_BYTES["lm-codes"]);
  });
});

describe("estimateDownload", () => {
  it("adds up the parts asked about", async () => {
    const estimate = await estimateDownload(SPEECH_PARTS, undefined);

    expect(estimate.bytes).toBe(
      SPEECH_PARTS.reduce((sum, part) => sum + MIOTTS_ASSET_BYTES[part], 0),
    );
  });

  it("says nothing persists where there is no Cache API", async () => {
    const estimate = await estimateDownload(SPEECH_PARTS, undefined);

    expect(estimate.cacheAvailable).toBe(false);
    expect(estimate.pending).toBe(estimate.bytes);
  });

  it("counts only what is not already cached", async () => {
    const cached = new Set([assetUrl("tokenizer"), assetUrl("lm-manifest")]);
    const cache = fakeCache({
      match: vi.fn(async (request) => (cached.has(String(request)) ? bytesResponse() : undefined)),
    });

    const estimate = await estimateDownload(SPEECH_PARTS, fakeStorage(cache));

    expect(estimate.pending).toBe(
      estimate.bytes - MIOTTS_ASSET_BYTES.tokenizer - MIOTTS_ASSET_BYTES["lm-manifest"],
    );
  });

  it("reports everything as pending when the cache cannot be read", async () => {
    // Better to overstate the download than to promise one that is not there.
    const cache = fakeCache({
      match: vi.fn(async () => {
        throw new Error("corrupt");
      }),
    });
    const warnings: string[] = [];

    const estimate = await estimateDownload(SPEECH_PARTS, fakeStorage(cache), (message) =>
      warnings.push(message),
    );

    expect(estimate.pending).toBe(estimate.bytes);
    expect(warnings).toHaveLength(SPEECH_PARTS.length);
  });

  it("reports its own cache problems to the console by default", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const storage = {
      open: vi.fn(async () => {
        throw new Error("quota");
      }),
    } as unknown as CacheStorage;

    const estimate = await estimateDownload(["tokenizer"], storage);

    expect(estimate.cacheAvailable).toBe(false);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it("never fetches", async () => {
    // A pre-flight that downloaded would defeat its own purpose.
    const cache = fakeCache();
    await estimateDownload(SPEECH_PARTS, fakeStorage(cache));

    expect(cache.put).not.toHaveBeenCalled();
  });
});
