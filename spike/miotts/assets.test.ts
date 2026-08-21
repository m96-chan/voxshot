import { describe, expect, it, vi } from "vitest";

import {
  ASSET_CACHE_NAME,
  assetModeFromSearch,
  estimateDownloads,
  fetchAsset,
  resolveAssets,
  type Asset,
} from "./assets";

/**
 * A CacheStorage stand-in. `put()` takes ownership of the response the way the
 * real one does, so `match()` hands back a clone — a test that read the same
 * body twice would otherwise pass against a cache that cannot.
 */
class FakeCache {
  readonly entries = new Map<string, Response>();
  readonly puts: string[] = [];

  async match(url: string): Promise<Response | undefined> {
    return this.entries.get(url)?.clone();
  }

  async put(url: string, response: Response): Promise<void> {
    this.puts.push(url);
    this.entries.set(url, response);
  }
}

function fakeStorage(cache: FakeCache): CacheStorage {
  return { open: async (name: string) => (name === ASSET_CACHE_NAME ? cache : new FakeCache()) } as unknown as CacheStorage;
}

/** A response whose body arrives in several chunks, so progress has something to report. */
function chunked(bytes: number, chunks = 4): Response {
  const size = Math.ceil(bytes / chunks);
  let sent = 0;
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (sent >= bytes) {
        controller.close();
        return;
      }
      const n = Math.min(size, bytes - sent);
      sent += n;
      controller.enqueue(new Uint8Array(n));
    },
  });
  return new Response(body, { headers: { "content-length": String(bytes), "content-type": "application/octet-stream" } });
}

describe("asset mode", () => {
  it("defaults to the Hugging Face URLs", () => {
    expect(assetModeFromSearch("")).toBe("hf");
    expect(assetModeFromSearch("?voice=x")).toBe("hf");
  });

  it("switches to same-origin assets for ?assets=local", () => {
    expect(assetModeFromSearch("?assets=local")).toBe("local");
    expect(assetModeFromSearch("?assets=local&mode=greedy")).toBe("local");
  });

  it("accepts an explicit ?assets=hf", () => {
    expect(assetModeFromSearch("?assets=hf")).toBe("hf");
  });

  it("refuses a value it does not know instead of silently downloading a gigabyte", () => {
    expect(() => assetModeFromSearch("?assets=locl")).toThrow(/assets=locl/);
  });
});

describe("the URL table", () => {
  const hf = resolveAssets("hf");
  const local = resolveAssets("local");

  it("points the default mode at Hugging Face", () => {
    expect(hf.tokenizer.url).toBe("https://huggingface.co/Aratako/MioTTS-0.6B/resolve/main/tokenizer.json");
    expect(hf.q8Manifest.url).toBe(
      "https://huggingface.co/m96-chan/MioTTS-0.6B-q8-webgpu/resolve/main/manifest.json",
    );
    expect(hf.q8Bin("weights.codes.bin").url).toBe(
      "https://huggingface.co/m96-chan/MioTTS-0.6B-q8-webgpu/resolve/main/weights.codes.bin",
    );
    expect(hf.encoderWeights.url).toBe(
      "https://huggingface.co/m96-chan/MioCodec-encoder-webgpu/resolve/main/encoder-weights.safetensors",
    );
  });

  it("points ?assets=local at the paths serve.mjs maps", () => {
    expect(local.tokenizer.url).toBe("./miotts/tokenizer.json");
    expect(local.q8Manifest.url).toBe("./miotts/q8/manifest.json");
    expect(local.q8Bin("weights.codes.bin").url).toBe("./miotts/q8/weights.codes.bin");
    expect(local.encoderWeights.url).toBe("./miotts/encoder-weights.safetensors");
  });

  it("keeps the MioCodec checkpoint on its canonical HF URL in BOTH modes", () => {
    // check-tts.mjs routes `https://huggingface.co/Aratako/MioCodec-25Hz-24kHz/**`
    // to serve.mjs's local copy with a 302. Moving this URL in local mode would
    // silently pull 523 MB from the CDN on every check run.
    const url = "https://huggingface.co/Aratako/MioCodec-25Hz-24kHz/resolve/main/model.safetensors";
    expect(hf.codecCheckpoint.url).toBe(url);
    expect(local.codecCheckpoint.url).toBe(url);
  });

  it("keeps the speaker fixture same-origin — it ships with the page", () => {
    expect(hf.fixture.url).toBe("./mio-codec-fixture.json");
    expect(local.fixture.url).toBe("./mio-codec-fixture.json");
  });

  it("caches the big downloads in hf mode and nothing in local mode", () => {
    expect([hf.tokenizer, hf.q8Bin("weights.codes.bin"), hf.codecCheckpoint, hf.encoderWeights].map((a) => a.cache))
      .toEqual([true, true, true, true]);
    expect(
      [local.tokenizer, local.q8Bin("weights.codes.bin"), local.codecCheckpoint, local.encoderWeights].map((a) => a.cache),
    ).toEqual([false, false, false, false]);
  });

  it("checks the cache for exactly the URLs the loaders fetch", () => {
    // The pre-flight cannot read the manifest (that would be a download before
    // the disclosure), so it names the q8 bins itself. If those names drifted
    // from what the loader asks for, the estimate would go quietly wrong.
    const speech = hf.preflight.speech.map((asset) => asset.url);
    expect(speech).toContain(hf.q8Bin("weights.codes.bin").url);
    expect(speech).toContain(hf.q8Bin("weights.scales.bin").url);
    expect(speech).toContain(hf.q8Bin("weights.norms.bin").url);
    expect(speech).toContain(hf.tokenizer.url);
    expect(speech).toContain(hf.codecCheckpoint.url);
    expect(hf.preflight.cloning.map((asset) => asset.url)).toEqual([hf.encoderWeights.url]);
  });

  it("states the byte totals the page discloses", () => {
    const sum = (assets: Asset[]) => assets.reduce((total, asset) => total + (asset.bytes ?? 0), 0);
    // text->speech: q8 (codes+scales+norms+manifest) + tokenizer + codec.
    expect(sum(hf.preflight.speech)).toBe(608_829_440 + 2_034_176 + 290_816 + 68_159 + 13_817_944 + 523_087_956);
    expect(sum(hf.preflight.cloning)).toBe(117_303_232);
  });
});

describe("estimateDownloads", () => {
  const plan = resolveAssets("hf");

  it("reports everything pending when nothing is cached", async () => {
    const estimate = await estimateDownloads(plan, fakeStorage(new FakeCache()));
    expect(estimate.cacheAvailable).toBe(true);
    expect(estimate.speechPending).toBe(estimate.speechBytes);
    expect(estimate.cloningPending).toBe(estimate.cloningBytes);
    expect(estimate.speechBytes).toBeCloseTo(1_148_128_491, 0);
  });

  it("subtracts what the cache already holds", async () => {
    const cache = new FakeCache();
    cache.entries.set(plan.codecCheckpoint.url, new Response(""));
    const estimate = await estimateDownloads(plan, fakeStorage(cache));
    expect(estimate.speechPending).toBe(estimate.speechBytes - 523_087_956);
    expect(estimate.cloningPending).toBe(estimate.cloningBytes);
  });

  it("reports nothing pending once every asset is cached", async () => {
    const cache = new FakeCache();
    for (const asset of [...plan.preflight.speech, ...plan.preflight.cloning]) {
      cache.entries.set(asset.url, new Response(""));
    }
    const estimate = await estimateDownloads(plan, fakeStorage(cache));
    expect(estimate.speechPending).toBe(0);
    expect(estimate.cloningPending).toBe(0);
  });

  it("treats an absent Cache API as 'everything will be downloaded'", async () => {
    const estimate = await estimateDownloads(plan, undefined);
    expect(estimate.cacheAvailable).toBe(false);
    expect(estimate.speechPending).toBe(estimate.speechBytes);
  });

  it("treats a broken Cache API the same way, without throwing", async () => {
    const broken = { open: () => Promise.reject(new Error("storage is toast")) } as unknown as CacheStorage;
    const estimate = await estimateDownloads(plan, broken);
    expect(estimate.cacheAvailable).toBe(false);
    expect(estimate.speechPending).toBe(estimate.speechBytes);
  });

  it("never claims a cache hit it could not verify", async () => {
    const cache = new FakeCache();
    cache.match = () => Promise.reject(new Error("match is toast"));
    const estimate = await estimateDownloads(plan, fakeStorage(cache));
    expect(estimate.speechPending).toBe(estimate.speechBytes);
  });

  it("reports local mode as uncached — serve.mjs is the cache there", async () => {
    const estimate = await estimateDownloads(resolveAssets("local"), fakeStorage(new FakeCache()));
    expect(estimate.cacheAvailable).toBe(false);
    expect(estimate.speechPending).toBe(estimate.speechBytes);
  });
});

describe("fetchAsset", () => {
  const asset: Asset = { url: "https://example.test/w.bin", bytes: 1024, cache: true };

  it("streams a miss, reports progress, and stores it", async () => {
    const cache = new FakeCache();
    const fetchFn = vi.fn(async () => chunked(1024));
    const seen: { loaded?: number; total?: number }[] = [];
    const buffer = await fetchAsset(asset, "weights", (_stage, detail) => seen.push({ ...detail }), {
      fetchFn: fetchFn as unknown as typeof fetch,
      cacheStorage: fakeStorage(cache),
    });

    expect(buffer.byteLength).toBe(1024);
    expect(seen.at(-1)).toEqual({ loaded: 1024, total: 1024 });
    expect(cache.puts).toEqual([asset.url]);
    // Stored WITH its length, so the next visit's progress bar still has a total.
    expect(cache.entries.get(asset.url)?.headers.get("content-length")).toBe("1024");
  });

  it("serves a hit from the cache without touching the network, and still reports progress", async () => {
    const cache = new FakeCache();
    cache.entries.set(asset.url, chunked(1024));
    const fetchFn = vi.fn(async () => {
      throw new Error("the network must not be used for a cache hit");
    });
    const seen: { loaded?: number; total?: number }[] = [];
    const buffer = await fetchAsset(asset, "weights", (_stage, detail) => seen.push({ ...detail }), {
      fetchFn: fetchFn as unknown as typeof fetch,
      cacheStorage: fakeStorage(cache),
    });

    expect(fetchFn).not.toHaveBeenCalled();
    expect(buffer.byteLength).toBe(1024);
    // Not silence: the caller's bar must reach the end on a hit too.
    expect(seen.at(-1)).toEqual({ loaded: 1024, total: 1024 });
  });

  it("never reads or writes the cache for an asset that opted out", async () => {
    const cache = new FakeCache();
    const match = vi.spyOn(cache, "match");
    const fetchFn = vi.fn(async () => chunked(64));
    await fetchAsset({ url: "./local.bin", cache: false }, "local", () => {}, {
      fetchFn: fetchFn as unknown as typeof fetch,
      cacheStorage: fakeStorage(cache),
    });
    expect(match).not.toHaveBeenCalled();
    expect(cache.puts).toEqual([]);
    expect(fetchFn).toHaveBeenCalledTimes(1);
  });

  it("falls through to a plain fetch when the cache cannot be opened", async () => {
    const warn = vi.fn();
    const fetchFn = vi.fn(async () => chunked(1024));
    const buffer = await fetchAsset(asset, "weights", () => {}, {
      fetchFn: fetchFn as unknown as typeof fetch,
      cacheStorage: { open: () => Promise.reject(new Error("storage is toast")) } as unknown as CacheStorage,
      warn,
    });
    expect(buffer.byteLength).toBe(1024);
    expect(fetchFn).toHaveBeenCalledTimes(1);
    expect(warn).toHaveBeenCalledWith(expect.stringContaining("storage is toast"));
  });

  it("falls through when the browser has no Cache API at all", async () => {
    const fetchFn = vi.fn(async () => chunked(1024));
    const buffer = await fetchAsset(asset, "weights", () => {}, {
      fetchFn: fetchFn as unknown as typeof fetch,
      cacheStorage: undefined,
    });
    expect(buffer.byteLength).toBe(1024);
    expect(fetchFn).toHaveBeenCalledTimes(1);
  });

  it("treats a cache that cannot answer match() as a miss", async () => {
    const cache = new FakeCache();
    cache.match = () => Promise.reject(new Error("match is toast"));
    const fetchFn = vi.fn(async () => chunked(1024));
    const buffer = await fetchAsset(asset, "weights", () => {}, {
      fetchFn: fetchFn as unknown as typeof fetch,
      cacheStorage: fakeStorage(cache),
    });
    expect(buffer.byteLength).toBe(1024);
    expect(fetchFn).toHaveBeenCalledTimes(1);
  });

  it("still returns the bytes when the cache refuses to store them", async () => {
    const cache = new FakeCache();
    cache.put = () => Promise.reject(new Error("quota exceeded"));
    const warn = vi.fn();
    const fetchFn = vi.fn(async () => chunked(1024));
    const buffer = await fetchAsset(asset, "weights", () => {}, {
      fetchFn: fetchFn as unknown as typeof fetch,
      cacheStorage: fakeStorage(cache),
      warn,
    });
    expect(buffer.byteLength).toBe(1024);
    expect(warn).toHaveBeenCalledWith(expect.stringContaining("quota exceeded"));
  });

  it("reports the URL and status when the fetch fails", async () => {
    const fetchFn = vi.fn(async () => new Response("nope", { status: 404 }));
    await expect(
      fetchAsset(asset, "weights", () => {}, {
        fetchFn: fetchFn as unknown as typeof fetch,
        cacheStorage: undefined,
      }),
    ).rejects.toThrow(/https:\/\/example\.test\/w\.bin: HTTP 404/);
  });

  it("does not cache a failed response", async () => {
    const cache = new FakeCache();
    const fetchFn = vi.fn(async () => new Response("nope", { status: 500 }));
    await expect(
      fetchAsset(asset, "weights", () => {}, {
        fetchFn: fetchFn as unknown as typeof fetch,
        cacheStorage: fakeStorage(cache),
      }),
    ).rejects.toThrow(/HTTP 500/);
    expect(cache.puts).toEqual([]);
  });

  it("copes with a response that has no readable body", async () => {
    const fetchFn = vi.fn(async () => {
      const response = new Response(new Uint8Array(8));
      Object.defineProperty(response, "body", { value: null });
      return response;
    });
    const seen: string[] = [];
    const buffer = await fetchAsset({ url: "./x.bin", cache: false }, "x", (stage) => seen.push(stage), {
      fetchFn: fetchFn as unknown as typeof fetch,
      cacheStorage: undefined,
    });
    expect(buffer.byteLength).toBe(8);
    expect(seen).toContain("x");
  });
});
