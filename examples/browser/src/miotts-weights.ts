import type { MioTtsWeightPart, MioTtsWeightSource } from "voxshot/miotts";

/**
 * Where MioTTS' weights come from, and how this page keeps them.
 *
 * The library deliberately answers neither question. `voxshot/miotts` asks for
 * bytes by part name and stops there, because the right answer depends on the
 * application — a private CDN, a Service Worker, the Cache API, IndexedDB, the
 * File System Access API, a bundled asset, `readFileSync` in Node. This file is
 * one such answer, and it is a *reference*: copy it and change the parts that
 * do not suit you.
 *
 * What it does:
 *
 * - maps each part onto a Hugging Face URL, and nothing else knows those URLs;
 * - keeps everything in the **Cache API**, so a second visit downloads nothing;
 * - reports progress, because 1.1 GB with no feedback reads as a hung page;
 * - can say what a run will cost *before* it starts, so the visitor is told
 *   before a gigabyte of traffic rather than after.
 *
 * The one posture worth copying whatever else you change: a `CacheStorage` that
 * cannot be opened, read or written is logged and stepped around, never thrown
 * from. A broken cache should cost a re-download, not the page.
 */

/** Published sizes, from the model repos' own file listings. */
export const MIOTTS_ASSET_BYTES: Readonly<Record<MioTtsWeightPart, number>> = {
  tokenizer: 13_817_944,
  "lm-manifest": 68_159,
  "lm-codes": 608_829_440,
  "lm-scales": 2_034_176,
  "lm-norms": 290_816,
  "codec-decoder": 523_087_956,
  "codec-encoder": 117_303_232,
};

/** What text-to-speech needs. Everything except the voice encoder. */
export const SPEECH_PARTS: readonly MioTtsWeightPart[] = [
  "tokenizer",
  "lm-manifest",
  "lm-codes",
  "lm-scales",
  "lm-norms",
  "codec-decoder",
];

/** What voice cloning adds on top. */
export const CLONING_PARTS: readonly MioTtsWeightPart[] = ["codec-encoder"];

/**
 * Cache name.
 *
 * Deliberately not `transformers-cache`: that one belongs to Transformers.js on
 * the same origin, and sharing a bucket would let either page's eviction take
 * out the other's gigabyte.
 */
export const MIOTTS_CACHE_NAME = "miotts-assets-v1";

const HF = "https://huggingface.co";

/**
 * Part to URL.
 *
 * Two of these repos are *derived* — the int8 language model and the voice
 * encoder are produced by scripts in this repository from upstream's
 * checkpoints, and their manifests carry the source sha256 and the script
 * revision so a download can be traced back to what those scripts read.
 */
const URLS: Readonly<Record<MioTtsWeightPart, string>> = {
  tokenizer: `${HF}/Aratako/MioTTS-0.6B/resolve/main/tokenizer.json`,
  "lm-manifest": `${HF}/m96-chan/MioTTS-0.6B-q8-webgpu/resolve/main/manifest.json`,
  "lm-codes": `${HF}/m96-chan/MioTTS-0.6B-q8-webgpu/resolve/main/weights.codes.bin`,
  "lm-scales": `${HF}/m96-chan/MioTTS-0.6B-q8-webgpu/resolve/main/weights.scales.bin`,
  "lm-norms": `${HF}/m96-chan/MioTTS-0.6B-q8-webgpu/resolve/main/weights.norms.bin`,
  "codec-decoder": `${HF}/Aratako/MioCodec-25Hz-24kHz/resolve/main/model.safetensors`,
  "codec-encoder": `${HF}/m96-chan/MioCodec-encoder-webgpu/resolve/main/encoder-weights.safetensors`,
};

/** The URL a part is fetched from. */
export function assetUrl(part: MioTtsWeightPart): string {
  return URLS[part];
}

export type WeightProgress = (
  part: MioTtsWeightPart,
  detail: { loaded: number; total?: number },
) => void;

export interface WeightSourceOptions {
  /** Pass `globalThis.caches`; `undefined` where the Cache API is unavailable. */
  cacheStorage?: CacheStorage | undefined;
  fetchFn?: typeof fetch;
  onProgress?: WeightProgress;
  /** Where a cache problem is reported. Defaults to `console.warn`. */
  onCacheProblem?: (message: string) => void;
}

/** Open the cache, or report why not and carry on without one. */
async function openCache(
  storage: CacheStorage | undefined,
  warn: (message: string) => void,
): Promise<Cache | null> {
  if (!storage) return null;
  try {
    return await storage.open(MIOTTS_CACHE_NAME);
  } catch (error) {
    warn(`The model cache could not be opened (${String(error)}); weights will be re-downloaded.`);
    return null;
  }
}

/** Read a response body, reporting progress as it arrives. */
async function readWithProgress(
  response: Response,
  part: MioTtsWeightPart,
  onProgress: WeightProgress | undefined,
): Promise<ArrayBuffer> {
  const declared = Number(response.headers.get("content-length"));
  const total = Number.isFinite(declared) && declared > 0 ? declared : MIOTTS_ASSET_BYTES[part];

  if (!onProgress || !response.body) {
    const buffer = await response.arrayBuffer();
    onProgress?.(part, { loaded: buffer.byteLength, total });
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress(part, { loaded, total });
  }
  const out = new Uint8Array(loaded);
  let at = 0;
  for (const chunk of chunks) {
    out.set(chunk, at);
    at += chunk.byteLength;
  }
  return out.buffer;
}

/**
 * A {@link MioTtsWeightSource} backed by Hugging Face and the Cache API.
 *
 * Each part is fetched at most once per engine, so there is no memo here — the
 * engine does not ask twice, and the cache covers the next visit.
 */
export function createHuggingFaceWeightSource(
  options: WeightSourceOptions = {},
): MioTtsWeightSource {
  const fetchFn = options.fetchFn ?? globalThis.fetch.bind(globalThis);
  const warn = options.onCacheProblem ?? ((message: string) => console.warn(message));

  return {
    async load(part: MioTtsWeightPart): Promise<ArrayBuffer> {
      const url = assetUrl(part);
      const cache = await openCache(options.cacheStorage, warn);

      if (cache) {
        try {
          const hit = await cache.match(url);
          if (hit) {
            const buffer = await hit.arrayBuffer();
            options.onProgress?.(part, { loaded: buffer.byteLength, total: buffer.byteLength });
            return buffer;
          }
        } catch (error) {
          warn(`Reading ${part} from the cache failed (${String(error)}); fetching it instead.`);
        }
      }

      const response = await fetchFn(url);
      if (!response.ok) {
        throw new Error(`Fetching ${part} from ${url} failed: ${response.status} ${response.statusText}`);
      }

      // Cloned before the body is read: a Response body can only be consumed
      // once, and putting the clone means the cache holds the bytes even
      // though this call also needs them.
      const forCache = cache ? response.clone() : null;
      const buffer = await readWithProgress(response, part, options.onProgress);
      if (cache && forCache) {
        try {
          await cache.put(url, forCache);
        } catch (error) {
          warn(`Storing ${part} in the cache failed (${String(error)}); it will be fetched again.`);
        }
      }
      return buffer;
    },
  };
}

export interface DownloadEstimate {
  /** Total size of the parts asked about. */
  bytes: number;
  /** How much of that is not already cached. */
  pending: number;
  /** False when this browser has no usable Cache API, so nothing persists. */
  cacheAvailable: boolean;
}

/**
 * What downloading `parts` would cost, without downloading anything.
 *
 * Worth doing before the first click: 1.1 GB is a number a visitor should be
 * given the chance to decline, and "already cached" is the difference between
 * a long wait and none.
 */
export async function estimateDownload(
  parts: readonly MioTtsWeightPart[],
  cacheStorage: CacheStorage | undefined,
  onCacheProblem: (message: string) => void = (message) => console.warn(message),
): Promise<DownloadEstimate> {
  const bytes = parts.reduce((sum, part) => sum + MIOTTS_ASSET_BYTES[part], 0);
  const cache = await openCache(cacheStorage, onCacheProblem);
  if (!cache) return { bytes, pending: bytes, cacheAvailable: false };

  let pending = 0;
  for (const part of parts) {
    try {
      if (!(await cache.match(assetUrl(part)))) pending += MIOTTS_ASSET_BYTES[part];
    } catch (error) {
      onCacheProblem(`Checking the cache for ${part} failed (${String(error)}).`);
      pending += MIOTTS_ASSET_BYTES[part];
    }
  }
  return { bytes, pending, cacheAvailable: true };
}
