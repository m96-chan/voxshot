/**
 * Where `examples/mio-tts.html` gets its ~1.15 GB of model assets, and how it
 * keeps them.
 *
 * The page is hosted on GitHub Pages (https://voxshot.m96-chan.dev/mio/), which
 * caps a file at 100 MB and a site at 1 GB — the q8 LM alone is 583 MB. So the
 * weights come from Hugging Face, and this module is the ONE place that says
 * which file lives where.
 *
 * ## The table
 *
 * | asset | repo | what it is |
 * | --- | --- | --- |
 * | `tokenizer.json` | `Aratako/MioTTS-0.6B` | upstream's tokenizer, verbatim |
 * | `manifest.json`, `weights.{codes,scales,norms}.bin` | `m96-chan/MioTTS-0.6B-q8-webgpu` | **derived**: `convert_weights.py` in this repo, int8 from upstream's fp checkpoint |
 * | `model.safetensors` | `Aratako/MioCodec-25Hz-24kHz` | upstream's codec checkpoint, verbatim |
 * | `encoder-weights.safetensors` | `m96-chan/MioCodec-encoder-webgpu` | **derived**: `../miocodec/export_encoder_weights.py`, MioCodec's encoder + WavLM's first two layers |
 *
 * The two derived repos are published FROM this repo's scripts; their manifests
 * carry the source checkpoints' sha256 and the script revision, so what the page
 * downloads can be traced back to what those scripts read (ISSUE #126).
 *
 * `?assets=local` swaps every URL back to the same-origin paths `serve.mjs`
 * maps, which is how `check-tts.mjs` avoids pulling a gigabyte from the CDN on
 * every run. The MioCodec checkpoint is the exception in both directions: it
 * keeps its canonical huggingface.co URL even in local mode, because the check
 * script (and `../miocodec/check-demo.mjs` before it) intercepts that URL with a
 * 302 to the local copy.
 *
 * ## Caching
 *
 * Everything large goes through the **Cache API**, the same mechanism
 * `examples/browser/src/model-cache.ts` uses for the Chatterbox demo — and with
 * the same posture: a CacheStorage that cannot be opened, read or written is
 * logged and stepped around, never thrown from. A broken cache must cost a
 * re-download, not the page.
 *
 * Local mode does not cache: `serve.mjs` is next to the browser, and filling a
 * gigabyte of disk on every `check:tts` run would be pure cost.
 */

export type AssetMode = "hf" | "local";

/** Progress sink — same shape as ../miocodec/browser.ts's `Progress`. */
export type Progress = (stage: string, detail?: { loaded?: number; total?: number }) => void;

export interface Asset {
  url: string;
  /** Known ahead of time, so the pre-flight disclosure needs no network. */
  bytes?: number;
  /** Whether `fetchAsset` should persist it in the Cache API. */
  cache: boolean;
}

/**
 * Cache name. Deliberately NOT `transformers-cache`: that one belongs to
 * Transformers.js on the same origin (the Chatterbox demo at the site root),
 * and sharing a bucket would let either page's eviction or clearing take out
 * the other's gigabyte.
 */
export const ASSET_CACHE_NAME = "miotts-assets-v1";

/** Sizes as published (bytes, from the HF repos' file listings). */
const BYTES = {
  q8Manifest: 68_159,
  q8Codes: 608_829_440,
  q8Scales: 2_034_176,
  q8Norms: 290_816,
  tokenizer: 13_817_944,
  codecCheckpoint: 523_087_956,
  encoderWeights: 117_303_232,
} as const;

/**
 * The q8 bin names.
 *
 * The loader asks for whatever `manifest.json` names (`q8Bin(manifest.files
 * .codes.name)`), because the manifest is the truth about the files it
 * describes. The pre-flight cannot read the manifest — that would be a download
 * before the page has disclosed anything — so it uses these. A manifest that
 * renamed a bin would only make the pre-flight pessimistic ("not cached" for a
 * file that is), never break a load.
 */
const Q8_BINS = {
  codes: { name: "weights.codes.bin", bytes: BYTES.q8Codes },
  scales: { name: "weights.scales.bin", bytes: BYTES.q8Scales },
  norms: { name: "weights.norms.bin", bytes: BYTES.q8Norms },
} as const;

const HF = {
  /** Derived from this repo's convert_weights.py — see the table above. */
  q8: "https://huggingface.co/m96-chan/MioTTS-0.6B-q8-webgpu/resolve/main/",
  /** Derived from this repo's ../miocodec/export_encoder_weights.py. */
  encoder: "https://huggingface.co/m96-chan/MioCodec-encoder-webgpu/resolve/main/",
  /** Upstream, verbatim. */
  miotts: "https://huggingface.co/Aratako/MioTTS-0.6B/resolve/main/",
  /** Upstream, verbatim. Also what mio-codec.html fetches. */
  miocodec: "https://huggingface.co/Aratako/MioCodec-25Hz-24kHz/resolve/main/",
} as const;

/** The one URL both pages and both check scripts agree on — see the module doc. */
export const CODEC_CHECKPOINT_URL = `${HF.miocodec}model.safetensors`;

export interface AssetPlan {
  mode: AssetMode;
  tokenizer: Asset;
  q8Manifest: Asset;
  /** A q8 bin by the name the manifest gives it. */
  q8Bin: (name: string) => Asset;
  codecCheckpoint: Asset;
  encoderWeights: Asset;
  /** The demo speaker embedding — ships with the page, so always same-origin. */
  fixture: Asset;
  /** What the pre-flight disclosure counts, split the way the user chooses. */
  preflight: { speech: Asset[]; cloning: Asset[] };
}

export function resolveAssets(mode: AssetMode): AssetPlan {
  const hf = mode === "hf";
  // Local mode is served by serve.mjs from the machine running the browser;
  // caching a gigabyte of it would cost disk and time and save nothing.
  const cache = hf;
  const asset = (url: string, bytes?: number): Asset => ({ url, bytes, cache });

  const q8Base = hf ? HF.q8 : "./miotts/q8/";
  const tokenizer = asset(hf ? `${HF.miotts}tokenizer.json` : "./miotts/tokenizer.json", BYTES.tokenizer);
  const q8Manifest = asset(`${q8Base}manifest.json`, BYTES.q8Manifest);
  const q8Bin = (name: string): Asset =>
    asset(`${q8Base}${name}`, Object.values(Q8_BINS).find((bin) => bin.name === name)?.bytes);
  const codecCheckpoint = asset(CODEC_CHECKPOINT_URL, BYTES.codecCheckpoint);
  const encoderWeights = asset(
    hf ? `${HF.encoder}encoder-weights.safetensors` : "./miotts/encoder-weights.safetensors",
    BYTES.encoderWeights,
  );

  return {
    mode,
    tokenizer,
    q8Manifest,
    q8Bin,
    codecCheckpoint,
    encoderWeights,
    fixture: { url: "./mio-codec-fixture.json", cache: false },
    preflight: {
      speech: [
        tokenizer,
        q8Manifest,
        q8Bin(Q8_BINS.codes.name),
        q8Bin(Q8_BINS.scales.name),
        q8Bin(Q8_BINS.norms.name),
        codecCheckpoint,
      ],
      cloning: [encoderWeights],
    },
  };
}

/**
 * `?assets=local` (serve.mjs) or the default `hf`.
 *
 * An unrecognised value throws rather than falling back: a typo'd
 * `?assets=locl` that silently started a 1.15 GB download from the CDN is
 * exactly the surprise this ISSUE exists to remove.
 */
export function assetModeFromSearch(search: string): AssetMode {
  const value = new URLSearchParams(search).get("assets");
  if (value === null || value === "hf") return "hf";
  if (value === "local") return "local";
  throw new Error(`unknown ?assets=${value} — use "hf" (default, Hugging Face) or "local" (serve.mjs)`);
}

export interface DownloadEstimate {
  mode: AssetMode;
  /** Bytes for text -> speech: tokenizer + q8 + the codec checkpoint. */
  speechBytes: number;
  /** Of those, what is not already cached. Equals speechBytes when there is no usable cache. */
  speechPending: number;
  /** What voice cloning adds on top: the encoder weights. */
  cloningBytes: number;
  cloningPending: number;
  /** False when the Cache API is missing, broken, or not in use (local mode) — then "pending" is a worst case. */
  cacheAvailable: boolean;
}

/**
 * What the page will download if the visitor clicks now.
 *
 * Reads the cache only; never fetches. Anything it cannot verify counts as
 * pending, so the number shown is never smaller than the truth.
 */
export async function estimateDownloads(
  plan: AssetPlan,
  cacheStorage: CacheStorage | undefined,
  warn: (message: string) => void = () => {},
): Promise<DownloadEstimate> {
  const sum = (assets: Asset[]) => assets.reduce((total, a) => total + (a.bytes ?? 0), 0);
  const speechBytes = sum(plan.preflight.speech);
  const cloningBytes = sum(plan.preflight.cloning);
  const worstCase: DownloadEstimate = {
    mode: plan.mode,
    speechBytes,
    speechPending: speechBytes,
    cloningBytes,
    cloningPending: cloningBytes,
    cacheAvailable: false,
  };

  const cachedAssets = [...plan.preflight.speech, ...plan.preflight.cloning].filter((a) => a.cache);
  if (cachedAssets.length === 0) return worstCase;
  const cache = await openAssetCache(cacheStorage, warn);
  if (!cache) return worstCase;

  const pending = async (assets: Asset[]) => {
    let total = 0;
    for (const a of assets) {
      if (a.cache && (await matchSafely(cache, a.url))) continue;
      total += a.bytes ?? 0;
    }
    return total;
  };
  return {
    ...worstCase,
    speechPending: await pending(plan.preflight.speech),
    cloningPending: await pending(plan.preflight.cloning),
    cacheAvailable: true,
  };
}

export interface FetchDeps {
  fetchFn?: typeof fetch;
  /** Pass `globalThis.caches`; undefined when the Cache API is unavailable. */
  cacheStorage?: CacheStorage;
  warn?: (message: string) => void;
}

/**
 * Fetch one asset, streaming, with progress — from the Cache API when it is
 * there, from the network otherwise, and storing what it downloaded.
 *
 * Progress is reported on BOTH paths: a cache hit runs the same drain loop and
 * so drives the bar to 100% immediately, rather than leaving the page silent
 * for the seconds it takes to read 600 MB back off disk.
 */
export async function fetchAsset(
  asset: Asset,
  label: string,
  report: Progress,
  deps: FetchDeps = {},
): Promise<ArrayBuffer> {
  const fetchFn = deps.fetchFn ?? fetch;
  const warn = deps.warn ?? ((message: string) => console.warn(message));
  const cache = asset.cache ? await openAssetCache(deps.cacheStorage, warn) : undefined;

  if (cache) {
    const hit = await matchSafely(cache, asset.url);
    if (hit) return await drain(hit, label, report);
  }

  report(label);
  // `no-store` for anything we persist ourselves: the HTTP disk cache would
  // otherwise hold a second copy of the same gigabyte, and an aborted download
  // can leave an entry there that wedges later fetches of the same URL
  // (model-cache.ts hit exactly that).
  const response = await fetchFn(asset.url, cache ? { cache: "no-store" } : undefined);
  if (!response.ok) throw new Error(`${asset.url}: HTTP ${response.status}`);
  const contentType = response.headers.get("content-type") ?? "application/octet-stream";
  const buffer = await drain(response, label, report);

  if (cache) {
    try {
      // Re-materialised from the bytes we already hold: the original body is
      // consumed by the drain above, and a `clone()` before it would make the
      // browser buffer the whole download a second time.
      await cache.put(
        asset.url,
        new Response(buffer, {
          headers: { "content-type": contentType, "content-length": String(buffer.byteLength) },
        }),
      );
    } catch (cause) {
      warn(
        `miotts: storing ${asset.url} in the browser cache failed (${describe(cause)}). ` +
          "Continuing — this asset will be downloaded again on the next visit.",
      );
    }
  }
  return buffer;
}

/** Read a response to completion, reporting progress as chunks land. */
async function drain(response: Response, label: string, report: Progress): Promise<ArrayBuffer> {
  report(label);
  const header = response.headers.get("content-length");
  const total = header ? Number(header) : undefined;
  const reader = response.body?.getReader();
  if (!reader) return await response.arrayBuffer();

  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    report(label, { loaded, total });
  }
  const buffer = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return buffer.buffer;
}

async function openAssetCache(
  cacheStorage: CacheStorage | undefined,
  warn: (message: string) => void,
): Promise<Cache | undefined> {
  if (!cacheStorage) return undefined;
  try {
    return await cacheStorage.open(ASSET_CACHE_NAME);
  } catch (cause) {
    warn(
      `miotts: opening the browser cache failed (${describe(cause)}). ` +
        "Continuing without it — the model will stream directly and be re-downloaded on the next visit.",
    );
    return undefined;
  }
}

/** A cache too broken to answer match() is treated as a plain miss. */
async function matchSafely(cache: Cache, url: string): Promise<Response | undefined> {
  try {
    return await cache.match(url);
  } catch {
    return undefined;
  }
}

function describe(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}
