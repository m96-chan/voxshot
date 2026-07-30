export const MODEL_ID = "onnx-community/chatterbox-ONNX";
export const MODEL_BASE = `https://huggingface.co/${MODEL_ID}/resolve/main/`;

/** Cache name Transformers.js reads from (`env.cacheKey` default). */
export const MODEL_CACHE_NAME = "transformers-cache";

export type LanguageModelDtype = "q4f16" | "q4";

export function modelFiles(dtype: LanguageModelDtype): string[] {
  return [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "onnx/embed_tokens.onnx",
    "onnx/embed_tokens.onnx_data",
    "onnx/speech_encoder.onnx",
    "onnx/speech_encoder.onnx_data",
    "onnx/conditional_decoder.onnx",
    "onnx/conditional_decoder.onnx_data",
    `onnx/language_model_${dtype}.onnx`,
    `onnx/language_model_${dtype}.onnx_data`,
  ];
}

export interface PrewarmModelCacheOptions {
  dtype: LanguageModelDtype;
  log: (message: string) => void;
  /** Pass `globalThis.caches`; undefined when the Cache API is unavailable. */
  cacheStorage: CacheStorage | undefined;
  fetchFn?: typeof fetch;
}

/**
 * Fetch every model file into transformers-cache while the page is idle.
 *
 * `from_pretrained` initialises each ONNX session as soon as that session's
 * file arrives, and session init blocks the thread that is also consuming the
 * remaining download streams — the unread stream backs up until the server
 * resets it. Pre-warming the cache means the engine later loads everything
 * from cache and no live download is left to kill.
 *
 * Returns true when every file is in the cache afterwards. CacheStorage can
 * be broken at runtime (corrupted browser storage, storage service crash,
 * private mode restrictions); in that case this logs a warning and returns
 * false instead of throwing — Transformers.js falls back to plain downloads
 * on its own, so a broken cache must not block model loading.
 */
export async function prewarmModelCache(options: PrewarmModelCacheOptions): Promise<boolean> {
  const { dtype, log } = options;
  const fetchFn = options.fetchFn ?? fetch;

  const cache = await openModelCache(options.cacheStorage, log);
  if (!cache) {
    return false;
  }

  for (const file of modelFiles(dtype)) {
    const url = MODEL_BASE + file;
    if (await matchesSafely(cache, url)) {
      continue;
    }
    log(`Fetching ${file}…`);
    // no-store bypasses the HTTP disk cache entirely: aborted downloads can
    // leave corrupt cache entries behind that wedge later fetches of the same
    // URL, and we persist into transformers-cache ourselves anyway.
    const response = await fetchFn(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Downloading ${file} failed: HTTP ${response.status}`);
    }
    try {
      await cache.put(url, await withProgress(response, file, log));
    } catch (cause) {
      log(
        `⚠ Writing to the browser cache failed (${describeError(cause)}). ` +
          "Continuing without the cache — the model will stream directly and " +
          "be re-downloaded on the next visit.",
      );
      return false;
    }
  }
  log("All model files are in the browser cache.");
  return true;
}

async function openModelCache(
  cacheStorage: CacheStorage | undefined,
  log: (message: string) => void,
): Promise<Cache | undefined> {
  if (!cacheStorage) {
    log(
      "⚠ The Cache API is unavailable in this browser. Skipping the model " +
        "pre-download — the model will stream directly and be re-downloaded " +
        "on the next visit.",
    );
    return undefined;
  }
  try {
    return await cacheStorage.open(MODEL_CACHE_NAME);
  } catch (cause) {
    log(
      `⚠ Opening the browser cache failed (${describeError(cause)}). ` +
        "Skipping the model pre-download — the model will stream directly " +
        "and be re-downloaded on the next visit. Restarting the browser or " +
        "clearing this site's data usually repairs the cache.",
    );
    return undefined;
  }
}

/** A cache too broken to answer match() is treated as a plain miss. */
async function matchesSafely(cache: Cache, url: string): Promise<boolean> {
  try {
    return (await cache.match(url)) !== undefined;
  } catch {
    return false;
  }
}

function describeError(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

/** Re-materialise a response while logging download progress in 25% steps. */
async function withProgress(
  response: Response,
  file: string,
  log: (message: string) => void,
): Promise<Response> {
  const total = Number(response.headers.get("Content-Length") ?? 0);
  if (!response.body || !Number.isFinite(total) || total <= 0) {
    return response;
  }

  const reader = response.body.getReader();
  const parts: BlobPart[] = [];
  let received = 0;
  let lastStep = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    parts.push(value as BlobPart);
    received += value.length;
    const step = Math.floor((received / total) * 4);
    if (step > lastStep) {
      lastStep = step;
      log(`Fetching ${file}: ${Math.min(100, step * 25)}%`);
    }
  }

  return new Response(new Blob(parts), { status: 200, headers: response.headers });
}
