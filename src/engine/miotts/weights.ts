import { InvalidInputError } from "../../errors.js";

/**
 * How the MioTTS engine gets hold of its weights.
 *
 * VoxShot's responsibility stops at inference. Where ~1.1 GB of weights come
 * from, how they are cached, and how progress is reported are the caller's
 * decisions, because the right answer depends on the application: a private
 * CDN, a Service Worker, the Cache API, IndexedDB, the File System Access API,
 * a bundled asset, or `readFileSync` in Node. An engine that picked one would
 * be an obstacle to everyone it picked wrong for.
 *
 * So the engine asks for bytes by part name and nothing else — there is no
 * `fetch`, no `caches`, no `indexedDB` and no `fs` anywhere below this module.
 */

/**
 * The parts the engine can ask for, in the order they are first needed.
 *
 * Names are stable API: callers switch over them to map a part onto a URL. They
 * are deliberately not file names — the q8 bins are named inside the manifest,
 * and a caller who has renamed or repacked them should not have to lie about it.
 *
 * `codec-encoder` is the odd one out: it is only ever requested by
 * {@link MioTtsEngine.embed}, so a caller who synthesises with an embedding
 * they already hold never pays for its 117 MB. That is why parts are fetched
 * one at a time instead of handed over as one bundle.
 */
export const MIOTTS_WEIGHT_PARTS = [
  "tokenizer",
  "lm-manifest",
  "lm-codes",
  "lm-scales",
  "lm-norms",
  "codec-decoder",
  "codec-encoder",
] as const;

/** One of {@link MIOTTS_WEIGHT_PARTS}. */
export type MioTtsWeightPart = (typeof MIOTTS_WEIGHT_PARTS)[number];

/** The caller's answer to "give me the bytes of this part". */
export interface MioTtsWeightSource {
  /**
   * Resolve `part` to its bytes.
   *
   * Called at most once per part per engine, so an implementation does not
   * need its own memo — though it is free to have one, and a caller that
   * shares a source between engines will want it.
   *
   * A view (`Uint8Array`, Node's `Buffer`) is accepted and copied out at
   * exactly its window.
   */
  load(part: MioTtsWeightPart): Promise<ArrayBuffer | ArrayBufferView>;
}

/** `ArrayBuffer.isView` narrowed for the union the source may return. */
function isView(value: unknown): value is ArrayBufferView {
  return ArrayBuffer.isView(value);
}

/**
 * Fetch one part and normalise it to a standalone `ArrayBuffer`.
 *
 * The copy for a view is not defensive politeness. `readFileSync` returns a
 * `Buffer` backed by a pooled `ArrayBuffer` shared with unrelated reads, so
 * `.buffer` is usually larger than the file and starts somewhere else — using
 * it directly would feed a neighbouring allocation to the safetensors parser.
 */
export async function loadPart(
  source: MioTtsWeightSource,
  part: MioTtsWeightPart,
): Promise<ArrayBuffer> {
  let bytes: ArrayBuffer | ArrayBufferView;
  try {
    bytes = await source.load(part);
  } catch (cause) {
    throw new Error(`The weight source failed to load "${part}".`, { cause });
  }

  if (bytes instanceof ArrayBuffer) return bytes;
  if (isView(bytes)) {
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
  }
  // Most often a `Response` that was returned instead of awaited through
  // `.arrayBuffer()`. Saying so here beats a parse failure on garbage bytes.
  throw new InvalidInputError(
    `The weight source returned ${describe(bytes)} for "${part}"; expected an ArrayBuffer or a view over one.`,
  );
}

/** Fetch one part and decode it as UTF-8 JSON. */
export async function readJsonPart<T>(
  source: MioTtsWeightSource,
  part: MioTtsWeightPart,
): Promise<T> {
  const buffer = await loadPart(source, part);
  const text = new TextDecoder().decode(buffer);
  try {
    return JSON.parse(text) as T;
  } catch (cause) {
    throw new InvalidInputError(`The bytes of "${part}" are not valid JSON.`, { cause });
  }
}

/**
 * A short type name for an error message, without stringifying the value —
 * the wrong value here is potentially hundreds of megabytes.
 *
 * `Object.prototype.toString` rather than `constructor.name` so that a
 * null-prototype object still names itself instead of throwing on the way to
 * reporting someone else's mistake.
 */
function describe(value: unknown): string {
  if (value === null) return "null";
  if (typeof value === "object") {
    return `a ${Object.prototype.toString.call(value).slice("[object ".length, -1)}`;
  }
  return `a ${typeof value}`;
}
