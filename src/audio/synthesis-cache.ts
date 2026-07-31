import { InvalidInputError } from "../errors.js";
import type { VoiceEmbedding } from "../voice/types.js";

const DEFAULT_MAX_ENTRIES_PER_VOICE = 32;

export interface SynthesisCacheOptions {
  /**
   * How many rendered chunks to keep per voice before evicting the least
   * recently used one.
   *
   * @defaultValue 32
   */
  maxEntriesPerVoice?: number;
}

/**
 * LRU cache of rendered audio, keyed by voice identity, text, speed and
 * expressiveness.
 *
 * Voices are compared by object identity (the `VoiceEmbedding` instance held
 * by `VoxShot`), so re-cloning the same reference audio starts a fresh cache.
 * Entries are held through a `WeakMap`, so dropping a voice frees its audio.
 */
export class SynthesisCache {
  readonly #maxEntriesPerVoice: number;
  readonly #store = new WeakMap<VoiceEmbedding, Map<string, Float32Array>>();

  constructor(options: SynthesisCacheOptions = {}) {
    const maxEntries = options.maxEntriesPerVoice ?? DEFAULT_MAX_ENTRIES_PER_VOICE;
    if (!Number.isFinite(maxEntries) || maxEntries <= 0) {
      throw new InvalidInputError("maxEntriesPerVoice must be a positive finite number.");
    }
    this.#maxEntriesPerVoice = maxEntries;
  }

  get(
    voice: VoiceEmbedding,
    text: string,
    speed: number,
    expressiveness?: number,
  ): Float32Array | undefined {
    const entries = this.#store.get(voice);
    const key = cacheKey(text, speed, expressiveness);
    const samples = entries?.get(key);
    if (entries && samples) {
      // Refresh recency: Map iteration order doubles as the LRU order.
      entries.delete(key);
      entries.set(key, samples);
    }
    return samples;
  }

  set(
    voice: VoiceEmbedding,
    text: string,
    speed: number,
    samples: Float32Array,
    expressiveness?: number,
  ): void {
    let entries = this.#store.get(voice);
    if (!entries) {
      entries = new Map();
      this.#store.set(voice, entries);
    }

    const key = cacheKey(text, speed, expressiveness);
    entries.delete(key);
    entries.set(key, samples);

    while (entries.size > this.#maxEntriesPerVoice) {
      const oldest = entries.keys().next().value as string;
      entries.delete(oldest);
    }
  }
}

/**
 * Anything that changes the rendered audio has to be in the key.
 *
 * Fields are joined with NUL because it cannot occur in any of them, so no
 * combination of values can be mistaken for another. Speed and
 * expressiveness happen to be numbers today, which would make a printable
 * separator safe, but the guarantee should not depend on that — and a file
 * containing NUL is treated as binary by git, so a separator quietly
 * downgraded to a space would not show up in a diff.
 *
 * An unspecified `expressiveness` keys separately from any explicit value,
 * even one equal to the engine's default: the cache cannot know what that
 * default is, so treating the two as equal would risk handing back audio
 * rendered at a different setting.
 */
function cacheKey(text: string, speed: number, expressiveness?: number): string {
  return `${speed}\0${expressiveness ?? "default"}\0${text}`;
}
