import type { VoiceEmbedding, VoiceStore } from "./types.js";
import { assertEmbedding, assertVoiceName, cloneEmbedding } from "./validate.js";

/**
 * Voice store that keeps everything in memory.
 *
 * This is the default in environments without IndexedDB and the store used by
 * tests; nothing survives a page reload.
 */
export class MemoryVoiceStore implements VoiceStore {
  readonly #voices = new Map<string, VoiceEmbedding>();

  async save(name: string, embedding: VoiceEmbedding): Promise<void> {
    this.#voices.set(assertVoiceName(name), cloneEmbedding(assertEmbedding(embedding)));
  }

  async load(name: string): Promise<VoiceEmbedding | undefined> {
    const stored = this.#voices.get(assertVoiceName(name));
    return stored ? cloneEmbedding(stored) : undefined;
  }

  async list(): Promise<string[]> {
    return [...this.#voices.keys()].sort();
  }

  async delete(name: string): Promise<boolean> {
    return this.#voices.delete(assertVoiceName(name));
  }

  async clear(): Promise<void> {
    this.#voices.clear();
  }
}
