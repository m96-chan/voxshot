/**
 * A speaker representation extracted from reference audio.
 *
 * The vector is opaque: its meaning and dimensionality belong to the engine
 * that produced it, so embeddings are not portable between engines.
 */
export interface VoiceEmbedding {
  readonly vector: Float32Array;
  /** Sample rate of the audio the embedding was extracted from. */
  readonly sampleRate: number;
  /** Unix epoch milliseconds of extraction. */
  readonly createdAt: number;
}

/** Persistence for named voices. */
export interface VoiceStore {
  /** Store `embedding` under `name`, replacing any previous entry. */
  save(name: string, embedding: VoiceEmbedding): Promise<void>;
  /** Read a voice back, or `undefined` when it does not exist. */
  load(name: string): Promise<VoiceEmbedding | undefined>;
  /** Every stored voice name, alphabetically sorted. */
  list(): Promise<string[]>;
  /** Remove a voice; resolves to `false` when there was nothing to remove. */
  delete(name: string): Promise<boolean>;
  /** Remove every stored voice. */
  clear(): Promise<void>;
}
