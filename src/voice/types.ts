/** Element type of a {@link VoiceTensor}. */
export type VoiceTensorType = "float32" | "int64";

/**
 * A named tensor belonging to a voice.
 *
 * Real models rarely describe a speaker with a single vector — Chatterbox, for
 * example, carries four tensors (one of them int64) between its speech encoder
 * and its decoder. Both element types survive IndexedDB's structured clone
 * untouched, so no serialization step is needed.
 */
export interface VoiceTensor {
  readonly type: VoiceTensorType;
  readonly dims: readonly number[];
  readonly data: Float32Array | BigInt64Array;
}

/**
 * A speaker representation extracted from reference audio.
 *
 * The contents are opaque: their meaning and dimensionality belong to the
 * engine that produced them, so embeddings are not portable between engines.
 * {@link VoiceEmbedding.engine} records the producer so a mismatch can be
 * reported instead of silently generating noise.
 */
export interface VoiceEmbedding {
  /** Primary speaker vector, for similarity checks and simple engines. */
  readonly vector: Float32Array;
  /** Sample rate of the audio the embedding was extracted from. */
  readonly sampleRate: number;
  /** Unix epoch milliseconds of extraction. */
  readonly createdAt: number;
  /** Name of the engine that produced this embedding, when known. */
  readonly engine?: string;
  /** Engine specific tensors that `vector` alone cannot express. */
  readonly tensors?: Readonly<Record<string, VoiceTensor>>;
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
