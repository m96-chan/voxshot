import { InvalidInputError } from "../errors.js";
import type { VoiceEmbedding } from "./types.js";

/** Normalise a voice name and reject blank ones. */
export function assertVoiceName(name: string): string {
  const trimmed = name.trim();
  if (trimmed.length === 0) {
    throw new InvalidInputError("A voice name must not be empty.");
  }
  return trimmed;
}

/** Reject embeddings that carry no data. */
export function assertEmbedding(embedding: VoiceEmbedding): VoiceEmbedding {
  if (embedding.vector.length === 0) {
    throw new InvalidInputError("A voice embedding must not be empty.");
  }
  return embedding;
}

/** Defensive copy so stores never alias the caller's buffer. */
export function cloneEmbedding(embedding: VoiceEmbedding): VoiceEmbedding {
  return {
    vector: Float32Array.from(embedding.vector),
    sampleRate: embedding.sampleRate,
    createdAt: embedding.createdAt,
  };
}
