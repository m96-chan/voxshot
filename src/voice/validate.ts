import { InvalidInputError } from "../errors.js";
import type { VoiceEmbedding, VoiceTensor } from "./types.js";

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

/** Defensive copy so stores never alias the caller's buffers. */
export function cloneEmbedding(embedding: VoiceEmbedding): VoiceEmbedding {
  const clone: Mutable<VoiceEmbedding> = {
    vector: Float32Array.from(embedding.vector),
    sampleRate: embedding.sampleRate,
    createdAt: embedding.createdAt,
  };
  if (embedding.engine !== undefined) {
    clone.engine = embedding.engine;
  }
  if (embedding.tensors) {
    clone.tensors = cloneTensors(embedding.tensors);
  }
  return clone;
}

/** Deep copy a tensor map, including its typed array payloads. */
export function cloneTensors(
  tensors: Readonly<Record<string, VoiceTensor>>,
): Record<string, VoiceTensor> {
  const clone: Record<string, VoiceTensor> = {};
  for (const [name, tensor] of Object.entries(tensors)) {
    clone[name] = {
      type: tensor.type,
      dims: [...tensor.dims],
      data:
        tensor.data instanceof BigInt64Array
          ? BigInt64Array.from(tensor.data)
          : Float32Array.from(tensor.data),
    };
  }
  return clone;
}

type Mutable<T> = { -readonly [K in keyof T]: T[K] };
