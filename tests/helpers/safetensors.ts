/**
 * Build a safetensors file in memory.
 *
 * The engine's checkpoints are hundreds of megabytes and not in git, so every
 * CI-side test that needs one builds a tiny synthetic checkpoint instead. This
 * is the writer for that — the inverse of `codec/safetensors.ts`'s reader, kept
 * in the tests because nothing in the library ever writes one.
 */

export interface SyntheticTensor {
  readonly shape: readonly number[];
  readonly data: Float32Array;
}

/** Serialise `tensors` into the bytes `Safetensors.parse` expects. */
export function buildSafetensors(
  tensors: Readonly<Record<string, SyntheticTensor>>,
  options: { dtype?: string; metadata?: Record<string, string> } = {},
): ArrayBuffer {
  const dtype = options.dtype ?? "F32";
  const header: Record<string, unknown> = {};
  if (options.metadata) header.__metadata__ = options.metadata;

  let offset = 0;
  const payloads: Float32Array[] = [];
  for (const [name, tensor] of Object.entries(tensors)) {
    const bytes = tensor.data.byteLength;
    header[name] = { dtype, shape: [...tensor.shape], data_offsets: [offset, offset + bytes] };
    payloads.push(tensor.data);
    offset += bytes;
  }

  const json = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(8 + json.byteLength + offset);
  new DataView(out.buffer).setBigUint64(0, BigInt(json.byteLength), true);
  out.set(json, 8);
  let at = 8 + json.byteLength;
  for (const payload of payloads) {
    out.set(new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength), at);
    at += payload.byteLength;
  }
  return out.buffer;
}

/**
 * Deterministic pseudo-random values in `[-scale, scale)`.
 *
 * A fixed generator rather than `Math.random`, so a failure is reproducible
 * and a shape mistake cannot hide behind a lucky draw.
 */
export function pseudoRandom(count: number, seed = 1, scale = 1): Float32Array {
  const out = new Float32Array(count);
  let state = seed >>> 0 || 1;
  for (let i = 0; i < count; i += 1) {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    out[i] = ((state / 0xffffffff) * 2 - 1) * scale;
  }
  return out;
}
