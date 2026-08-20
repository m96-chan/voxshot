/**
 * The slice of safetensors this port needs, and no more.
 *
 * The format is a little-endian u64 header length, that many bytes of JSON
 * naming every tensor's dtype, shape and byte range, then the data. There is no
 * compression and no framing beyond that, which is why reading it directly is a
 * hundred lines rather than a dependency.
 *
 * Forked from `spike/miocodec/safetensors.ts` with one addition: **BF16**.
 * The MioCodec checkpoint is f32 throughout; this LM's is bf16 throughout, so
 * refusing everything but F32 would refuse the whole file. BF16 is exactly the
 * top 16 bits of an f32 — same sign, same 8 exponent bits, 7 mantissa bits —
 * so widening is `u16 << 16` into a u32 bit pattern, lossless and exact. The
 * refusal stays for every *other* dtype (F16 in particular is a different bit
 * layout and silently mangling it would produce plausible garbage).
 */

export interface TensorView {
  readonly data: Float32Array;
  readonly shape: readonly number[];
}

interface HeaderEntry {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

const BYTES_PER: Record<string, number> = { F32: 4, BF16: 2 };

export class Safetensors {
  private constructor(
    private readonly header: Record<string, HeaderEntry>,
    private readonly buffer: ArrayBuffer,
    private readonly dataStart: number,
  ) {}

  static parse(buffer: ArrayBuffer): Safetensors {
    if (buffer.byteLength < 8) {
      throw new Error(`not safetensors: ${buffer.byteLength} bytes is shorter than the header length`);
    }
    const view = new DataView(buffer);
    // u64, read as BigInt and converted: the header is never near 2^53, but
    // doing the arithmetic in BigInt is the only way to say that honestly.
    const headerLength = Number(view.getBigUint64(0, true));
    if (headerLength <= 0 || 8 + headerLength > buffer.byteLength) {
      throw new Error(`header claims ${headerLength} bytes, file has ${buffer.byteLength}`);
    }
    const json = new TextDecoder().decode(new Uint8Array(buffer, 8, headerLength));
    const header = JSON.parse(json) as Record<string, HeaderEntry>;
    delete (header as Record<string, unknown>).__metadata__;
    return new Safetensors(header, buffer, 8 + headerLength);
  }

  names(): string[] {
    return Object.keys(this.header);
  }

  has(name: string): boolean {
    return name in this.header;
  }

  /**
   * One tensor as f32, whether stored as F32 or BF16.
   *
   * Copied rather than viewed, for F32 too. A view would alias the whole
   * checkpoint — a gigabyte held alive by one 1024-element gamma — and
   * `Float32Array` over an unaligned offset throws anyway, which safetensors
   * offsets are free to be. BF16 has to materialise a new buffer regardless.
   */
  tensor(name: string): TensorView {
    const entry = this.header[name];
    if (!entry) {
      throw new Error(`no tensor "${name}" in the checkpoint`);
    }
    const bytesPer = BYTES_PER[entry.dtype];
    if (!bytesPer) {
      throw new Error(`"${name}" is ${entry.dtype}; only F32 and BF16 are read here`);
    }
    const [start, end] = entry.data_offsets;
    const count = entry.shape.reduce((a, b) => a * b, 1);
    if (end - start !== count * bytesPer) {
      throw new Error(
        `"${name}" spans ${end - start} bytes but its ${entry.dtype} shape ` +
          `${entry.shape.join("x")} needs ${count * bytesPer}`,
      );
    }
    const bytes = new Uint8Array(this.buffer, this.dataStart + start, end - start);
    if (entry.dtype === "F32") {
      const copy = new Uint8Array(bytes.byteLength);
      copy.set(bytes);
      return { data: new Float32Array(copy.buffer), shape: entry.shape };
    }
    // BF16 -> F32: place each u16 in the high half of a u32 and reinterpret.
    // Byte-by-byte assembly (little-endian) because the source span may be
    // unaligned for a Uint16Array view.
    const wide = new Uint32Array(count);
    for (let i = 0; i < count; i += 1) {
      wide[i] = (bytes[2 * i]! | (bytes[2 * i + 1]! << 8)) << 16;
    }
    return { data: new Float32Array(wide.buffer), shape: entry.shape };
  }
}
