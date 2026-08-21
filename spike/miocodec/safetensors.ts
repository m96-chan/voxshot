/**
 * The slice of safetensors this decoder needs, and no more.
 *
 * The format is a little-endian u64 header length, that many bytes of JSON
 * naming every tensor's dtype, shape and byte range, then the data. There is no
 * compression and no framing beyond that, which is why reading it directly is a
 * hundred lines rather than a dependency — and why the same code works against
 * a `Buffer` in Node and an `ArrayBuffer` from `fetch` in a browser.
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
    // u64, read as two u32s: the header is never near 2^53, but doing the
    // arithmetic in BigInt and converting is the only way to say that honestly.
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
   * One tensor as f32.
   *
   * Only `F32` is accepted. The alternative — silently widening an `F16` or a
   * `BF16` — would produce a tensor of entirely plausible numbers carrying half
   * the precision the caller assumed, and every later comparison would be
   * chasing that instead of the bug it was written for. This checkpoint is f32
   * throughout; a rung that is not should say so here rather than downstream.
   */
  tensor(name: string): TensorView {
    const entry = this.header[name];
    if (!entry) {
      throw new Error(`no tensor "${name}" in the checkpoint`);
    }
    if (entry.dtype !== "F32") {
      throw new Error(`"${name}" is ${entry.dtype}; only F32 is read here`);
    }
    const [start, end] = entry.data_offsets;
    const count = entry.shape.reduce((a, b) => a * b, 1);
    if (end - start !== count * 4) {
      throw new Error(
        `"${name}" spans ${end - start} bytes but its shape ${entry.shape.join("x")} needs ${count * 4}`,
      );
    }
    // Copied rather than viewed. A view would alias the whole checkpoint —
    // half a gigabyte held alive by one 512-element bias — and `Float32Array`
    // over an unaligned offset throws anyway, which safetensors offsets are
    // free to be.
    const bytes = new Uint8Array(this.buffer, this.dataStart + start, end - start);
    const copy = new Uint8Array(bytes.byteLength);
    copy.set(bytes);
    return { data: new Float32Array(copy.buffer), shape: entry.shape };
  }
}
