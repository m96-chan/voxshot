import { describe, expect, it } from "vitest";

import { encodeWav } from "../../src/audio/wav.js";
import { InvalidInputError } from "../../src/errors.js";

function readAscii(view: DataView, offset: number, length: number): string {
  let out = "";
  for (let index = 0; index < length; index += 1) {
    out += String.fromCharCode(view.getUint8(offset + index));
  }
  return out;
}

describe("encodeWav", () => {
  it("writes a RIFF/WAVE header", () => {
    const buffer = encodeWav(new Float32Array([0, 0.5, -0.5]), 16_000);
    const view = new DataView(buffer);

    expect(readAscii(view, 0, 4)).toBe("RIFF");
    expect(readAscii(view, 8, 4)).toBe("WAVE");
    expect(readAscii(view, 12, 4)).toBe("fmt ");
    expect(readAscii(view, 36, 4)).toBe("data");
  });

  it("describes 16 bit mono PCM at the given sample rate", () => {
    const view = new DataView(encodeWav(new Float32Array([0]), 24_000));

    expect(view.getUint32(16, true)).toBe(16); // fmt chunk size
    expect(view.getUint16(20, true)).toBe(1); // PCM
    expect(view.getUint16(22, true)).toBe(1); // channels
    expect(view.getUint32(24, true)).toBe(24_000); // sample rate
    expect(view.getUint32(28, true)).toBe(48_000); // byte rate
    expect(view.getUint16(32, true)).toBe(2); // block align
    expect(view.getUint16(34, true)).toBe(16); // bits per sample
  });

  it("sizes the buffer as header plus two bytes per sample", () => {
    const buffer = encodeWav(new Float32Array(10), 16_000);

    expect(buffer.byteLength).toBe(44 + 20);
    expect(new DataView(buffer).getUint32(4, true)).toBe(36 + 20);
    expect(new DataView(buffer).getUint32(40, true)).toBe(20);
  });

  it("converts float samples to signed 16 bit", () => {
    const view = new DataView(encodeWav(new Float32Array([0, 1, -1]), 8_000));

    expect(view.getInt16(44, true)).toBe(0);
    expect(view.getInt16(46, true)).toBe(32_767);
    expect(view.getInt16(48, true)).toBe(-32_768);
  });

  it("clamps samples outside the -1..1 range", () => {
    const view = new DataView(encodeWav(new Float32Array([4, -4]), 8_000));

    expect(view.getInt16(44, true)).toBe(32_767);
    expect(view.getInt16(46, true)).toBe(-32_768);
  });

  it("encodes an empty signal as a header only file", () => {
    expect(encodeWav(new Float32Array(0), 16_000).byteLength).toBe(44);
  });

  it("rejects a non positive sample rate", () => {
    expect(() => encodeWav(new Float32Array([0]), 0)).toThrow(InvalidInputError);
  });
});
