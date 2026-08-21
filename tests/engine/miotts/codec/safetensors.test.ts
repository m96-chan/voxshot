import { describe, expect, it } from "vitest";

import { Safetensors } from "../../../../src/engine/miotts/codec/safetensors.js";
import { buildSafetensors } from "../../../helpers/safetensors.js";

const tensors = {
  "decoder.bias": { shape: [3], data: Float32Array.from([1, 2, 3]) },
  "decoder.weight": { shape: [2, 2], data: Float32Array.from([4, 5, 6, 7]) },
};

describe("Safetensors", () => {
  it("reads a tensor back by name, with its shape", () => {
    const file = Safetensors.parse(buildSafetensors(tensors));

    expect(file.tensor("decoder.weight").shape).toEqual([2, 2]);
    expect(Array.from(file.tensor("decoder.weight").data)).toEqual([4, 5, 6, 7]);
  });

  it("copies rather than viewing, so one gamma cannot pin the checkpoint", () => {
    // A view would alias the whole file — half a gigabyte held alive by a
    // 512-element bias — and Float32Array over an unaligned offset throws
    // anyway, which safetensors offsets are free to be.
    const buffer = buildSafetensors(tensors);
    const file = Safetensors.parse(buffer);
    const tensor = file.tensor("decoder.bias");

    expect(tensor.data.byteLength).toBe(3 * 4);
    expect(tensor.data.buffer.byteLength).toBe(3 * 4);
    expect(tensor.data.buffer).not.toBe(buffer);
  });

  it("lists its tensors and answers whether one is present", () => {
    const file = Safetensors.parse(buildSafetensors(tensors));

    expect(file.names()).toEqual(["decoder.bias", "decoder.weight"]);
    expect(file.has("decoder.bias")).toBe(true);
    expect(file.has("decoder.missing")).toBe(false);
  });

  it("does not report __metadata__ as a tensor", () => {
    const file = Safetensors.parse(buildSafetensors(tensors, { metadata: { format: "pt" } }));

    expect(file.names()).not.toContain("__metadata__");
  });

  it("refuses a file too short to hold a header length", () => {
    expect(() => Safetensors.parse(new ArrayBuffer(4))).toThrow(/shorter than the header/);
  });

  it("refuses a header length that runs past the end of the file", () => {
    const buffer = buildSafetensors(tensors);
    new DataView(buffer).setBigUint64(0, BigInt(buffer.byteLength * 2), true);

    expect(() => Safetensors.parse(buffer)).toThrow(/header claims/);
  });

  it("refuses a header length of zero", () => {
    const buffer = buildSafetensors(tensors);
    new DataView(buffer).setBigUint64(0, 0n, true);

    expect(() => Safetensors.parse(buffer)).toThrow(/header claims/);
  });

  it("names a tensor that is not in the checkpoint", () => {
    const file = Safetensors.parse(buildSafetensors(tensors));

    expect(() => file.tensor("decoder.absent")).toThrow(/no tensor "decoder.absent"/);
  });

  it("refuses a dtype it would have to widen", () => {
    // Silently widening BF16 or F16 would hand back plausible numbers carrying
    // half the precision the caller assumed, and every later comparison would
    // chase that instead of the bug it was written for.
    const file = Safetensors.parse(buildSafetensors(tensors, { dtype: "BF16" }));

    expect(() => file.tensor("decoder.bias")).toThrow(/only F32 is read here/);
  });

  it("refuses a byte range that disagrees with the declared shape", () => {
    // A truncated download, or a shape edited without re-dumping: either way
    // the tensor read would be silently short.
    const file = Safetensors.parse(
      buildSafetensors({ "decoder.bias": { shape: [4], data: Float32Array.from([1, 2, 3]) } }),
    );

    expect(() => file.tensor("decoder.bias")).toThrow(/spans 12 bytes but its shape 4 needs 16/);
  });
});
