import { describe, expect, it } from "vitest";
import { Gpu } from "../../src/engine/miotts/codec/gpu.js";

/**
 * The GPU seam's *validation* only — no device, no Dawn (which kills a Vitest
 * worker before a single test runs; see vitest.config.ts). The dummy device
 * below is never touched: the divisibility check must throw before the first
 * upload, and a test that reached `createBuffer` on this object would fail
 * with its own loud TypeError instead of the expected message.
 */
describe("Gpu.conv1d channel/group validation", () => {
  const gpu = Gpu.fromDevice({} as unknown as GPUDevice, "validation-only dummy");
  const input = new Float32Array(12);
  const weight = new Float32Array(12);

  it("throws when Cin is not divisible by groups", async () => {
    // Cin=3, groups=2: the reference conv1d throws here; the WGSL uniform is a
    // Uint32Array, where 1.5 would truncate to 1 silently.
    await expect(gpu.conv1d(input, weight, null, 3, 2, 4, 1, 0, 1, 2)).rejects.toThrow(
      /Cin=3 and Cout=2 must both be divisible by groups=2/,
    );
  });

  it("throws when Cout is not divisible by groups", async () => {
    await expect(gpu.conv1d(input, weight, null, 4, 3, 3, 1, 0, 1, 2)).rejects.toThrow(
      /Cin=4 and Cout=3 must both be divisible by groups=2/,
    );
  });
});
