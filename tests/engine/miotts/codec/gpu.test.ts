import { beforeEach, describe, expect, it } from "vitest";

import { Gpu, gpuBackend } from "../../../../src/engine/miotts/codec/gpu.js";
import { createFakeGpu, type FakeGpu } from "../../../helpers/fake-gpu.js";

/**
 * The codec's GPU backend, checked for the decisions it makes rather than the
 * arithmetic it dispatches.
 *
 * Three of them are load-bearing and none is visible in the output: weights
 * stay resident across calls while activations do not, an absent bias is
 * uploaded as explicit zeros rather than left unbound, and `"same"` padding is
 * resolved to a number here because the kernel has no convention of its own.
 * All three are checkable against a recording device; whether the kernels
 * compute the right thing is web-xpu-ops' question, and whether this port
 * still matches the reference is `npm run test:models`'.
 */

let fake: FakeGpu;
let gpu: Gpu;

beforeEach(() => {
  fake = createFakeGpu();
  gpu = Gpu.fromDevice(fake.device, "fake adapter");
});

describe("residency", () => {
  it("uploads a weight once however many times it is used", async () => {
    // The decoder asks for the same two hundred matrices on every decode.
    // Re-uploading a [1536, 512] matrix per call would move more bytes than
    // the multiply reads.
    const weight = new Float32Array(16).fill(0.5);
    const first = new Float32Array(8);
    const second = new Float32Array(8);

    await gpu.matmul(first, weight, 4, 4, 2);
    const afterFirst = fake.log.bytesWritten;
    await gpu.matmul(second, weight, 4, 4, 2);

    // The second call pays for its activation and its uniform, not for the
    // weight again.
    expect(fake.log.bytesWritten - afterFirst).toBeLessThan(afterFirst);
  });

  it("uploads a different weight separately", async () => {
    const a = new Float32Array(16).fill(0.5);
    const b = new Float32Array(16).fill(0.25);
    const activation = new Float32Array(8);

    await gpu.matmul(activation, a, 4, 4, 2);
    const afterFirst = fake.log.bytesWritten;
    await gpu.matmul(activation, b, 4, 4, 2);

    expect(fake.log.bytesWritten - afterFirst).toBeGreaterThanOrEqual(b.byteLength);
  });

  it("releases the activation buffer but keeps the weight's", async () => {
    // Activations are per call; the weight has to survive for the next one.
    const weight = new Float32Array(16).fill(0.5);
    await gpu.matmul(new Float32Array(8), weight, 4, 4, 2);

    const live = fake.log.buffers.filter((buffer) => !buffer.destroyed);
    expect(live).toHaveLength(1);
    expect(live[0]!.size).toBe(weight.byteLength);
  });

  it("shares one zero bias per width instead of caching a copy per call", async () => {
    // An absent bias is uploaded as explicit zeros — an unreferenced binding
    // throws rather than reading zeros — so without sharing, every biasless
    // conv would leave another array in the resident cache.
    const input = new Float32Array(8);
    const weight = new Float32Array(4).fill(1);

    await gpu.conv1d(input, weight, null, 1, 1, 8, 4, 0);
    const afterFirst = fake.log.buffers.length;
    await gpu.conv1d(input, weight, null, 1, 1, 8, 4, 0);

    // The second call re-uploads only its activation, its output, its uniform
    // and its staging buffer — not the weight and not another zero bias.
    expect(fake.log.buffers.length - afterFirst).toBeLessThan(afterFirst);
  });
});

describe("dispatching", () => {
  it("runs one pass and one submit per op", async () => {
    fake.resetCounters();

    await gpu.matmul(new Float32Array(8), new Float32Array(16), 4, 4, 2);

    expect(fake.log.passes).toBe(1);
    expect(fake.log.submits).toBe(1);
    expect(fake.log.dispatches).toBe(1);
  });

  it("compiles each kernel once and reuses the pipeline", async () => {
    await gpu.matmul(new Float32Array(8), new Float32Array(16), 4, 4, 2);
    const afterFirst = fake.log.pipelines;
    await gpu.matmul(new Float32Array(8), new Float32Array(16), 4, 4, 2);

    expect(fake.log.pipelines).toBe(afterFirst);
  });

  it("reads back exactly the output it asked for", async () => {
    const out = await gpu.matmul(new Float32Array(8), new Float32Array(16), 4, 4, 2);

    expect(out).toHaveLength(4 * 4);
  });
});

describe("conv1d", () => {
  it("computes torch's output length", async () => {
    // floor((L + 2p - (K-1) - 1) / stride) + 1. Getting this wrong sizes the
    // output buffer wrong, and the readback would be short or padded.
    const out = await gpu.conv1d(new Float32Array(10), new Float32Array(3), null, 1, 1, 10, 3, 1);

    expect(out).toHaveLength(10);
  });

  it("accounts for stride", async () => {
    const out = await gpu.conv1d(new Float32Array(10), new Float32Array(3), null, 1, 1, 10, 3, 1, 2);

    expect(out).toHaveLength(5);
  });

  it("refuses channel counts that are not divisible by the groups", async () => {
    // The uniform is a Uint32Array, so a fractional Cin/groups would be
    // truncated silently where the reference conv1d throws.
    await expect(
      gpu.conv1d(new Float32Array(10), new Float32Array(3), null, 3, 4, 10, 3, 1, 1, 2),
    ).rejects.toThrow(/divisible by groups=2/);
  });
});

describe("gpuBackend", () => {
  it("names the adapter it is running on", () => {
    // A silent fallback to the CPU path is the thing to avoid; the caller
    // reports which backend actually ran.
    expect(gpuBackend(gpu).name).toBe("WebGPU (fake adapter)");
  });

  it("resolves 'same' padding to the crop the kernel needs", async () => {
    // The kernel takes a number and has no convention of its own. `"same"`
    // means crop (nFft - hop) / 2 from each end, leaving hop * frames samples
    // — and the length is the observable half of that decision.
    const frames = 4;
    const nFft = 16;
    const hop = 4;
    const bins = nFft / 2 + 1;

    const out = await gpuBackend(gpu).istft(
      new Float32Array(frames * bins),
      new Float32Array(frames * bins),
      new Float32Array(nFft),
      frames,
      nFft,
      hop,
    );

    expect(out).toHaveLength(hop * frames);
  });
});
