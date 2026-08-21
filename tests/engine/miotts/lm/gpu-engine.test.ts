import { beforeEach, describe, expect, it } from "vitest";

import { createGpuEngine } from "../../../../src/engine/miotts/lm/gpu-engine.js";
import { loadWeightsQ8 } from "../../../../src/engine/miotts/lm/weights-q8.js";
import { createFakeGpu, type FakeGpu } from "../../../helpers/fake-gpu.js";
import { TINY_CONFIG, buildSyntheticQ8 } from "../../../helpers/q8.js";

/**
 * The decode loop's *shape*, checked with a `GPUDevice` that records instead of
 * computing.
 *
 * Everything this engine claims about cost is a claim about wiring: buffers are
 * built once at construction, a decode step is one submit, and the only thing
 * that crosses back from the GPU is the logits. Those are voxshot's own
 * decisions — the kernels are web-xpu-ops' and are tested there — and they are
 * checkable on a machine with no GPU.
 *
 * What is deliberately absent is any claim about the numbers. The fake reads
 * back zeros. Whether the engine computes Qwen3 is what the real weights and
 * `npm run test:models` are for.
 */

const weights = loadWeightsQ8(buildSyntheticQ8());
const maxSeqLen = 16;

let gpu: FakeGpu;

beforeEach(() => {
  gpu = createFakeGpu();
});

describe("createGpuEngine", () => {
  it("builds every buffer up front and none inside the decode loop", async () => {
    // The whole point of a resident engine. An allocation per step would mean
    // ~600 MiB of churn per token on the real model.
    const engine = await createGpuEngine(gpu.device, weights, { maxSeqLen });
    const afterBuild = gpu.log.buffers.length;

    await engine.decodeStep(0);
    await engine.decodeStep(1);
    await engine.decodeStep(2);

    expect(gpu.log.buffers).toHaveLength(afterBuild);
    expect(engine.stats.buffersCreated).toBe(afterBuild);
  });

  it("compiles each distinct kernel once, not once per layer", async () => {
    // Seven entry points across 28 layers on the real model. One module per
    // source, reused by every layer's pipeline.
    await createGpuEngine(gpu.device, weights, { maxSeqLen });

    expect(gpu.log.shaderModules).toBeLessThanOrEqual(7);
    expect(gpu.log.pipelines).toBeGreaterThan(0);
  });

  it("charges VRAM linearly in maxSeqLen, and the KV cache is most of it", async () => {
    // This is what makes `maxSeqLen` worth exposing: on the real model 768
    // positions cost ~176 MiB and 256 cost ~59 MiB, which only holds if the
    // relationship is linear. Measured as equal increments rather than against
    // a formula — a formula here would just be the implementation's buffer
    // list copied into the test, and would pass however wrong both were.
    const bytesAt = async (maxSeqLen: number) => {
      const fake = createFakeGpu();
      await createGpuEngine(fake.device, weights, { maxSeqLen });
      return fake.log.buffers.reduce((sum, buffer) => sum + buffer.size, 0);
    };
    const [at16, at32, at48] = await Promise.all([bytesAt(16), bytesAt(32), bytesAt(48)]);

    expect(at32 - at16).toBe(at48 - at32);

    // The cache itself: k and v, per layer, per kv head. The rest of what
    // scales is attention scratch, and should stay small beside it.
    const kvPerPosition =
      2 * TINY_CONFIG.numLayers * TINY_CONFIG.numKvHeads * TINY_CONFIG.headDim * 4;
    const perPosition = (at32 - at16) / 16;
    expect(perPosition).toBeGreaterThanOrEqual(kvPerPosition);
    expect(perPosition).toBeLessThan(2 * kvPerPosition);
  });
});

describe("decodeStep", () => {
  it("records one submit per token", async () => {
    const engine = await createGpuEngine(gpu.device, weights, { maxSeqLen });
    gpu.resetCounters();

    await engine.decodeStep(0);

    expect(gpu.log.submits).toBe(1);
  });

  it("reads back the logits and nothing else", async () => {
    // One readback per step, of exactly `vocabSize` floats. Anything else
    // crossing the bus would be a stall the engine does not account for.
    const engine = await createGpuEngine(gpu.device, weights, { maxSeqLen });
    gpu.resetCounters();

    const logits = await engine.decodeStep(0);

    expect(logits).toHaveLength(TINY_CONFIG.vocabSize);
    expect(gpu.log.bytesRead).toBe(TINY_CONFIG.vocabSize * 4);
    expect(engine.stats.readbackBytesPerStep).toBe(TINY_CONFIG.vocabSize * 4);
  });

  it("does the same amount of work on every step", async () => {
    // A step whose cost depended on the position would make throughput drift
    // over a generation. The attention scan is bounded by a uniform instead.
    const engine = await createGpuEngine(gpu.device, weights, { maxSeqLen });

    const measure = async (id: number) => {
      gpu.resetCounters();
      await engine.decodeStep(id);
      return { dispatches: gpu.log.dispatches, copies: gpu.log.copies.length };
    };

    const first = await measure(0);
    const later = await measure(3);

    expect(later).toEqual(first);
    expect(engine.stats.dispatchesPerStep).toBe(first.dispatches);
    // `copiesPerStep` counts the KV-cache appends only — the staging copies
    // that carry the logits out are not part of the per-layer work, and the
    // stats say so by not counting them. Hence the inequality rather than an
    // equality: the encoder records both.
    expect(first.copies).toBeGreaterThan(engine.stats.copiesPerStep);
    expect(engine.stats.copiesPerStep).toBe(2 * TINY_CONFIG.numLayers);
  });

  it("skips the vocabulary projection and the readback on a prompt token", async () => {
    // Prefill is the decode path token by token, and only the last token's
    // logits are wanted. Every earlier step should skip the 164k-row matvec
    // and its 657 KB readback entirely.
    const engine = await createGpuEngine(gpu.device, weights, { maxSeqLen });

    gpu.resetCounters();
    const full = await engine.decodeStep(0);
    const fullWork = { dispatches: gpu.log.dispatches, read: gpu.log.bytesRead };

    gpu.resetCounters();
    const skipped = await engine.decodeStep(1, { skipLogits: true });

    expect(full).not.toBeNull();
    expect(skipped).toBeNull();
    expect(gpu.log.bytesRead).toBe(0);
    expect(gpu.log.dispatches).toBeLessThan(fullWork.dispatches);
    expect(gpu.log.submits).toBe(1);
  });

  it("still writes the KV cache on a skipped step", async () => {
    // The point of a prefill step IS the cache write; skipping the logits must
    // not skip that too, or the prompt would not be in the context.
    const engine = await createGpuEngine(gpu.device, weights, { maxSeqLen });

    gpu.resetCounters();
    await engine.decodeStep(0, { skipLogits: true });

    expect(gpu.log.copies.length).toBeGreaterThanOrEqual(2 * TINY_CONFIG.numLayers);
  });

  it("advances one position per step", async () => {
    const engine = await createGpuEngine(gpu.device, weights, { maxSeqLen });

    expect(engine.position).toBe(0);
    await engine.decodeStep(0);
    await engine.decodeStep(1, { skipLogits: true });

    expect(engine.position).toBe(2);
  });

  it("uploads one embedding row per step", async () => {
    // The gather runs on the CPU and the row is written into a device buffer:
    // hiddenSize floats, and nothing else, per token.
    const engine = await createGpuEngine(gpu.device, weights, { maxSeqLen });
    gpu.resetCounters();

    await engine.decodeStep(2);

    expect(gpu.log.bytesWritten).toBeGreaterThanOrEqual(TINY_CONFIG.hiddenSize * 4);
  });
});

describe("reset", () => {
  it("rewinds the position without reallocating anything", async () => {
    // Weights stay resident across generations; only the position moves. The
    // cache is overwritten in place and every scan is bounded by the current
    // position, so stale tail entries are never read.
    const engine = await createGpuEngine(gpu.device, weights, { maxSeqLen });
    await engine.decodeStep(0);
    await engine.decodeStep(1);
    const afterUse = gpu.log.buffers.length;

    engine.reset();

    expect(engine.position).toBe(0);
    expect(gpu.log.buffers).toHaveLength(afterUse);
  });
});

describe("destroy", () => {
  it("releases every buffer it allocated", async () => {
    const engine = await createGpuEngine(gpu.device, weights, { maxSeqLen });

    engine.destroy();

    expect(gpu.log.buffers.every((buffer) => buffer.destroyed)).toBe(true);
  });
});
