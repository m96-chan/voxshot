import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);

import { decode, decoderConfig, type Signal } from "./decoder.js";
import { Gpu, gpuBackend } from "./gpu.js";

/**
 * The same graph on WebGPU, against the same goldens, plus a timing.
 *
 *     cd spike/dacvae && npm run check:gpu
 *
 * Separate from `check.ts` rather than a flag on it, and the reason is a
 * measured environment limitation rather than taste. Dawn's Node binding
 * aborts — `std::system_error: Invalid argument`, reported by glibc as "the
 * futex facility returned an unexpected error code" — when the dispatches are
 * driven from inside a nested async call the way `check.ts` structures them.
 * Bisected: the identical sequence of dispatches at module top level, with the
 * comparisons deferred until after the decode, runs clean and repeatedly.
 * Neither heavy CPU work between dispatches (64 MB of allocate-and-scan per
 * stage) nor file reads between them reproduce it, so the trigger is not the
 * obvious one and is not chased further here — see the README.
 *
 * The consequence for the check is small and arguably an improvement: stages
 * are collected during the decode and compared afterwards, so the timing below
 * measures decoding rather than decoding plus comparison.
 *
 * Both backends are held to the reference, never to each other. "The two agree"
 * would be satisfied by two identical mistakes.
 */

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN = join(HERE, "golden");
const TOLERANCE = 1e-5;

interface GoldenIndex {
  config: { sample_rate: number; hop_length: number };
  tensors: Record<string, { shape: number[] }>;
}

const index = (() => {
  try {
    return JSON.parse(readFileSync(join(GOLDEN, "index.json"), "utf8")) as GoldenIndex;
  } catch {
    throw new Error(
      "golden/index.json is missing. Rebuild it with\n" +
        "  cd spike/dacvae && .venv/bin/python dump_golden.py",
    );
  }
})();

function goldenTensor(name: string): Float32Array {
  const bytes = readFileSync(join(GOLDEN, `${name}.f32`));
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

function peakRelative(mine: Float32Array, reference: Float32Array): number {
  if (mine.length !== reference.length) {
    throw new Error(`length ${mine.length} vs reference ${reference.length}`);
  }
  let worst = 0;
  let peak = 0;
  for (let i = 0; i < mine.length; i += 1) {
    const expected = reference[i]!;
    const difference = Math.abs(mine[i]! - expected);
    if (difference > worst) worst = difference;
    const magnitude = Math.abs(expected);
    if (magnitude > peak) peak = magnitude;
  }
  return peak > 0 ? worst / peak : worst;
}

const config = decoderConfig();
const gpu = create([]);
const adapter = await gpu.requestAdapter();
if (!adapter) throw new Error("no WebGPU adapter — check the driver, not this script");
const device = await adapter.requestDevice({
  requiredLimits: {
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
  },
});
const info = adapter.info
  ? [adapter.info.vendor, adapter.info.architecture].filter(Boolean).join(" ") || "unknown"
  : "unknown";
const backend = gpuBackend(Gpu.fromDevice(device, info));

const latentShape = index.tensors.latent_mean!.shape as [number, number, number];
const latent: Signal = {
  data: goldenTensor("latent_mean"),
  channels: latentShape[1],
  length: latentShape[2],
};
const seconds = (latent.length * config.hopLength) / config.sampleRate;

process.stderr.write(
  `${config.sampleRate} Hz, hop ${config.hopLength}, rates [${config.decoderRates.join(", ")}]\n` +
    `latent [${latent.channels}, ${latent.length}] -> waveform [1, ${latent.length * config.hopLength}]\n` +
    `backend: ${backend.name}\n\n`,
);

// Collected, not compared, while the dispatches are in flight.
const runs = Number(process.env.RUNS ?? 3);
const stages: { stage: string; signal: Signal }[] = [];
const timings: number[] = [];
for (let run = 0; run < runs; run += 1) {
  stages.length = 0;
  const started = Date.now();
  await decode(latent, {
    backend,
    trace: (stage, signal) => {
      if (run === runs - 1) stages.push({ stage, signal });
    },
  });
  timings.push((Date.now() - started) / 1000);
}

let failed = 0;
let worstStage = { stage: "", relative: 0 };
for (const { stage, signal } of stages) {
  const relative = peakRelative(signal.data, goldenTensor(stage));
  const ok = relative <= TOLERANCE;
  if (!ok) failed += 1;
  if (relative > worstStage.relative) worstStage = { stage, relative };
  process.stderr.write(
    `${ok ? "ok  " : "FAIL"}  ${stage.padEnd(20)} [${signal.channels}, ${signal.length}]`.padEnd(52) +
      ` peak-relative ${relative.toExponential(3)}\n`,
  );
}

const best = Math.min(...timings);
process.stderr.write(
  `\n${runs} runs: ${timings.map((t) => t.toFixed(2)).join(" / ")} s` +
    `  — best RTF ${(best / seconds).toFixed(2)} for ${seconds.toFixed(2)} s of audio\n`,
);
if (failed === 0) {
  process.stderr.write(
    `all ${stages.length} stages agree within ${TOLERANCE.toExponential(0)} ` +
      `(worst: ${worstStage.stage} at ${worstStage.relative.toExponential(3)})\n`,
  );
} else {
  process.stderr.write(`${failed} stage(s) disagree\n`);
  process.exitCode = 1;
}
