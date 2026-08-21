import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { create, globals } from "webgpu";

import { cpuBackend, type Backend } from "./backend.js";
import { decode, decoderConfig, type Signal } from "./decoder.js";
import { Gpu, gpuBackend } from "./gpu.js";

/**
 * The port against Meta's reference, stage by stage.
 *
 *     cd spike/dacvae && npm run check
 *
 * Stage by stage rather than end to end, because the decoder upsamples 1920x
 * and an error introduced in the first block is unrecognisable by the last —
 * "the waveform is wrong" would be true and useless. Each block boundary is
 * compared on its own, so the first line that goes red is the first thing that
 * is actually broken.
 *
 * ## What counts as agreement
 *
 * Peak-relative: the worst absolute difference over the largest magnitude the
 * reference produced at that stage. Absolute error is meaningless to compare
 * across stages whose scales differ by three orders of magnitude, and relative
 * error per element explodes wherever the reference is near zero.
 *
 * The threshold is f32 accumulation noise, not a quality judgement. These are
 * the same operations in the same order over the same weights; anything past
 * ~1e-5 peak-relative is a different computation, not a rounding difference.
 *
 * `--gpu` runs the same graph through the WGSL kernels instead. The comparison
 * is against the same goldens, so the two backends are held to the reference
 * rather than to each other — "they agree" would be satisfied by two identical
 * mistakes.
 */

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN = join(HERE, "golden");

/** f32 accumulation over a few thousand terms, with room to spare. */
const TOLERANCE = 1e-5;

interface GoldenIndex {
  config: { sample_rate: number; hop_length: number };
  tensors: Record<string, { shape: number[]; bytes: number }>;
}

function loadIndex(): GoldenIndex {
  try {
    return JSON.parse(readFileSync(join(GOLDEN, "index.json"), "utf8")) as GoldenIndex;
  } catch {
    throw new Error(
      "golden/index.json is missing. The goldens are deliberately not in git; rebuild them with\n" +
        "  cd spike/dacvae && .venv/bin/python dump_golden.py",
    );
  }
}

function loadTensor(name: string): { data: Float32Array; shape: number[] } {
  const meta = loadIndex().tensors[name];
  if (!meta) throw new Error(`golden has no tensor "${name}"`);
  const bytes = readFileSync(join(GOLDEN, `${name}.f32`));
  const data = new Float32Array(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  return { data, shape: meta.shape };
}

interface Difference {
  worst: number;
  peak: number;
  relative: number;
  at: number;
}

function compare(mine: Float32Array, reference: Float32Array): Difference {
  if (mine.length !== reference.length) {
    throw new Error(`length ${mine.length} vs reference ${reference.length}`);
  }
  let worst = 0;
  let peak = 0;
  let at = -1;
  for (let i = 0; i < mine.length; i += 1) {
    const expected = reference[i]!;
    const difference = Math.abs(mine[i]! - expected);
    if (difference > worst) {
      worst = difference;
      at = i;
    }
    const magnitude = Math.abs(expected);
    if (magnitude > peak) peak = magnitude;
  }
  return { worst, peak, relative: peak > 0 ? worst / peak : worst, at };
}

async function pickBackend(): Promise<Backend> {
  if (!process.argv.includes("--gpu")) return cpuBackend;
  Object.assign(globalThis, globals);
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
  return gpuBackend(Gpu.fromDevice(device, info));
}

async function main(): Promise<void> {
  const config = decoderConfig();
  const backend = await pickBackend();
  const golden = loadIndex();
  if (golden.config.sample_rate !== config.sampleRate || golden.config.hop_length !== config.hopLength) {
    throw new Error(
      `golden is ${golden.config.sample_rate} Hz / hop ${golden.config.hop_length}, ` +
        `weights say ${config.sampleRate} / ${config.hopLength}`,
    );
  }

  const latentGolden = loadTensor("latent_mean");
  const [, channels, frames] = latentGolden.shape as [number, number, number];
  const latent: Signal = { data: latentGolden.data, channels, length: frames };

  console.log(
    `${config.sampleRate} Hz, hop ${config.hopLength}, rates [${config.decoderRates.join(", ")}]`,
  );
  console.log(`latent [${channels}, ${frames}] -> waveform [1, ${frames * config.hopLength}]`);
  console.log(`backend: ${backend.name}\n`);

  const results: { stage: string; difference: Difference; ok: boolean }[] = [];
  const seen = new Set<string>();

  const started = Date.now();
  await decode(latent, {
    backend,
    trace(stage, signal) {
      seen.add(stage);
      const reference = loadTensor(stage);
      const difference = compare(signal.data, reference.data);
      const ok = difference.relative <= TOLERANCE;
      results.push({ stage, difference, ok });
      const shape = `[${signal.channels}, ${signal.length}]`;
      console.log(
        `${ok ? "ok  " : "FAIL"}  ${stage.padEnd(20)} ${shape.padEnd(16)} ` +
          `peak-relative ${difference.relative.toExponential(3)}  ` +
          `(worst ${difference.worst.toExponential(3)} of peak ${difference.peak.toExponential(3)})`,
      );
    },
  });

  // Every golden stage has to have been visited. A port that silently skipped
  // one would otherwise pass on the stages it did run.
  const missed = Object.keys(golden.tensors).filter(
    (name) => !seen.has(name) && name.startsWith("decoder_model_"),
  );
  if (missed.length > 0) {
    console.log(`\nFAIL  stages in the golden that the port never produced: ${missed.join(", ")}`);
  }

  const elapsed = (Date.now() - started) / 1000;
  const seconds = (frames * config.hopLength) / config.sampleRate;
  const failed = results.filter((r) => !r.ok);
  console.log();
  console.log(
    `${seconds.toFixed(2)} s of audio in ${elapsed.toFixed(2)} s ` +
      `— RTF ${(elapsed / seconds).toFixed(2)} (${backend.name})`,
  );
  if (failed.length === 0 && missed.length === 0) {
    const worst = results.reduce((a, b) => (a.difference.relative > b.difference.relative ? a : b));
    console.log(
      `all ${results.length} stages agree within ${TOLERANCE.toExponential(0)} peak-relative ` +
        `(worst: ${worst.stage} at ${worst.difference.relative.toExponential(3)})`,
    );
    return;
  }
  console.log(`${failed.length} stage(s) disagree; the first is ${failed[0]!.stage}`);
  process.exitCode = 1;
}

await main();
