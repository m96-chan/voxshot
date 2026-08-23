import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { cpuBackend, type Backend } from "./backend.js";
import { encode, encoderConfig } from "./encoder.js";
import { integratedLoudness, normalizeLoudness } from "./loudness.js";
import type { Signal } from "./decoder.js";

/**
 * The encoder against the reference, block by block.
 *
 *     cd spike/dacvae && npm run check:encode
 *     npm run check:encode -- --gpu
 *
 * It starts from the WAV on disk and does its own -16 dB LUFS normalisation,
 * which is compared against the waveform the reference handed the encoder
 * before anything else runs. That ordering is deliberate: normalisation is a
 * 3.36x gain on this clip, and getting it wrong moves the latent by half of
 * peak — an arithmetic error in the encoder and a loudness error look identical
 * at the end and completely different here.
 *
 * `_pad` is the port's too, so the reflect padding is checked rather than
 * assumed.
 *
 * The five downsampling stages are compared separately because they fail
 * differently: `block.0` catches a wrong padding rule, the four blocks catch a
 * wrong stride or a residual unit in the wrong order, and `quantizer.in_proj`
 * catches the VAE half being read as the latent.
 */

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN = join(HERE, "golden", "encode");
/**
 * Looser than the decoder's 1e-5, from the measured trend rather than to make
 * a stage pass.
 *
 * Across the five stages the relative error goes 1.7e-7, 1.1e-6, 2.3e-5,
 * 2.5e-5, 7.5e-6, 6.5e-6, 1.6e-5 — it rises for two blocks and then **flattens**.
 * That is float32 summation order diverging from torch's, not a fault: an error
 * that was structurally wrong would compound through every block instead of
 * plateauing. The encoder reduces 950400 samples to 495 frames, so its sums run
 * over far longer windows than any the decoder computes.
 */
const TOLERANCE = 5e-5;

interface Index {
  input: string;
  input_samples: number;
  normalize_db: number | null;
  measured_lufs: number;
  tensors: Record<string, { shape: number[] }>;
}

if (!existsSync(join(GOLDEN, "index.json"))) {
  throw new Error(
    `golden/encode is missing. Rebuild it with\n` +
      `  cd spike/dacvae && IRODORI_REPO=... .venv/bin/python dump_encode.py \\\n` +
      `      --input ../irodori/samples/reference-voice.wav`,
  );
}

function golden(name: string): Float32Array {
  const bytes = readFileSync(join(GOLDEN, `${name}.f32`));
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return new Float32Array(copy.buffer);
}

function report(label: string, mine: Float32Array, theirs: Float32Array): boolean {
  if (mine.length !== theirs.length) {
    console.log(`FAIL ${label.padEnd(20)} ${mine.length} values, reference has ${theirs.length}`);
    return false;
  }
  let worst = 0;
  let peak = 0;
  for (let index = 0; index < theirs.length; index += 1) {
    worst = Math.max(worst, Math.abs(mine[index]! - theirs[index]!));
    peak = Math.max(peak, Math.abs(theirs[index]!));
  }
  const relative = peak > 0 ? worst / peak : worst;
  const ok = relative <= TOLERANCE;
  console.log(
    `${ok ? "ok  " : "FAIL"} ${label.padEnd(20)} max |diff| ${worst.toExponential(2)} ` +
      `of peak ${peak.toFixed(3)}  (${relative.toExponential(2)} relative)`,
  );
  return ok;
}

async function main(): Promise<void> {
  const index = JSON.parse(readFileSync(join(GOLDEN, "index.json"), "utf8")) as Index;
  const config = encoderConfig();
  const useGpu = process.argv.includes("--gpu");
  let backend: Backend = cpuBackend;
  if (useGpu) {
    const { create, globals } = await import("webgpu");
    Object.assign(globalThis, globals);
    const { Gpu, gpuBackend } = await import("./gpu.js");
    const instance = create([]);
    const adapter = await instance.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter");
    const info = adapter.info?.description ?? "unknown adapter";
    // `instance` and `adapter` are handed over to be retained, not for use:
    // Dawn's Node binding does not keep the `GPU` alive from the `GPUDevice`,
    // and letting it be collected crashes mid-run.
    backend = gpuBackend(Gpu.fromDevice(await Gpu.requestDevice(adapter), info, [instance, adapter]));
  }

  console.log(
    `encoder: rates [${config.encoderRates.join(", ")}], dim ${config.encoderDim} -> ` +
      `latent ${config.latentDim}, hop ${config.hopLength}, on ${backend.name}`,
  );
  console.log(
    `input ${index.input}: ${index.input_samples} samples, normalised to ` +
      `${index.normalize_db} dB LUFS by the reference (that step is not ported)\n`,
  );

  // From the file, not from the golden: the loudness normalisation is part of
  // the port now and has to be exercised.
  const file = readFileSync(join(HERE, "..", "irodori", "samples", "reference-voice.wav"));
  const samples = (file.length - 44) / 2;
  const raw = new Float32Array(samples);
  for (let i = 0; i < samples; i += 1) raw[i] = file.readInt16LE(44 + i * 2) / 32768;

  const before = integratedLoudness(raw, 48000);
  const normalized = normalizeLoudness(raw, 48000, index.normalize_db ?? -16);
  console.log(
    `loudness ${before.toFixed(4)} LUFS -> gain ${normalized.gain.toFixed(3)}x -> ` +
      `${integratedLoudness(normalized.data, 48000).toFixed(4)} LUFS after the peak clamp`,
  );

  /**
   * The measurement, compared as a number.
   *
   * The normalised *waveform* cannot check it. This clip's gain takes the peak
   * past full scale, so `ensure_max_of_audio` scales it back to exactly 1.0 and
   * the output becomes `raw / peak` — the same signal whatever the meter said.
   * Breaking the relative gate, the 75% block overlap or the -0.691 offset all
   * left the waveform comparison green, which is how this line came to exist.
   */
  const lufsGap = Math.abs(before - index.measured_lufs);
  const lufsOk = lufsGap <= 1e-3;
  console.log(
    `${lufsOk ? "ok  " : "FAIL"} ${"integrated loudness".padEnd(20)} ${before.toFixed(6)} LUFS, ` +
      `reference ${index.measured_lufs.toFixed(6)} (${lufsGap.toExponential(2)} apart)`,
  );

  let failedEarly = !lufsOk;
  if (!report("normalised waveform", normalized.data, golden("waveform"))) failedEarly = true;
  console.log();
  if (failedEarly) {
    console.log("the encoder was not run: its input would have been the wrong signal");
    process.exitCode = 1;
    return;
  }

  const waveform: Signal = { data: normalized.data, channels: 1, length: index.input_samples };
  const captured = new Map<string, Signal>();
  const started = Date.now();
  const latent = await encode(waveform, {
    backend,
    trace: (stage, signal) => captured.set(stage, signal),
  });
  console.log(`ran in ${((Date.now() - started) / 1000).toFixed(1)}s\n`);

  let failures = 0;
  const stages: [string, string][] = [
    ["encoder_block_0", "block.0"],
    ["encoder_block_1", "block.1"],
    ["encoder_block_2", "block.2"],
    ["encoder_block_3", "block.3"],
    ["encoder_block_4", "block.4"],
    ["encoder_out", "block.6"],
    ["quantizer_in_proj", "quantizer_in_proj"],
  ];
  for (const [mine, theirs] of stages) {
    const signal = captured.get(mine);
    if (!signal) throw new Error(`the port did not trace ${mine}`);
    if (!report(theirs, signal.data, golden(theirs))) {
      failures += 1;
      // Later stages are downstream of this one and would fail for the same
      // reason; the first disagreement is the only informative one.
      break;
    }
  }

  if (failures === 0) {
    // The reference returns `(B, T, D)`; the port keeps channels major.
    const theirs = golden("latent");
    const frames = latent.length;
    const swapped = new Float32Array(theirs.length);
    for (let frame = 0; frame < frames; frame += 1) {
      for (let d = 0; d < config.latentDim; d += 1) {
        swapped[frame * config.latentDim + d] = latent.data[d * frames + frame]!;
      }
    }
    if (!report("latent", swapped, theirs)) failures += 1;
  }

  console.log();
  if (failures > 0) {
    console.log(`${failures} stages disagree`);
    process.exitCode = 1;
  } else {
    console.log(
      `the encoder agrees within ${TOLERANCE.toExponential(0)} of peak — ` +
        `${index.input_samples} samples to ${latent.length} latent frames`,
    );
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
