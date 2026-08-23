import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);

import { cpuBackend, type Backend } from "./backend.js";
import { decode, decoderConfig, type Signal } from "./decoder.js";
import { Gpu, gpuBackend } from "./gpu.js";

/**
 * Real speech through the port, and something to listen to.
 *
 *     cd spike/dacvae && .venv/bin/python dump_roundtrip.py --input clip.wav
 *     npm run roundtrip            # WebGPU
 *     npm run roundtrip -- --cpu   # reference implementations
 *
 * `check.ts` already says the port computes what the reference computes, stage
 * by stage. It says so on a synthetic signal, because that is the right input
 * for comparing arithmetic. It therefore says nothing about whether speech
 * survives the codec, and a port could be numerically perfect while the codec
 * itself is unusable at 32 latent dimensions.
 *
 * So this asks the other question, on speech, and writes three files: what went
 * in, what the reference got back, and what the port got back. The numbers
 * below are a summary; the files are the actual answer, and someone has to
 * listen to them.
 *
 * ## Two comparisons, and they mean different things
 *
 * **port against reference** is a claim about this code. It should be at the
 * f32 noise floor, and anything else is a defect here.
 *
 * **round trip against input** is a claim about the *codec*, which no amount of
 * correct porting can improve. It is reported so the two are not confused: a
 * disappointing reconstruction would be Meta's and Aratako's, not ours.
 */

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "roundtrip");

interface Index {
  source: { file: string; start: number; seconds: number };
  config: { sample_rate: number; hop_length: number };
  tensors: Record<string, { shape: number[] }>;
}

const index = (() => {
  try {
    return JSON.parse(readFileSync(join(ROOT, "index.json"), "utf8")) as Index;
  } catch {
    throw new Error(
      "roundtrip/index.json is missing. Build it with\n" +
        "  cd spike/dacvae && .venv/bin/python dump_roundtrip.py --input path/to/48k.wav",
    );
  }
})();

function load(name: string): Float32Array {
  const bytes = readFileSync(join(ROOT, `${name}.f32`));
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

/** 16-bit PCM WAV, so the result can be played without another dependency. */
function writeWav(path: string, samples: Float32Array, sampleRate: number): void {
  const bytes = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(bytes);
  const ascii = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  ascii(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  ascii(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i += 1) {
    // Clamped, not wrapped: the tanh at the end of the decoder keeps values
    // inside [-1, 1], but a clip that wrapped would sound like a fault in the
    // codec rather than in this writer.
    const clamped = Math.max(-1, Math.min(1, samples[i]!));
    view.setInt16(44 + i * 2, Math.round(clamped * 32767), true);
  }
  writeFileSync(path, Buffer.from(bytes));
}

function correlation(a: Float32Array, b: Float32Array): number {
  const n = Math.min(a.length, b.length);
  let sumA = 0;
  let sumB = 0;
  for (let i = 0; i < n; i += 1) {
    sumA += a[i]!;
    sumB += b[i]!;
  }
  const meanA = sumA / n;
  const meanB = sumB / n;
  let cov = 0;
  let varA = 0;
  let varB = 0;
  for (let i = 0; i < n; i += 1) {
    const da = a[i]! - meanA;
    const db = b[i]! - meanB;
    cov += da * db;
    varA += da * da;
    varB += db * db;
  }
  return cov / Math.sqrt(varA * varB);
}

function peakRelative(mine: Float32Array, reference: Float32Array): number {
  let worst = 0;
  let peak = 0;
  for (let i = 0; i < mine.length; i += 1) {
    const difference = Math.abs(mine[i]! - reference[i]!);
    if (difference > worst) worst = difference;
    const magnitude = Math.abs(reference[i]!);
    if (magnitude > peak) peak = magnitude;
  }
  return peak > 0 ? worst / peak : worst;
}

function rms(samples: Float32Array): number {
  let sum = 0;
  for (const value of samples) sum += value * value;
  return Math.sqrt(sum / samples.length);
}

const config = decoderConfig();

/**
 * The device is acquired here, at module top level, and not inside a helper.
 *
 * Not a style choice. Doing the identical sequence inside an `async function`
 * kills Dawn's Node binding — SIGSEGV here, a futex abort in `check.ts` —
 * reproducibly and regardless of input size. Two independent instances of the
 * same shape, so the seam is written down where someone will hit it.
 */
let backend: Backend;
if (process.argv.includes("--cpu")) {
  backend = cpuBackend;
} else {
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
  backend = gpuBackend(Gpu.fromDevice(device, info, [gpu, adapter]));
}
const shape = index.tensors.latent!.shape as [number, number, number];
const latent: Signal = { data: load("latent"), channels: shape[1], length: shape[2] };
const input = load("input");
const reference = load("reference_decode");

const started = Date.now();
const decoded = await decode(latent, { backend });
const elapsed = (Date.now() - started) / 1000;

writeWav(join(ROOT, "input.wav"), input, config.sampleRate);
writeWav(join(ROOT, "reference.wav"), reference, config.sampleRate);
writeWav(join(ROOT, "port.wav"), decoded.data, config.sampleRate);

const audioSeconds = decoded.length / config.sampleRate;
process.stderr.write(
  `${index.source.file}  ${index.source.start}s +${index.source.seconds}s  ` +
    `${config.sampleRate} Hz\n` +
    `backend: ${backend.name}\n\n` +
    `port vs reference : peak-relative ${peakRelative(decoded.data, reference).toExponential(3)}` +
    `   correlation ${correlation(decoded.data, reference).toFixed(6)}\n` +
    `round trip vs input: correlation ${correlation(decoded.data, input).toFixed(4)}` +
    `   rms ${rms(decoded.data).toFixed(4)} against ${rms(input).toFixed(4)}\n\n` +
    `${audioSeconds.toFixed(2)} s of audio in ${elapsed.toFixed(2)} s — RTF ${(elapsed / audioSeconds).toFixed(2)}\n` +
    `wrote ${ROOT}/{input,reference,port}.wav — listen to them\n`,
);
