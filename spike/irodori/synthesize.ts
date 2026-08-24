import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { cpuBackend, type Backend } from "../dacvae/backend.js";
import type { Gpu as IrodoriGpu } from "./gpu.js";
import { decode } from "../dacvae/decoder.js";
import { encode } from "../dacvae/encoder.js";
import { normalizeLoudness } from "../dacvae/loudness.js";
import { conditionNorm, prependMeanToken, project } from "./conditions.js";
import type { Context } from "./dit.js";
import { framesFrom, predictDuration } from "./duration.js";
import { loadBertWeights } from "./bert-weights.js";
import { loadModelWeights } from "./model-weights.js";
import { runModernBert, realLength } from "./modernbert.js";
import { sample } from "./sampler.js";
import { sampleGpu } from "./sampler-gpu.js";
import { patchReference, runSpeakerEncoder } from "./speaker-encoder.js";
import { normalizeText } from "./text.js";
import { loadTokenizer, type TokenizerJson } from "./tokenizer.js";

/**
 * Irodori end to end, in TypeScript, on web-xpu-ops.
 *
 *     cd spike/irodori && npm run say -- "こんにちは、いい天気ですね。"
 *
 * Text in, 48 kHz WAV out. Every stage is this port: normalisation, the Unigram
 * tokenizer, ModernBERT-ja's 25 layers, the projector, the duration predictor,
 * the 12-block DiT under 32 steps of rectified flow, and `spike/dacvae`'s
 * decoder. Nothing calls torch.
 *
 *     npm run say -- "こんにちは。" --ref path/to/voice.wav
 *
 * With `--ref` the clip is encoded here — loudness-normalised to -16 LUFS and
 * run through `spike/dacvae`'s encoder — so the voice is whatever was handed
 * in. Without it, the speaker condition is the latent `dump_golden.py` recorded
 * for `samples/reference-voice.wav`, which is exact and needs no GPU.
 *
 * ## What this is not
 *
 * **It is slow.** web-xpu-ops' CPU reference is the definition of correct and
 * the slowest thing available, and there is no GPU backend for the Irodori
 * half. The sixteen guided flow steps at batch 3 are most of it — about
 * twenty-four minutes for a four-second utterance.
 *
 * The codec does not have to be: `spike/dacvae` has a WebGPU backend, and this
 * uses it when a device is available and falls back to the CPU reference when
 * one is not. Which one ran is printed.
 *
 * That is a gap in the port, not in the model, and it is not hidden behind a
 * plausible-sounding result: the header this prints says which parts ran and on
 * what.
 */

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN = join(HERE, "golden");
const REVISION = "77675fc96a7e445e982e2ba90246b816efc74ec6";
const TOKENIZER_JSON = join(
  homedir(),
  ".cache/huggingface/hub/models--sbintuitions--modernbert-ja-310m/snapshots",
  REVISION,
  "tokenizer.json",
);
const STEPS = 32;
const SAMPLE_RATE = 48000;

function golden(name: string): Float32Array {
  const bytes = readFileSync(join(GOLDEN, `${name}.f32`));
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return new Float32Array(copy.buffer);
}

/** 16-bit PCM, mono. */
function wav(samples: Float32Array, rate: number): Buffer {
  const data = Buffer.alloc(samples.length * 2);
  for (let index = 0; index < samples.length; index += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[index]!));
    data.writeInt16LE(Math.round(clamped * 32767), index * 2);
  }
  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + data.length, 4);
  header.write("WAVEfmt ", 8);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(rate, 24);
  header.writeUInt32LE(rate * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36);
  header.writeUInt32LE(data.length, 40);
  return Buffer.concat([header, data]);
}

/** Zeroed state and mask — how the reference drops a condition for guidance. */
function dropped(context: Context, dim: number): Context {
  return {
    state: new Float32Array(context.keep.length * dim),
    keep: context.keep.map(() => false),
  };
}

/** Stack the conditional first, then each guided variant, on the batch axis. */
function stack(members: Record<string, Context>[], dim: Record<string, number>): Record<string, Context> {
  const out: Record<string, Context> = {};
  for (const name of Object.keys(members[0]!)) {
    const width = dim[name]!;
    const tokens = members[0]![name]!.keep.length;
    const state = new Float32Array(members.length * tokens * width);
    const keep: boolean[] = [];
    members.forEach((member, index) => {
      state.set(member[name]!.state, index * tokens * width);
      keep.push(...member[name]!.keep);
    });
    out[name] = { state, keep };
  }
  return out;
}

/**
 * One device for both halves, or the CPU reference for both.
 *
 * The Irodori half has its own engine now (`gpu.ts`) and the codec has
 * `spike/dacvae`'s, and they share a `GPUDevice` — two devices would mean two
 * copies of every weight. Falling back rather than failing keeps the script
 * runnable where there is no adapter, and which one ran is printed either way:
 * "it produced audio" and "it produced audio in seconds" are different claims.
 */
async function acquire(): Promise<{ codec: Backend; dit: IrodoriGpu | null }> {
  try {
    const { create, globals } = await import("webgpu");
    Object.assign(globalThis, globals);
    const codecGpu = await import("../dacvae/gpu.js");
    const ditGpu = await import("./gpu.js");
    const instance = create([]);
    const adapter = await instance.requestAdapter();
    if (!adapter) return { codec: cpuBackend, dit: null };
    const info = adapter.info?.description ?? "unknown adapter";
    // `requestDevice` asks for the adapter's buffer limits: the codec encoder's
    // first activation is 243 MB for a twenty-second clip, over WebGPU's
    // 128 MiB default.
    const device = await ditGpu.Gpu.requestDevice(adapter);
    // `instance` and `adapter` are retained deliberately: Dawn's Node binding
    // does not keep the `GPU` alive from the `GPUDevice`.
    const retain = [instance, adapter];
    const dit = ditGpu.Gpu.fromDevice(device, info, retain);
    dit.begin();
    return { codec: codecGpu.gpuBackend(codecGpu.Gpu.fromDevice(device, info, retain)), dit };
  } catch (error) {
    console.log(`no WebGPU (${(error as Error).message}); using the CPU reference`);
    return { codec: cpuBackend, dit: null };
  }
}

/**
 * A reference clip as the model wants it: mono 48 kHz, -16 LUFS, `[frames, 32]`.
 *
 * `patch_sequence_with_mask` is applied by the caller, not here, because the
 * recorded latent has already been through it and both paths have to arrive at
 * the same shape.
 */
async function encodeReference(
  path: string,
  backend: Backend,
): Promise<{ latent: Float32Array; keep: boolean[] }> {
  const file = readFileSync(path);
  if (file.toString("ascii", 0, 4) !== "RIFF" || file.toString("ascii", 8, 12) !== "WAVE") {
    throw new Error(`${path} is not a RIFF/WAVE file`);
  }
  const channels = file.readUInt16LE(22);
  const rate = file.readUInt32LE(24);
  const bits = file.readUInt16LE(34);
  if (rate !== 48000 || bits !== 16) {
    throw new Error(
      `${path} is ${rate} Hz ${bits}-bit; this reads 48 kHz 16-bit PCM only ` +
        `(the reference resamples, and that is not ported)`,
    );
  }
  const frames = (file.length - 44) / 2 / channels;
  const mono = new Float32Array(frames);
  for (let frame = 0; frame < frames; frame += 1) {
    let sum = 0;
    for (let channel = 0; channel < channels; channel += 1) {
      sum += file.readInt16LE(44 + (frame * channels + channel) * 2) / 32768;
    }
    mono[frame] = sum / channels;
  }

  const normalized = normalizeLoudness(mono, rate, -16);
  const encoded = await encode({ data: normalized.data, channels: 1, length: frames }, { backend });
  // `[latentDim, frames]` to `[frames, latentDim]`, which is what the speaker
  // encoder reads.
  const out = new Float32Array(encoded.data.length);
  for (let frame = 0; frame < encoded.length; frame += 1) {
    for (let d = 0; d < encoded.channels; d += 1) {
      out[frame * encoded.channels + d] = encoded.data[d * encoded.length + frame]!;
    }
  }
  return { latent: out, keep: new Array<boolean>(encoded.length).fill(true) };
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const args = argv.filter((value) => !value.startsWith("--"));
  const refAt = argv.indexOf("--ref");
  const refWav = refAt >= 0 ? argv[refAt + 1] : undefined;
  const text = args.find((value) => value !== refWav) ?? "こんにちは、今日はいい天気ですね。";
  const output = join(HERE, "samples", "port-01.wav");

  if (!existsSync(TOKENIZER_JSON)) {
    throw new Error(`ModernBERT-ja's tokenizer.json is missing at\n  ${TOKENIZER_JSON}`);
  }

  const started = Date.now();
  const since = () => `${((Date.now() - started) / 1000).toFixed(0)}s`;
  const step = (what: string) => console.log(`[${since().padStart(5)}] ${what}`);

  console.log(`text  ${JSON.stringify(text)}`);
  console.log(
    refWav
      ? `voice ${refWav}, encoded here\n`
      : `voice samples/reference-voice.wav, as the latent dump_golden.py recorded (pass --ref to encode one)\n`,
  );

  const bert = loadBertWeights(join(GOLDEN, "bert")).weights;
  const weights = loadModelWeights(join(GOLDEN, "model"));
  const { config } = weights;
  step("weights loaded");

  // --- text ---------------------------------------------------------------
  const tokenizer = loadTokenizer(
    JSON.parse(readFileSync(TOKENIZER_JSON, "utf8")) as TokenizerJson,
  );
  const normalized = normalizeText(text).trim();
  // `PretrainedTextTokenizer.encode`: no special tokens, then BOS prepended by
  // hand. Not the `<s> ... </s>` wrapping the tokenizer would default to.
  const body = tokenizer.encodePieces(normalized).slice(0, config.max_text_len - 1);
  const ids = new Int32Array(config.max_text_len).fill(3); // pad_token_id
  ids[0] = tokenizer.bosId;
  ids.set(body, 1);
  const textKeep = Array.from({ length: config.max_text_len }, (_, at) => at <= body.length);
  step(`tokenized: ${body.length + 1} of ${config.max_text_len} positions`);

  // Padding is exact to drop — see `realLength` — so 25 layers run over the
  // real tokens rather than over 256.
  const real = realLength(textKeep);
  const backbone = runModernBert({
    weights: bert,
    inputIds: ids.subarray(0, real),
    keep: textKeep.slice(0, real),
  });
  step(`ModernBERT-ja: ${bert.config.numLayers} layers over ${real} positions`);

  const backboneDim = config.pretrained_hidden;
  const projected = project(backbone.hidden, textKeep.slice(0, real), weights.projectors.text, weights.normEps);
  const textState = conditionNorm(projected, weights.norms.text, real, config.text_dim, weights.normEps);
  const textContext: Context = { state: textState, keep: textKeep.slice(0, real) };
  void backboneDim;

  // --- speaker ------------------------------------------------------------
  const { codec: backend, dit: gpu } = await acquire();
  let refLatent: Float32Array;
  let refKeep: boolean[];
  let refDim: number;
  if (refWav) {
    const encoded = await encodeReference(refWav, backend);
    refLatent = encoded.latent;
    refKeep = encoded.keep;
    refDim = config.latent_dim;
    step(`reference encoded on ${backend.name}: ${refKeep.length} latent frames`);
    // `patch_sequence_with_mask` with speaker_patch_size, which the recorded
    // latent has already been through.
    const patchedRef = patchReference(refLatent, refKeep, refDim, config.speaker_patch_size);
    refLatent = patchedRef.latent;
    refKeep = patchedRef.keep;
  } else {
    refLatent = golden("speaker_encoder.in.latent");
    refKeep = Array.from(golden("speaker_encoder.in.mask"), (value) => value !== 0);
  }
  const patched = patchReference(refLatent, refKeep, refLatent.length / refKeep.length, 1);
  const speakerRun = runSpeakerEncoder({ weights, latent: patched.latent, keep: patched.keep });
  const speakerNormed = conditionNorm(
    speakerRun.state,
    weights.speaker.outNorm,
    refKeep.length,
    config.speaker_dim,
    weights.normEps,
  );
  const withMean = prependMeanToken(speakerNormed, refKeep, config.speaker_dim);
  const speakerContext: Context = { state: withMean.state, keep: withMean.keep };
  step(`speaker encoder: ${refKeep.length} reference frames -> ${withMean.keep.length} tokens`);

  // --- caption ------------------------------------------------------------
  // No caption in this request, so the state is zero and the mask is false —
  // which is what makes the guided batch three rather than four.
  const captionDim = config.caption_dim ?? config.text_dim;
  const captionTokens = 512;
  const captionContext: Context = {
    state: new Float32Array(captionTokens * captionDim),
    keep: new Array<boolean>(captionTokens).fill(false),
  };

  // --- duration -----------------------------------------------------------
  const logFrames = predictDuration({
    weights,
    textState,
    textKeep: textContext.keep,
    speakerVec: withMean.state.subarray(0, config.speaker_dim),
    captionVec: null,
  });
  const frames = Math.max(1, Math.round(framesFrom(logFrames)));
  step(`duration: ${framesFrom(logFrames).toFixed(2)} -> ${frames} latent frames (${(frames / 25).toFixed(2)}s)`);

  // --- flow ---------------------------------------------------------------
  const plain = { text: textContext, speaker: speakerContext, caption: captionContext };
  const dims = {
    text: config.text_dim,
    speaker: config.speaker_dim,
    caption: captionDim,
  };
  const guided = stack(
    [
      plain,
      { ...plain, text: dropped(textContext, config.text_dim) },
      { ...plain, speaker: dropped(speakerContext, config.speaker_dim) },
    ],
    dims,
  );

  // Gaussian noise. This is the one place the port cannot reproduce the
  // reference bit for bit — `torch.randn` with a seeded generator is a
  // different algorithm — so a render is a sample from the same distribution
  // rather than the same sample.
  const noise = new Float32Array(frames * config.latent_dim);
  let seed = 42;
  const uniform = () => {
    // xorshift32, so a run is at least reproducible against itself.
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return ((seed >>> 0) + 1) / 4294967297;
  };
  for (let index = 0; index < noise.length; index += 2) {
    const radius = Math.sqrt(-2 * Math.log(uniform()));
    const angle = 2 * Math.PI * uniform();
    noise[index] = radius * Math.cos(angle);
    if (index + 1 < noise.length) noise[index + 1] = radius * Math.sin(angle);
  }

  const flowStarted = Date.now();
  const report = (at: number, t: number) =>
    process.stdout.write(`\r[${since().padStart(5)}] flow ${at}/${STEPS} t=${t.toFixed(3)}   `);
  const latent = gpu
    ? await sampleGpu({
        gpu,
        weights,
        noise,
        guided,
        plain,
        scales: [3.0, 5.0],
        steps: STEPS,
        onStep: report,
      })
    : sample({
        weights,
        cond: weights.cond,
        noise,
        conditions: { guided, plain, scales: [3.0, 5.0] },
        steps: STEPS,
        onStep: report,
      });
  process.stdout.write("\r".padEnd(44) + "\r");
  step(
    `flow: ${STEPS} steps on ${gpu ? "WebGPU" : "the CPU reference"} in ` +
      `${((Date.now() - flowStarted) / 1000).toFixed(1)}s`,
  );

  // --- codec --------------------------------------------------------------
  // `decode` wants `[latentDim, frames]`; the sampler produced `[frames, latentDim]`.
  const channelMajor = new Float32Array(latent.length);
  for (let frame = 0; frame < frames; frame += 1) {
    for (let d = 0; d < config.latent_dim; d += 1) {
      channelMajor[d * frames + frame] = latent[frame * config.latent_dim + d]!;
    }
  }
  const audio = await decode(
    { channels: config.latent_dim, length: frames, data: channelMajor },
    { backend },
  );
  step(
    `DACVAE decode on ${backend.name}: ${audio.length} samples ` +
      `(${(audio.length / SAMPLE_RATE).toFixed(2)}s at ${SAMPLE_RATE} Hz)`,
  );

  writeFileSync(output, wav(audio.data, SAMPLE_RATE));
  let peak = 0;
  let energy = 0;
  for (const value of audio.data) {
    peak = Math.max(peak, Math.abs(value));
    energy += value * value;
  }
  console.log(
    `\nwrote ${output}\n` +
      `  peak ${peak.toFixed(3)}, rms ${Math.sqrt(energy / audio.data.length).toFixed(3)}, ` +
      `${((Date.now() - started) / 1000).toFixed(0)}s total`,
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
