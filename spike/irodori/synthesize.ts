import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { decode } from "../dacvae/decoder.js";
import { conditionNorm, prependMeanToken, project } from "./conditions.js";
import type { Context } from "./dit.js";
import { framesFrom, predictDuration } from "./duration.js";
import { loadBertWeights } from "./bert-weights.js";
import { loadModelWeights } from "./model-weights.js";
import { runModernBert, realLength } from "./modernbert.js";
import { sample } from "./sampler.js";
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
 * ## Two things this is not
 *
 * **The voice is fixed.** The reference clip reaches the model as a DACVAE
 * latent, and the DACVAE *encoder* is not ported — `spike/dacvae` deliberately
 * dumped only the decode path. So the speaker conditioning here is the latent
 * `dump_golden.py` recorded for `samples/reference-voice.wav`, and this
 * synthesizes arbitrary text in that one voice. Cloning a new voice needs the
 * encoder: 27.3 M parameters and 119 tensors, mirroring the decoder and using
 * the same three operations, plus the LUFS loudness normalisation
 * `encode_waveform` applies first.
 *
 * **It is slow.** web-xpu-ops' CPU reference is the definition of correct and
 * the slowest thing available; there is no GPU backend for the Irodori half
 * yet. Expect roughly half an hour per utterance, most of it the sixteen guided
 * flow steps at batch 3. `spike/dacvae` already has a WebGPU backend and it is
 * used here for the codec.
 *
 * Both are gaps in the port, not in the model, and neither is hidden behind a
 * plausible-sounding result: the header this prints says which parts ran.
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

async function main(): Promise<void> {
  const args = process.argv.slice(2).filter((value) => !value.startsWith("--"));
  const text = args[0] ?? "こんにちは、今日はいい天気ですね。";
  const output = join(HERE, "samples", "port-01.wav");

  if (!existsSync(TOKENIZER_JSON)) {
    throw new Error(`ModernBERT-ja's tokenizer.json is missing at\n  ${TOKENIZER_JSON}`);
  }

  const started = Date.now();
  const since = () => `${((Date.now() - started) / 1000).toFixed(0)}s`;
  const step = (what: string) => console.log(`[${since().padStart(5)}] ${what}`);

  console.log(`text  ${JSON.stringify(text)}`);
  console.log(`voice samples/reference-voice.wav, as a recorded DACVAE latent (the encoder is not ported)\n`);

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
  const refLatent = golden("speaker_encoder.in.latent");
  const refKeep = Array.from(golden("speaker_encoder.in.mask"), (value) => value !== 0);
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

  const latent = sample({
    weights,
    cond: weights.cond,
    noise,
    conditions: { guided, plain, scales: [3.0, 5.0] },
    steps: STEPS,
    onStep: (at, t) => process.stdout.write(`\r[${since().padStart(5)}] flow ${at}/${STEPS} t=${t.toFixed(3)}   `),
  });
  process.stdout.write("\r".padEnd(44) + "\r");
  step(`flow: ${STEPS} steps done`);

  // --- codec --------------------------------------------------------------
  // `decode` wants `[latentDim, frames]`; the sampler produced `[frames, latentDim]`.
  const channelMajor = new Float32Array(latent.length);
  for (let frame = 0; frame < frames; frame += 1) {
    for (let d = 0; d < config.latent_dim; d += 1) {
      channelMajor[d * frames + frame] = latent[frame * config.latent_dim + d]!;
    }
  }
  const audio = await decode({ channels: config.latent_dim, length: frames, data: channelMajor });
  step(`DACVAE decode: ${audio.length} samples (${(audio.length / SAMPLE_RATE).toFixed(2)}s at ${SAMPLE_RATE} Hz)`);

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
