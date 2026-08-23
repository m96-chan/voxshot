import { createServer } from "node:http";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);

import { gpuBackend, Gpu as CodecGpu } from "../dacvae/gpu.js";
import { decodeGpu } from "../dacvae/decode-gpu.js";
import { encode } from "../dacvae/encoder.js";
import { ResidentGpu } from "../dacvae/gpu-resident.js";
import { normalizeLoudness } from "../dacvae/loudness.js";
import { loadBertWeights } from "./bert-weights.js";
import { conditionNorm, prependMeanToken, project } from "./conditions.js";
import type { Context } from "./dit.js";
import { framesFrom, predictDuration } from "./duration.js";
import { Gpu } from "./gpu.js";
import { loadModelWeights } from "./model-weights.js";
import { realLength } from "./modernbert.js";
import { prepareBert, runBertGpu } from "./modernbert-gpu.js";
import { sampleGpu } from "./sampler-gpu.js";
import { patchReference, runSpeakerEncoder } from "./speaker-encoder.js";
import { normalizeText } from "./text.js";
import { loadTokenizer, type TokenizerJson } from "./tokenizer.js";

/**
 * Type a sentence, hear it.
 *
 *     cd spike/irodori && npm run serve
 *     open http://127.0.0.1:8123
 *
 * The point of this file is that it is **not** a page of pre-rendered results.
 * Everything below the text box is computed when the request arrives, by this
 * port, on this machine's GPU.
 *
 * ## What is warm and what is not
 *
 * The model is loaded once and the device stays alive for the process, so a
 * request pays for its own text and nothing else. The reference clip's latent
 * and the speaker state are computed once at startup — they depend on the
 * voice, not the sentence — which is what makes a request about ten seconds
 * rather than twenty.
 *
 * ## Not concurrent
 *
 * One request at a time, serialised behind a promise. The scratch pool is a
 * single set of buffers named by what they hold; two renders at once would
 * write into each other and produce two wrong answers rather than an error.
 * A queue is the honest fix at this size, and this says so instead of looking
 * like it scales.
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
const PORT = Number(process.env.PORT ?? 8123);
const STEPS = 32;
const SAMPLE_RATE = 48000;
const MAX_TEXT = 200;

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

/** 48 kHz 16-bit PCM only; the reference resamples and that is not ported. */
function readWav(path: string): Float32Array {
  const file = readFileSync(path);
  const channels = file.readUInt16LE(22);
  const rate = file.readUInt32LE(24);
  const bits = file.readUInt16LE(34);
  if (rate !== SAMPLE_RATE || bits !== 16) {
    throw new Error(`${path} is ${rate} Hz ${bits}-bit; this reads 48 kHz 16-bit PCM only`);
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
  return mono;
}

function dropped(context: Context, dim: number): Context {
  return { state: new Float32Array(context.keep.length * dim), keep: context.keep.map(() => false) };
}

function stack(members: Record<string, Context>[], dims: Record<string, number>): Record<string, Context> {
  const out: Record<string, Context> = {};
  for (const name of Object.keys(members[0]!)) {
    const width = dims[name]!;
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
  if (!existsSync(TOKENIZER_JSON)) throw new Error(`tokenizer.json is missing at\n  ${TOKENIZER_JSON}`);

  const instance = create([]);
  const adapter = await instance.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter — this server needs one to be usable");
  const info = adapter.info?.description ?? "unknown adapter";
  const device = await Gpu.requestDevice(adapter);
  const retain = [instance, adapter];
  const gpu = Gpu.fromDevice(device, info, retain);
  gpu.begin();
  // Two engines on one device. The encoder still uses the readback path — it
  // runs once at startup, where four seconds is not worth a second port — and
  // the decoder uses the resident one, where it was most of every request.
  const codec = gpuBackend(CodecGpu.fromDevice(device, info, retain));
  const codecResident = new ResidentGpu(device, info, retain);

  const started = Date.now();
  const bert = loadBertWeights(join(GOLDEN, "bert")).weights;
  const weights = loadModelWeights(join(GOLDEN, "model"));
  const { config } = weights;
  const tokenizer = loadTokenizer(JSON.parse(readFileSync(TOKENIZER_JSON, "utf8")) as TokenizerJson);
  console.log(`weights loaded in ${((Date.now() - started) / 1000).toFixed(1)}s on ${info}`);

  // The voice, once. It depends on the clip and not on the sentence, so paying
  // for it per request would double the wait for nothing.
  const voicePath = join(HERE, "samples", "reference-voice.wav");
  const voiceStarted = Date.now();
  const normalized = normalizeLoudness(readWav(voicePath), SAMPLE_RATE, -16);
  const encoded = await encode(
    { data: normalized.data, channels: 1, length: normalized.data.length },
    { backend: codec },
  );
  const refLatent = new Float32Array(encoded.data.length);
  for (let frame = 0; frame < encoded.length; frame += 1) {
    for (let d = 0; d < encoded.channels; d += 1) {
      refLatent[frame * encoded.channels + d] = encoded.data[d * encoded.length + frame]!;
    }
  }
  const patchedRef = patchReference(
    refLatent,
    new Array<boolean>(encoded.length).fill(true),
    config.latent_dim,
    config.speaker_patch_size,
  );
  const speakerRun = runSpeakerEncoder({ weights, latent: patchedRef.latent, keep: patchedRef.keep });
  const speakerNormed = conditionNorm(
    speakerRun.state,
    weights.speaker.outNorm,
    patchedRef.keep.length,
    config.speaker_dim,
    weights.normEps,
  );
  const withMean = prependMeanToken(speakerNormed, patchedRef.keep, config.speaker_dim);
  const speakerContext: Context = { state: withMean.state, keep: withMean.keep };
  console.log(
    `voice ready in ${((Date.now() - voiceStarted) / 1000).toFixed(1)}s — ` +
      `${encoded.length} latent frames, ${withMean.keep.length} speaker tokens`,
  );

  const captionDim = config.caption_dim ?? config.text_dim;
  const captionContext: Context = {
    state: new Float32Array(512 * captionDim),
    keep: new Array<boolean>(512).fill(false),
  };

  /** One render at a time — see the module note. */
  let queue: Promise<unknown> = Promise.resolve();

  async function say(text: string): Promise<{ audio: Buffer; timings: Record<string, number>; frames: number }> {
    const t0 = Date.now();
    const normalizedText = normalizeText(text).trim();
    const body = tokenizer.encodePieces(normalizedText).slice(0, config.max_text_len - 1);
    const ids = new Int32Array(config.max_text_len).fill(3);
    ids[0] = tokenizer.bosId;
    ids.set(body, 1);
    const textKeep = Array.from({ length: config.max_text_len }, (_, at) => at <= body.length);
    const real = realLength(textKeep);

    const t1 = Date.now();
    const backbone = await gpu.read(
      runBertGpu(prepareBert(gpu, bert, ids.subarray(0, real), textKeep.slice(0, real))),
    );
    const projected = project(backbone, textKeep.slice(0, real), weights.projectors.text, weights.normEps);
    const textState = conditionNorm(projected, weights.norms.text, real, config.text_dim, weights.normEps);
    const textContext: Context = { state: textState, keep: textKeep.slice(0, real) };

    const t2 = Date.now();
    const logFrames = predictDuration({
      weights,
      textState,
      textKeep: textContext.keep,
      speakerVec: withMean.state.subarray(0, config.speaker_dim),
      captionVec: null,
    });
    const frames = Math.max(1, Math.round(framesFrom(logFrames)));

    const plain = { text: textContext, speaker: speakerContext, caption: captionContext };
    const dims = { text: config.text_dim, speaker: config.speaker_dim, caption: captionDim };
    const guided = stack(
      [
        plain,
        { ...plain, text: dropped(textContext, config.text_dim) },
        { ...plain, speaker: dropped(speakerContext, config.speaker_dim) },
      ],
      dims,
    );

    // Gaussian noise. `torch.randn` with a seeded generator is not reproducible
    // in JavaScript, so a render is a draw from the same distribution rather
    // than the same draw — and the seed moves per request so two goes at the
    // same sentence are not the same audio.
    const noise = new Float32Array(frames * config.latent_dim);
    let seed = (Date.now() & 0x7fffffff) || 1;
    const uniform = () => {
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

    const t3 = Date.now();
    const latent = await sampleGpu({
      gpu,
      weights,
      noise,
      guided,
      plain,
      scales: [3.0, 5.0],
      steps: STEPS,
    });

    const t4 = Date.now();
    const channelMajor = new Float32Array(latent.length);
    for (let frame = 0; frame < frames; frame += 1) {
      for (let d = 0; d < config.latent_dim; d += 1) {
        channelMajor[d * frames + frame] = latent[frame * config.latent_dim + d]!;
      }
    }
    const decoded = decodeGpu(codecResident, codecResident.writeInto("codec.latent", channelMajor), frames);
    const audio = { data: await codecResident.read(decoded.tensor), length: decoded.length };
    const t5 = Date.now();

    return {
      audio: wav(audio.data, SAMPLE_RATE),
      frames,
      timings: {
        tokenize: t1 - t0,
        text_encoder: t2 - t1,
        duration: t3 - t2,
        flow: t4 - t3,
        decode: t5 - t4,
        total: t5 - t0,
      },
    };
  }

  const page = readFileSync(join(HERE, "demo.html"));
  createServer((request, response) => {
    const url = new URL(request.url ?? "/", `http://${request.headers.host}`);
    if (url.pathname === "/") {
      response.writeHead(200, { "content-type": "text/html; charset=utf-8" });
      response.end(page);
      return;
    }
    if (url.pathname === "/say") {
      const text = (url.searchParams.get("text") ?? "").slice(0, MAX_TEXT);
      if (!text.trim()) {
        response.writeHead(400, { "content-type": "text/plain; charset=utf-8" });
        response.end("text is required");
        return;
      }
      queue = queue.then(async () => {
        try {
          const result = await say(text);
          console.log(
            `"${text.slice(0, 30)}" -> ${result.frames} frames in ${result.timings.total} ms ` +
              `(flow ${result.timings.flow}, decode ${result.timings.decode})`,
          );
          response.writeHead(200, {
            "content-type": "audio/wav",
            "content-length": String(result.audio.length),
            "x-timings": JSON.stringify(result.timings),
            "x-frames": String(result.frames),
          });
          response.end(result.audio);
        } catch (error) {
          console.error(error);
          response.writeHead(500, { "content-type": "text/plain; charset=utf-8" });
          response.end(String((error as Error).message ?? error));
        }
      });
      return;
    }
    response.writeHead(404).end();
  }).listen(PORT, "127.0.0.1", () => {
    console.log(`\nlistening on http://127.0.0.1:${PORT}`);
  });
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
