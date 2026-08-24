import { createServer } from "node:http";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);

import { decodeGpu } from "../dacvae/decode-gpu.js";
import { encodeGpu, padForHop } from "../dacvae/encode-gpu.js";
import { encoderConfig } from "../dacvae/encoder.js";
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
import { patchReference } from "./speaker-encoder.js";
import { prepareSpeaker, runSpeakerGpu } from "./speaker-encoder-gpu.js";
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
 * ## Voices are swappable, which is the whole point of the model
 *
 * Irodori is zero-shot: the reference clip *is* the speaker, and a demo with
 * one baked-in voice shows everything about the model except the interesting
 * part. `POST /voice` takes 48 kHz mono float32 PCM, encodes it, runs the
 * speaker encoder, and keeps the result under an id that `/say` can name.
 *
 * **The browser does the decoding and resampling**, through
 * `decodeAudioData` and an `OfflineAudioContext`. That is not modesty about
 * where the work happens: `encode_waveform` resamples with torchaudio and this
 * port has no resampler, so the alternative would be refusing everything that
 * is not already 48 kHz. The page says so.
 *
 * ## What is warm and what is not
 *
 * The model is loaded once and the device stays alive for the process. A voice
 * is encoded when it arrives and cached after that, so swapping back to one you
 * have used is free; a sentence pays only for itself.
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
/**
 * How long a reference clip may be.
 *
 * The model's own config allows 120 s. This is lower because the encoder widens
 * a 48 kHz waveform to 64 channels before it downsamples anything — 30 s is
 * 368 MB for that one activation — and because a demo that takes a minute to
 * accept a voice is not one.
 */
const MAX_CLIP_SECONDS = 30;

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

  /**
   * Device errors, made loud.
   *
   * The checks call `gpu.check()` and this did not, so a validation error here
   * produced a dispatch against whatever the driver handed back and the request
   * returned noise with a 200. "Used in submit while destroyed" is the one that
   * has actually happened; there is no reason to think it is the only one.
   *
   * Both the uncaptured handler and the per-request scope are here because they
   * catch different things: the scope covers what a request records, the
   * handler covers everything else including work already in flight.
   */
  let deviceFault: string | null = null;
  device.addEventListener("uncapturederror", (event) => {
    const message = (event as GPUUncapturedErrorEvent).error.message;
    deviceFault = message;
    console.error(`\n[device error] ${message}\n`);
  });
  device.lost.then((reason) => {
    deviceFault = `device lost: ${reason.message}`;
    console.error(`\n[device lost] ${reason.message}\n`);
  });
  // Both codec directions on the resident engine, sharing the device with the
  // model's. The readback path is still what `check.ts` and `check-encode.ts`
  // compare against; it is not what a request runs.
  const codecResident = new ResidentGpu(device, info, retain);
  codecResident.begin();

  const started = Date.now();
  const bert = loadBertWeights(join(GOLDEN, "bert")).weights;
  const weights = loadModelWeights(join(GOLDEN, "model"));
  const { config } = weights;
  const tokenizer = loadTokenizer(JSON.parse(readFileSync(TOKENIZER_JSON, "utf8")) as TokenizerJson);
  console.log(`weights loaded in ${((Date.now() - started) / 1000).toFixed(1)}s on ${info}`);

  interface Voice {
    id: string;
    label: string;
    seconds: number;
    context: Context;
    /** The first token, the masked-mean summary the duration predictor wants. */
    summary: Float32Array;
    ms: number;
  }

  /**
   * A clip to a speaker condition.
   *
   * Loudness-normalise, encode, patch, eight blocks, `speaker_norm`, prepend
   * the mean token. Every step of that is this port; the browser only supplied
   * 48 kHz mono samples.
   */
  async function makeVoice(id: string, label: string, samples: Float32Array): Promise<Voice> {
    const began = Date.now();
    const normalized = normalizeLoudness(samples, SAMPLE_RATE, -16);
    const padded = padForHop(normalized.data, encoderConfig().hopLength);
    const encoded = encodeGpu(codecResident, codecResident.writeInto("voice.wave", padded), padded.length);
    const channelMajor = await codecResident.read(encoded.tensor);
    const latent = new Float32Array(channelMajor.length);
    for (let frame = 0; frame < encoded.length; frame += 1) {
      for (let d = 0; d < encoded.channels; d += 1) {
        latent[frame * encoded.channels + d] = channelMajor[d * encoded.length + frame]!;
      }
    }
    const patched = patchReference(
      latent,
      new Array<boolean>(encoded.length).fill(true),
      config.latent_dim,
      config.speaker_patch_size,
    );
    const patchedDim = patched.latent.length / patched.keep.length;
    const state = await gpu.read(
      runSpeakerGpu(
        prepareSpeaker(gpu, weights, patched.keep),
        gpu.writeInto("voice.latent", patched.latent),
        patchedDim,
      ),
    );
    await gpu.check(`encoding ${label}`);
    // The encoder's activations are the largest scratch in the process — 243 MB
    // each for a twenty-second clip — and they are wanted only while a voice is
    // arriving. Holding them between voices is gigabytes for nothing.
    const freed = codecResident.releaseScratch();
    const withMean = prependMeanToken(state, patched.keep, config.speaker_dim);
    void freed;
    return {
      id,
      label,
      seconds: samples.length / SAMPLE_RATE,
      context: { state: withMean.state, keep: withMean.keep },
      summary: withMean.state.subarray(0, config.speaker_dim),
      ms: Date.now() - began,
    };
  }

  const voices = new Map<string, Voice>();
  const builtIn = await makeVoice(
    "reference",
    "reference-voice.wav",
    readWav(join(HERE, "samples", "reference-voice.wav")),
  );
  voices.set(builtIn.id, builtIn);
  console.log(
    `built-in voice ready in ${(builtIn.ms / 1000).toFixed(1)}s — ` +
      `${builtIn.seconds.toFixed(1)}s of audio, ${builtIn.context.keep.length} speaker tokens`,
  );

  const mb = (bytes: number) => `${(bytes / 1e6).toFixed(0)} MB`;
  function vram(label: string): void {
    const model = gpu.breakdown();
    const codec = codecResident.breakdown();
    console.log(
      `\n[VRAM ${label}] model ${mb(model.weights)} weights + ${mb(model.scratch)} scratch, ` +
        `codec ${mb(codec.weights)} weights + ${mb(codec.scratch)} scratch`,
    );
    for (const [name, engine] of [["model", model], ["codec", codec]] as const) {
      for (const [slot, count, bytes] of engine.groups.slice(0, 4)) {
        if (bytes < 50e6) continue;
        console.log(`    ${name} ${slot.padEnd(14)} ${String(count).padStart(3)} slots  ${mb(bytes)}`);
      }
    }
  }
  vram("after startup");

  const captionDim = config.caption_dim ?? config.text_dim;
  const captionContext: Context = {
    state: new Float32Array(512 * captionDim),
    keep: new Array<boolean>(512).fill(false),
  };

  /** One render at a time — see the module note. */
  let queue: Promise<unknown> = Promise.resolve();

  async function say(
    text: string,
    voice: Voice,
  ): Promise<{ audio: Buffer; timings: Record<string, number>; frames: number }> {
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
      speakerVec: voice.summary,
      captionVec: null,
    });
    const frames = Math.max(1, Math.round(framesFrom(logFrames)));

    const plain = { text: textContext, speaker: voice.context, caption: captionContext };
    const dims = { text: config.text_dim, speaker: config.speaker_dim, caption: captionDim };
    const guided = stack(
      [
        plain,
        { ...plain, text: dropped(textContext, config.text_dim) },
        { ...plain, speaker: dropped(voice.context, config.speaker_dim) },
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

  /** Read a whole request body, up to a cap. */
  function body(request: import("node:http").IncomingMessage, cap: number): Promise<Buffer> {
    return new Promise((resolve, reject) => {
      const parts: Buffer[] = [];
      let size = 0;
      request.on("data", (chunk: Buffer) => {
        size += chunk.length;
        if (size > cap) {
          reject(new Error(`clip is larger than ${(cap / 1e6).toFixed(0)} MB of samples`));
          request.destroy();
          return;
        }
        parts.push(chunk);
      });
      request.on("end", () => resolve(Buffer.concat(parts)));
      request.on("error", reject);
    });
  }

  const listing = () =>
    JSON.stringify(
      [...voices.values()].map((voice) => ({
        id: voice.id,
        label: voice.label,
        seconds: Number(voice.seconds.toFixed(2)),
        tokens: voice.context.keep.length,
        ms: voice.ms,
      })),
    );

  createServer((request, response) => {
    const url = new URL(request.url ?? "/", `http://${request.headers.host}`);
    const send = (code: number, type: string, payload: string | Buffer, extra: Record<string, string> = {}) => {
      response.writeHead(code, { "content-type": type, ...extra });
      response.end(payload);
    };

    if (url.pathname === "/") {
      send(200, "text/html; charset=utf-8", page);
      return;
    }

    if (url.pathname === "/voices") {
      send(200, "application/json; charset=utf-8", listing());
      return;
    }

    if (url.pathname === "/voice" && request.method === "POST") {
      // Serialised with the renders: the scratch pool is one set of buffers, so
      // encoding a voice while a sentence is in flight would corrupt both.
      queue = queue.then(async () => {
        try {
          const raw = await body(request, MAX_CLIP_SECONDS * SAMPLE_RATE * 4);
          if (raw.length % 4 !== 0) throw new Error("body is not whole float32 samples");
          const samples = new Float32Array(raw.buffer, raw.byteOffset, raw.length / 4);
          if (samples.length < SAMPLE_RATE) throw new Error("a clip shorter than one second is not enough voice");
          const label = (url.searchParams.get("label") ?? "uploaded").slice(0, 60);
          const id = `v${voices.size}`;
          // Copied out of the request buffer: the Float32Array above is a view
          // onto a pooled Buffer that Node is free to reuse.
          const voice = await makeVoice(id, label, new Float32Array(samples));
          voices.set(id, voice);
          console.log(`voice "${label}" (${voice.seconds.toFixed(1)}s) ready in ${voice.ms} ms`);
          send(200, "application/json; charset=utf-8", JSON.stringify({ id, voices: JSON.parse(listing()) }));
        } catch (error) {
          console.error(error);
          send(400, "text/plain; charset=utf-8", String((error as Error).message ?? error));
        }
      });
      return;
    }

    if (url.pathname === "/say") {
      const text = (url.searchParams.get("text") ?? "").slice(0, MAX_TEXT);
      const voice = voices.get(url.searchParams.get("voice") ?? "reference");
      if (!text.trim()) {
        send(400, "text/plain; charset=utf-8", "text is required");
        return;
      }
      if (!voice) {
        send(404, "text/plain; charset=utf-8", "no such voice — upload one or use \"reference\"");
        return;
      }
      queue = queue.then(async () => {
        try {
          if (deviceFault) throw new Error(`the device is in a bad state: ${deviceFault}`);
          const result = await say(text, voice);
          // Whatever this request recorded, checked before its audio is
          // returned. An invalid dispatch is silent otherwise.
          await gpu.check("synthesising");
          await codecResident.check("decoding");
          console.log(
            `"${text.slice(0, 30)}" as ${voice.label} -> ${result.frames} frames in ` +
              `${result.timings.total} ms (flow ${result.timings.flow}, decode ${result.timings.decode})`,
          );
          if (url.searchParams.get("vram")) vram("after this request");
          send(200, "audio/wav", result.audio, {
            "content-length": String(result.audio.length),
            "x-timings": JSON.stringify(result.timings),
            "x-frames": String(result.frames),
          });
        } catch (error) {
          console.error(error);
          send(500, "text/plain; charset=utf-8", String((error as Error).message ?? error));
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
