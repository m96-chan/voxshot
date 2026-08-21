/**
 * Drive `voxshot/miotts` end to end against the real weights.
 *
 *     cd spike/miotts && npx tsx check-engine.ts [--text "…"] [--reference clip.wav]
 *
 * Everything else that checks this pipeline goes around the public API. The
 * CI suite drives synthetic models, `npm run test:models` checks each stage
 * against the reference implementation's own output, and `check-tts.mjs`
 * drives the demo page — which is `browser.ts`, not the engine. None of them
 * would notice if `createMioTtsEngine` were wired to the wrong thing.
 *
 * This is the one that would: `VoxShot.create({ engine })` over a
 * `MioTtsWeightSource` reading the local artifacts, cloning a voice and
 * speaking. It is a check script rather than a test because it needs ~1.1 GB
 * of weights and a GPU, and because Dawn's Node binding kills a Vitest worker
 * before a single test runs (measured; see this spike's vitest.config.ts).
 *
 * `--reference` also makes this the way a caller extracts a speaker embedding
 * through the public API: it prints the 128 floats and writes them beside the
 * audio, so a fixed-voice application can bake them in and never ship the
 * encoder's 117 MB.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { create, globals } from "webgpu";

import { VoxShot } from "../../src/index.js";
import type { Platform, PcmAudio } from "../../src/platform.js";
import { createMioTtsEngine } from "../../src/engine/miotts/index.js";
import type { MioTtsWeightPart, MioTtsWeightSource } from "../../src/engine/miotts/weights.js";
import { encodeWav } from "../../src/audio/wav.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const HUB = join(homedir(), ".cache", "huggingface", "hub");

/** Resolve a file inside a Hugging Face snapshot by following `refs/main`. */
function hubFile(repo: string, name: string): string {
  const revision = readFileSync(join(HUB, repo, "refs", "main"), "utf8").trim();
  return join(HUB, repo, "snapshots", revision, name);
}

/**
 * The local artifacts, as a weight source.
 *
 * This is the whole point of the contract: the same engine that reads from
 * Hugging Face and the Cache API in a browser reads from disk here, and
 * nothing in the engine changes.
 */
const PATHS: Record<MioTtsWeightPart, () => string> = {
  tokenizer: () => hubFile("models--Aratako--MioTTS-0.6B", "tokenizer.json"),
  "lm-manifest": () => join(HERE, "q8", "manifest.json"),
  "lm-codes": () => join(HERE, "q8", "weights.codes.bin"),
  "lm-scales": () => join(HERE, "q8", "weights.scales.bin"),
  "lm-norms": () => join(HERE, "q8", "weights.norms.bin"),
  "codec-decoder": () => hubFile("models--Aratako--MioCodec-25Hz-24kHz", "model.safetensors"),
  "codec-encoder": () => join(HERE, "..", "miocodec", "golden-encoder", "encoder-weights.safetensors"),
};

const fromDisk: MioTtsWeightSource = {
  async load(part) {
    const path = PATHS[part]();
    process.stdout.write(`  ${part.padEnd(14)} ${path}\n`);
    // A Buffer is a view over a pooled ArrayBuffer; `loadPart` copies it out
    // at its own window, which is exactly what that copy is for.
    return readFileSync(path);
  },
};

/**
 * A 16-bit PCM WAV, as mono `PcmAudio`.
 *
 * Node has no `decodeAudioData`, and `VoxShot.cloneVoice` takes ready-made PCM
 * as well as an encoded file — so the platform's decoder is never reached and
 * this script does not need one. Deliberately minimal: it is a check harness,
 * not a decoder, and anything it cannot read it refuses rather than guesses at.
 */
function readWav(path: string): PcmAudio {
  const bytes = readFileSync(path);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.toString("ascii", 0, 4) !== "RIFF" || bytes.toString("ascii", 8, 12) !== "WAVE") {
    throw new Error(`${path} is not a RIFF/WAVE file`);
  }
  let at = 12;
  let channels = 0;
  let sampleRate = 0;
  let bits = 0;
  while (at + 8 <= bytes.byteLength) {
    const id = bytes.toString("ascii", at, at + 4);
    const size = view.getUint32(at + 4, true);
    const body = at + 8;
    if (id === "fmt ") {
      const format = view.getUint16(body, true);
      if (format !== 1) throw new Error(`${path} is WAVE format ${format}; only PCM is read here`);
      channels = view.getUint16(body + 2, true);
      sampleRate = view.getUint32(body + 4, true);
      bits = view.getUint16(body + 14, true);
    } else if (id === "data") {
      if (bits !== 16) throw new Error(`${path} is ${bits}-bit; only 16-bit PCM is read here`);
      const frames = size / 2 / channels;
      const samples = new Float32Array(frames);
      for (let frame = 0; frame < frames; frame += 1) {
        let sum = 0;
        for (let channel = 0; channel < channels; channel += 1) {
          sum += view.getInt16(body + (frame * channels + channel) * 2, true) / 32_768;
        }
        samples[frame] = sum / channels;
      }
      return { samples, sampleRate };
    }
    at = body + size + (size % 2);
  }
  throw new Error(`${path} has no data chunk`);
}

/**
 * Only the seam VoxShot actually reaches here.
 *
 * The reference arrives as PCM and the audio is written to a file, so nothing
 * asks the decoder or the player to do anything — but the interface wants
 * them, and a stub that throws is more honest than one that pretends.
 */
const nodePlatform: Platform = {
  decoder: {
    decode: () => {
      throw new Error("check-engine reads WAV itself; nothing should reach the decoder");
    },
  },
  player: {
    play: () => {
      throw new Error("check-engine writes a file; nothing should reach the player");
    },
  },
  gpu: { isAvailable: async () => true },
};

function arg(name: string, fallback: string): string {
  const at = process.argv.indexOf(`--${name}`);
  return at >= 0 && process.argv[at + 1] !== undefined ? process.argv[at + 1]! : fallback;
}

async function main(): Promise<void> {
  Object.assign(globalThis, globals);
  const gpu = create(["enable-dawn-features=allow_unsafe_apis"]);
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter — check the driver, not this script");
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });

  const text = arg("text", "こんにちは、今日はいい天気ですね");
  const reference = arg("reference", "");

  process.stdout.write("loading:\n");
  const started = Date.now();
  const engine = await createMioTtsEngine({ weights: fromDisk, device });
  const tts = await VoxShot.create({ engine, device: "webgpu", platform: nodePlatform });
  process.stdout.write(`load: ${((Date.now() - started) / 1000).toFixed(1)} s\n`);
  process.stdout.write(`device: ${tts.device} (${engine.adapterInfo})\n`);

  if (!reference) {
    throw new Error(
      "--reference is required: the engine has no built-in voice, so a caller supplies one " +
        "(clone a clip here, or hand VoxShot a saved 128-float embedding).",
    );
  }

  const encodeStarted = Date.now();
  const voice = await tts.cloneVoice(readWav(reference));
  const encodeMs = Date.now() - encodeStarted;
  process.stdout.write(
    `voice: ${voice.vector.length} dims in ${(encodeMs / 1000).toFixed(1)} s ` +
      `from ${engine.lastReferenceSeconds.toFixed(2)} s of audio` +
      `${engine.referenceWasTrimmed ? " (trimmed)" : ""}\n`,
  );

  const speakStarted = Date.now();
  const audio = await tts.speak(text);
  const speakMs = Date.now() - speakStarted;
  const seconds = audio.samples.length / audio.sampleRate;
  process.stdout.write(
    `speak: ${seconds.toFixed(2)} s of audio in ${(speakMs / 1000).toFixed(2)} s ` +
      `(RTF ${(speakMs / 1000 / seconds).toFixed(2)})\n`,
  );

  const stem = arg("out", join(HERE, "check-engine-out"));
  writeFileSync(`${stem}.wav`, Buffer.from(encodeWav(audio.samples, audio.sampleRate)));
  writeFileSync(`${stem}.embedding.json`, JSON.stringify(Array.from(voice.vector)));
  writeFileSync(`${stem}.embedding.bin`, Buffer.from(voice.vector.buffer.slice(0)));
  process.stdout.write(`wrote ${stem}.wav, ${stem}.embedding.{json,bin}\n`);

  await tts.dispose();
}

await main();
