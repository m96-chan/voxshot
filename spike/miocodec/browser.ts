import { cpuBackend, decode, MIOCODEC_24K, Weights, type Backend } from "./decoder.js";
import { Gpu, gpuBackend } from "./gpu.js";
import { Safetensors } from "./safetensors.js";

/**
 * The browser entry point for `examples/mio-tts.html`.
 *
 * Fetches MioCodec's checkpoint straight from the Hugging Face CDN, decodes the
 * committed token fixture, and hands back PCM. No server, no build step beyond
 * the bundle, and nothing cached on anyone's behalf.
 *
 * What this is **not**: text to speech. MioTTS's language model turns text into
 * these tokens and has no ONNX export yet, so the tokens here come from a real
 * speech clip encoded offline. This page is the second half of that pipeline —
 * the codec — running end to end in the browser.
 */

const CHECKPOINT = "https://huggingface.co/Aratako/MioCodec-25Hz-24kHz/resolve/main/model.safetensors";

export interface Fixture {
  repo_id: string;
  source: string;
  sample_rate: number;
  stft_length: number;
  tokens: number[];
  global_embedding: number[];
}

export interface Progress {
  (stage: string, detail?: { loaded?: number; total?: number }): void;
}

/**
 * The checkpoint, streamed so the page can show progress.
 *
 * Half a gigabyte arrives over a minute or two on an ordinary connection, and a
 * page that says nothing for that long looks broken. `Content-Length` is
 * present on this CDN but not promised anywhere, so the total is optional and
 * the caller has to cope with not having one.
 */
export async function fetchCheckpoint(report: Progress): Promise<ArrayBuffer> {
  report("downloading the checkpoint");
  const response = await fetch(CHECKPOINT);
  if (!response.ok) throw new Error(`checkpoint: HTTP ${response.status}`);

  const header = response.headers.get("content-length");
  const total = header ? Number(header) : undefined;
  const reader = response.body?.getReader();
  if (!reader) return await response.arrayBuffer();

  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    report("downloading the checkpoint", { loaded, total });
  }

  const buffer = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return buffer.buffer;
}

export interface DecodeReport {
  pcm: Float32Array;
  sampleRate: number;
  seconds: number;
  elapsedMs: number;
  realTimeFactor: number;
  /** Which backend actually ran. Never inferred by the caller — see below. */
  backend: string;
}

export type BackendChoice = "auto" | "gpu" | "cpu";

/**
 * Pick a backend, and **say which one was picked**.
 *
 * `"auto"` prefers WebGPU and falls back to the reference implementations,
 * because a page that works slowly beats a page that does not work. What it
 * must not do is fall back silently: an RTF is meaningless without knowing what
 * produced it, so the choice is returned rather than logged.
 *
 * `"gpu"` throws instead of falling back — the check script asks for it
 * explicitly, and a silent downgrade there would report the CPU path's numbers
 * as the GPU path's.
 */
async function chooseBackend(choice: BackendChoice): Promise<{ backend: Backend; gpu: Gpu | null }> {
  if (choice === "cpu") return { backend: cpuBackend, gpu: null };
  const gpu = await Gpu.create();
  if (gpu) return { backend: gpuBackend(gpu), gpu };
  if (choice === "gpu") throw new Error("WebGPU is unavailable, and the GPU backend was required");
  return { backend: cpuBackend, gpu: null };
}

export async function run(
  fixture: Fixture,
  report: Progress,
  choice: BackendChoice = "auto",
): Promise<DecodeReport> {
  const buffer = await fetchCheckpoint(report);

  report("parsing the checkpoint");
  const weights = new Weights(Safetensors.parse(buffer));

  const { backend, gpu } = await chooseBackend(choice);
  report(`decoding on ${backend.name}`);
  // Yield first: the stages between dispatches run on the main thread and hold
  // it, so without this the "decoding" line never paints and the page looks
  // stuck on "parsing".
  await new Promise((resolve) => setTimeout(resolve, 0));

  try {
    const started = performance.now();
    const { waveform } = await decode(
      Float32Array.from(fixture.tokens),
      Float32Array.from(fixture.global_embedding),
      fixture.stft_length,
      MIOCODEC_24K,
      weights,
      backend,
    );
    const elapsedMs = performance.now() - started;
    const seconds = waveform.length / fixture.sample_rate;

    return {
      pcm: waveform,
      sampleRate: fixture.sample_rate,
      seconds,
      elapsedMs,
      realTimeFactor: elapsedMs / 1000 / seconds,
      backend: backend.name,
    };
  } finally {
    gpu?.destroy();
  }
}

/** PCM to a WAV blob, so the result can be played and saved without a library. */
export function toWav(pcm: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + pcm.length * 2);
  const view = new DataView(buffer);
  const ascii = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  ascii(0, "RIFF");
  view.setUint32(4, 36 + pcm.length * 2, true);
  ascii(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, "data");
  view.setUint32(40, pcm.length * 2, true);
  for (let i = 0; i < pcm.length; i += 1) {
    // Clamped before scaling: the decoder has no output limiter, and a sample
    // past ±1 would wrap to the opposite rail as an audible click rather than
    // clipping.
    const sample = Math.max(-1, Math.min(1, pcm[i]!));
    view.setInt16(44 + i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}
