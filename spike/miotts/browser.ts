import { sampleNext, type SamplerOptions } from "../../../web-xpu-ops/llm/sampler.js";
import { fetchCheckpoint, toWav, type Fixture } from "../miocodec/browser.js";
import { cpuBackend, decode, MIOCODEC_24K, Weights, type Backend } from "../miocodec/decoder.js";
import { Gpu, gpuBackend } from "../miocodec/gpu.js";
import { Safetensors } from "../miocodec/safetensors.js";
import { createGpuEngine, type GpuEngine, type GpuEngineStats } from "./gpu-engine.js";
import { normalizeText } from "./text.js";
import { loadTokenizer, speechIndexOf, type Tokenizer, type TokenizerJson } from "./tokenizer.js";
import { loadWeightsQ8, type Qwen3WeightsQ8, type WeightsQ8Manifest } from "./weights-q8.js";

/**
 * Browser entry for `examples/mio-tts.html` — the full pipeline, text to
 * speech, in one page: the Qwen3-0.6B LM (q8, WebGPU-resident, gpu-engine.ts)
 * turns text into `<|s_n|>` speech tokens, and the MioCodec decoder (imported
 * across spikes from ../miocodec) turns those into 24 kHz PCM.
 *
 * Assets, all same-origin relative URLs (serve.mjs maps them):
 *   ./miotts/tokenizer.json        the HF tokenizer snapshot
 *   ./miotts/q8/manifest.json      convert_weights.py's manifest
 *   ./miotts/q8/weights.*.bin      codes / scales / norms (~580 MiB, streamed)
 *   ./mio-codec-fixture.json       carries the demo global_embedding (speaker)
 *   MioCodec checkpoint            the same HF URL mio-codec.html uses, so the
 *                                  existing route/302-to-local-cache trick works
 *
 * sha256 of the q8 bins is **skipped** in the browser (`loadWeightsQ8`'s
 * hasher is optional and its `Sha256Fn` is synchronous; SubtleCrypto is async
 * and a ~600 MiB digest costs seconds on the load path). Byte-length checks
 * against the manifest still run. Node loads (weights-q8-node.ts) keep the
 * full hash check.
 *
 * ## E2E hook (deliberate, for check scripts)
 *
 * A driver script (playwright etc.) reads `window.__result` after clicking
 * `#run` and waiting for `#metrics:not(.hidden)`:
 *
 *   window.__result = {
 *     status: "done",
 *     adapter: string, mode: "greedy" | "sample", seed: number,
 *     promptIds: number[], generatedIds: number[],   // generatedIds includes the eos when one was produced
 *     speechIndices: number[], nonSpeechIds: number[], // nonSpeechIds excludes eos; should be []
 *     lmMs, lmSteps, lmTokensPerSec, decodeMs, totalMs, audioSeconds, rtf: number,
 *       // rtf follows the codec spike's convention: processing seconds per
 *       // second of audio, so smaller is better and < 1 is faster than realtime.
 *   }
 *
 * On failure `window.__result = { status: "error", error: string }` and
 * `#status` shows the message. `__result` is reset to null when a run starts.
 */

const EOS_IDS = [151645, 151643];
const MAX_NEW_TOKENS = 700; // the MioTTS reference server's own cap
const MAX_SEQ_LEN = 768; // prompt (~15) + 700 generated, with slack
const SAMPLE_RATE_EXPECTED = 24000;

interface RunResult {
  status: "done";
  adapter: string;
  mode: "greedy" | "sample";
  seed: number;
  promptIds: number[];
  generatedIds: number[];
  speechIndices: number[];
  nonSpeechIds: number[];
  lmMs: number;
  lmSteps: number;
  lmTokensPerSec: number;
  decodeMs: number;
  totalMs: number;
  audioSeconds: number;
  rtf: number;
  /** gpu-engine.ts' allocation/submit counters — lets a driver assert the loop shape. */
  engineStats: GpuEngineStats;
}

declare global {
  interface Window {
    __result: RunResult | { status: "error"; error: string } | null;
  }
}

type Progress = (stage: string, detail?: { loaded?: number; total?: number }) => void;

/** Streamed fetch with progress — the same shape as ../miocodec's fetchCheckpoint, for arbitrary URLs. */
async function fetchWithProgress(url: string, label: string, report: Progress): Promise<ArrayBuffer> {
  report(label);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
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
    report(label, { loaded, total });
  }
  const buffer = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return buffer.buffer;
}

/** Deterministic xorshift32 in [0, 1) so sampled runs are reproducible from a seed. */
function xorshift32(seed: number): () => number {
  let s = seed >>> 0 || 0x9e3779b9;
  return () => {
    s ^= s << 13;
    s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 4294967296;
  };
}

interface Loaded {
  adapter: string;
  engine: GpuEngine;
  tokenizer: Tokenizer;
  normalize: (text: string) => string;
  codecWeights: Weights;
  codecBackend: Backend;
  fixture: Fixture;
}

/**
 * The WebGPU device, requested at the adapter's own limits — the packed
 * embedTokens table alone is ~168 MiB of storage and its largest lm_head
 * chunk binding is ~67 MiB, past some defaults. Mirrors
 * ../miocodec/gpu.ts#Gpu.create; unlike that page there is no CPU fallback
 * for the LM (the run path deliberately has no CPU oracle in it), so absence
 * of WebGPU is an error, not a downgrade.
 */
async function requestDevice(): Promise<{ device: GPUDevice; adapter: string }> {
  const gpu = (globalThis.navigator as Navigator | undefined)?.gpu;
  if (!gpu) throw new Error("WebGPU is unavailable — this page needs navigator.gpu for the language model");
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error("requestAdapter() returned null — no WebGPU adapter");
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  const info = adapter.info
    ? [adapter.info.vendor, adapter.info.architecture, adapter.info.description].filter(Boolean).join(" ") || "unknown adapter"
    : "unknown adapter";
  return { device, adapter: info };
}

async function loadEverything(report: Progress): Promise<Loaded> {
  const { device, adapter } = await requestDevice();

  report("loading tokenizer.json");
  const tokenizerJson = (await (await fetch("./miotts/tokenizer.json")).json()) as TokenizerJson;
  const tokenizer = await loadTokenizer(tokenizerJson);

  report("loading the q8 manifest");
  const manifestResponse = await fetch("./miotts/q8/manifest.json");
  if (!manifestResponse.ok) throw new Error(`q8 manifest: HTTP ${manifestResponse.status}`);
  const manifest = (await manifestResponse.json()) as WeightsQ8Manifest;

  const codes = await fetchWithProgress(`./miotts/q8/${manifest.files.codes.name}`, "LM weights (codes)", report);
  const scales = await fetchWithProgress(`./miotts/q8/${manifest.files.scales.name}`, "LM weights (scales)", report);
  const norms = await fetchWithProgress(`./miotts/q8/${manifest.files.norms.name}`, "LM weights (norms)", report);

  report("unpacking the q8 weights");
  await new Promise((resolve) => setTimeout(resolve, 0)); // let the status paint before ~1s of packing
  const weights: Qwen3WeightsQ8 = loadWeightsQ8({ manifest, codes, scales, norms }); // sha256 skipped — module doc

  report("uploading the LM to the GPU");
  const engine = await createGpuEngine(device, weights, { maxSeqLen: MAX_SEQ_LEN });

  // MioCodec: the same checkpoint URL and streaming loader as mio-codec.html.
  const checkpoint = await fetchCheckpoint(report);
  report("parsing the codec checkpoint");
  const codecWeights = new Weights(Safetensors.parse(checkpoint));

  report("loading the speaker fixture");
  const fixtureResponse = await fetch("./mio-codec-fixture.json");
  if (!fixtureResponse.ok) throw new Error(`mio-codec-fixture.json: HTTP ${fixtureResponse.status}`);
  const fixture = (await fixtureResponse.json()) as Fixture;
  if (fixture.sample_rate !== SAMPLE_RATE_EXPECTED) {
    throw new Error(`fixture sample_rate ${fixture.sample_rate}, expected ${SAMPLE_RATE_EXPECTED}`);
  }

  // The codec reuses the LM's device (Gpu.fromDevice) — one device, weights
  // resident across runs, and no second requestDevice. cpuBackend remains the
  // fallback shape but is unreachable here: without WebGPU we threw above.
  const codecBackend: Backend = gpuBackend(Gpu.fromDevice(device, adapter)) ?? cpuBackend;

  // The reference server's normalize_text, ported in text.ts and now bundled
  // statically (it was a runtime probe while text.ts belonged to a parallel
  // agent that had not landed it yet).
  return { adapter, engine, tokenizer, normalize: normalizeText, codecWeights, codecBackend, fixture };
}

interface GenerateOutcome {
  result: RunResult;
  pcm: Float32Array;
  sampleRate: number;
}

async function generate(
  loaded: Loaded,
  text: string,
  mode: "greedy" | "sample",
  seed: number,
  report: Progress,
): Promise<GenerateOutcome> {
  const { engine, tokenizer, normalize, codecWeights, codecBackend, fixture } = loaded;

  const promptIds = tokenizer.encodeChat(normalize(text));
  if (promptIds.length + 1 >= MAX_SEQ_LEN) {
    throw new Error(`prompt is ${promptIds.length} tokens; the engine was built for maxSeqLen=${MAX_SEQ_LEN}`);
  }
  const maxNew = Math.min(MAX_NEW_TOKENS, MAX_SEQ_LEN - promptIds.length);
  const sampler: SamplerOptions =
    mode === "greedy"
      ? { mode: "greedy" }
      // The MioTTS reference server's own sampling: temperature 0.8, top_p 1.0.
      : { mode: "top-p", temperature: 0.8, topP: 1.0, rng: xorshift32(seed) };

  engine.reset();
  report("generating speech tokens");
  const lmStart = performance.now();
  // Prefill = the decode path, one token at a time (see gpu-engine.ts' module
  // doc — 15 tokens do not earn a batched prefill). Only the last logits matter.
  let lmSteps = 0;
  let logits: Float32Array | null = null;
  for (const id of promptIds) {
    logits = await engine.decodeStep(id);
    lmSteps += 1;
  }

  const generatedIds: number[] = [];
  for (let step = 0; step < maxNew; step += 1) {
    const id = sampleNext(logits!, generatedIds, sampler);
    generatedIds.push(id);
    if (EOS_IDS.includes(id)) break;
    if (step + 1 < maxNew) {
      logits = await engine.decodeStep(id);
      lmSteps += 1;
    }
    if (step % 10 === 0) report(`generating speech tokens — ${generatedIds.length}`);
  }
  const lmMs = performance.now() - lmStart;

  const speechIndices: number[] = [];
  const nonSpeechIds: number[] = [];
  for (const id of generatedIds) {
    if (EOS_IDS.includes(id)) continue;
    const index = speechIndexOf(id);
    if (index === null) nonSpeechIds.push(id);
    else speechIndices.push(index);
  }
  if (speechIndices.length === 0) throw new Error("the LM produced no speech tokens");

  report(`decoding ${speechIndices.length} speech tokens on ${codecBackend.name}`);
  await new Promise((resolve) => setTimeout(resolve, 0)); // let the status paint
  const decodeStart = performance.now();
  const { waveform } = await decode(
    Float32Array.from(speechIndices),
    Float32Array.from(fixture.global_embedding),
    // The "aligned" path: 2 STFT frames per 25 Hz token -> n*960 samples at 24 kHz.
    2 * speechIndices.length,
    MIOCODEC_24K,
    codecWeights,
    codecBackend,
  );
  const decodeMs = performance.now() - decodeStart;

  const audioSeconds = waveform.length / fixture.sample_rate;
  const totalMs = lmMs + decodeMs;
  return {
    result: {
      status: "done",
      adapter: loaded.adapter,
      mode,
      seed,
      promptIds,
      generatedIds,
      speechIndices,
      nonSpeechIds,
      lmMs,
      lmSteps,
      lmTokensPerSec: lmSteps / (lmMs / 1000),
      decodeMs,
      totalMs,
      audioSeconds,
      rtf: totalMs / 1000 / audioSeconds,
      engineStats: { ...engine.stats },
    },
    pcm: waveform,
    sampleRate: fixture.sample_rate,
  };
}

// ---------------------------------------------------------------------------
// Page wiring

function element<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`#${id} is missing from the page`);
  return node as T;
}

function main(): void {
  const button = element<HTMLButtonElement>("run");
  const textArea = element<HTMLTextAreaElement>("text");
  const modeSelect = element<HTMLSelectElement>("mode");
  const seedInput = element<HTMLInputElement>("seed");
  const bar = element<HTMLProgressElement>("bar");
  const status = element<HTMLParagraphElement>("status");
  const player = element<HTMLAudioElement>("player");
  const metrics = element<HTMLDListElement>("metrics");

  window.__result = null;
  let loadedPromise: Promise<Loaded> | null = null;

  const formatMB = (bytes: number) => `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  const report: Progress = (stage, detail) => {
    if (detail?.loaded !== undefined && detail.total) {
      bar.value = detail.loaded / detail.total;
      status.textContent = `${stage} — ${formatMB(detail.loaded)} / ${formatMB(detail.total)}`;
    } else if (detail?.loaded !== undefined) {
      bar.removeAttribute("value");
      status.textContent = `${stage} — ${formatMB(detail.loaded)}`;
    } else {
      bar.removeAttribute("value");
      status.textContent = stage;
    }
  };

  button.addEventListener("click", async () => {
    button.disabled = true;
    bar.classList.remove("hidden");
    metrics.classList.add("hidden");
    window.__result = null;
    try {
      // Loaded once, kept across runs: weights stay resident on the device and
      // a second click only pays the generation itself.
      loadedPromise ??= loadEverything(report);
      const loaded = await loadedPromise;

      const mode = modeSelect.value === "sample" ? "sample" : "greedy";
      const seed = Number(seedInput.value) || 42;
      const { result, pcm, sampleRate } = await generate(loaded, textArea.value, mode, seed, report);

      bar.classList.add("hidden");
      status.textContent = "完了";
      player.src = URL.createObjectURL(toWav(pcm, sampleRate));
      player.classList.remove("hidden");
      element("m-adapter").textContent = result.adapter;
      element("m-prompt-tokens").textContent = String(result.promptIds.length);
      element("m-gen-tokens").textContent =
        `${result.generatedIds.length} (${result.speechIndices.length} speech)` +
        (result.nonSpeechIds.length > 0 ? ` + ${result.nonSpeechIds.length} unexpected` : "");
      element("m-lm-ms").textContent = `${result.lmMs.toFixed(0)} ms (${result.lmSteps} steps)`;
      element("m-lm-tps").textContent = `${result.lmTokensPerSec.toFixed(1)} tok/s`;
      element("m-decode-ms").textContent = `${result.decodeMs.toFixed(0)} ms`;
      element("m-total-ms").textContent = `${result.totalMs.toFixed(0)} ms`;
      element("m-rtf").textContent = `${result.rtf.toFixed(2)} (${result.audioSeconds.toFixed(2)} s audio in ${(result.totalMs / 1000).toFixed(2)} s)`;
      // The E2E hook — set before #metrics unhides, so a driver that waited on
      // the selector always sees the finished result.
      window.__result = result;
      metrics.classList.remove("hidden");
    } catch (error) {
      bar.classList.add("hidden");
      const message = error instanceof Error ? error.message : String(error);
      status.textContent = `失敗: ${message}`;
      window.__result = { status: "error", error: message };
    } finally {
      button.disabled = false;
    }
  });
}

main();
