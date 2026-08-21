import { type SamplerOptions } from "../../../web-xpu-ops/llm/sampler.js";
import { fetchCheckpoint, toWav, type Fixture } from "../miocodec/browser.js";
import { MAX_SEQ_LEN, maxNewFor } from "./constants.js";
import { cpuBackend, decode, MIOCODEC_24K, Weights, type Backend } from "../miocodec/decoder.js";
import { encodeGlobal } from "../miocodec/encoder.js";
import { Gpu, gpuBackend } from "../miocodec/gpu.js";
import { Safetensors } from "../miocodec/safetensors.js";
import { createGpuEngine, type GpuEngine, type GpuEngineStats } from "./gpu-engine.js";
import { createSamplerStats, sampleNextTopK, xorshift32 } from "./sampler.js";
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
 *     voice: "default" | "reference",
 *     encodeMs?: number, embedding?: number[128],    // only when voice === "reference"
 *     promptIds: number[], generatedIds: number[],   // generatedIds includes the eos when one was produced
 *     speechIndices: number[], nonSpeechIds: number[], // nonSpeechIds excludes eos; should be []
 *     prefillMs, lmMs, lmSteps, lmTokensPerSec, decodeMs, totalMs, audioSeconds, rtf: number,
 *     samplerFallbacks: number,                      // steps that paid for upstream's full sort
 *       // rtf follows the codec spike's convention: processing seconds per
 *       // second of audio, so smaller is better and < 1 is faster than realtime.
 *   }
 *
 * On failure `window.__result = { status: "error", error: string }` and
 * `#status` shows the message. `__result` is reset to null when a run starts.
 *
 * Voice cloning has its own hook, because the encode happens on file *choice*,
 * not on run: after a file lands in `#refaudio` (with `#voice` = "reference"),
 * `window.__voice` moves "encoding" -> `{ status: "ready", key, encodeMs,
 * embedding: number[128] }` (or `{ status: "error", error }`), and
 * `window.__voiceWave()` returns the browser-resampled 24 kHz mono samples the
 * embedding was computed from — so a checker can run the reference encoder on
 * the *same* input and separate GPU-kernel error from resampler skew (the page
 * resamples with Chrome's WebAudio sinc, the golden used torchaudio's).
 */

const EOS_IDS = [151645, 151643];
const SAMPLE_RATE_EXPECTED = 24000;
/**
 * The reference clip is trimmed to this many seconds before it reaches the
 * encoder — the MioTTS reference server's own `max_reference_seconds = 20.0`.
 *
 * Not a nicety: WavLM's attention is O(T²), so a 3-minute clip means T≈9000
 * and 12·T² f32 scratch arrays, ~3.9 GB, which is a dead tab rather than a
 * slow one. `encodeGlobal` carries its own hard cap for any other caller; this
 * is the page matching the server it is a port of.
 */
const MAX_REFERENCE_SECONDS = 20;

interface RunResult {
  status: "done";
  adapter: string;
  mode: "greedy" | "sample";
  seed: number;
  /** Whose 128-dim global embedding conditioned the decoder. */
  voice: "default" | "reference";
  /** GPU encode wall time for the reference clip (cached per file, so paid once). */
  encodeMs?: number;
  /** The encoded embedding itself, for the E2E check. Reference voice only. */
  embedding?: number[];
  promptIds: number[];
  generatedIds: number[];
  speechIndices: number[];
  nonSpeechIds: number[];
  /** The prompt's share of lmMs — the KV-only steps (skipLogits) plus the one full step. */
  prefillMs: number;
  lmMs: number;
  lmSteps: number;
  lmTokensPerSec: number;
  decodeMs: number;
  totalMs: number;
  audioSeconds: number;
  rtf: number;
  /** gpu-engine.ts' allocation/submit counters — lets a driver assert the loop shape. */
  engineStats: GpuEngineStats;
  /**
   * Steps whose draw fell outside sampler.ts' top-k window and paid for
   * upstream's full-vocabulary sort. Greedy runs report 0 by construction.
   * A driver watches this because it is what the sampled tok/s hangs on:
   * ISSUE #120 was every step taking that path.
   */
  samplerFallbacks: number;
}

type VoiceHook =
  | { status: "encoding"; key: string }
  | { status: "ready"; key: string; encodeMs: number; embedding: number[] }
  | { status: "error"; error: string };

declare global {
  interface Window {
    __result: RunResult | { status: "error"; error: string } | null;
    /** Voice-clone E2E hook — see the module doc. */
    __voice: VoiceHook | null;
    /** The browser-resampled 24 kHz mono input of the current voice, for the check's same-input oracle. */
    __voiceWave: (() => number[]) | null;
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

  // The three bins in parallel (they are independent streams off the same
  // server); progress is aggregated under one label so the bar stays coherent.
  const parts: { loaded: number; total?: number }[] = [{ loaded: 0 }, { loaded: 0 }, { loaded: 0 }];
  const partReport = (index: number): Progress => (_stage, detail) => {
    if (detail?.loaded === undefined) return;
    parts[index] = { loaded: detail.loaded, total: detail.total };
    const loaded = parts.reduce((sum, p) => sum + p.loaded, 0);
    const total = parts.every((p) => p.total !== undefined)
      ? parts.reduce((sum, p) => sum + (p.total ?? 0), 0)
      : undefined;
    report("LM weights (q8)", { loaded, total });
  };
  const [codes, scales, norms] = await Promise.all([
    fetchWithProgress(`./miotts/q8/${manifest.files.codes.name}`, "LM weights (q8)", partReport(0)),
    fetchWithProgress(`./miotts/q8/${manifest.files.scales.name}`, "LM weights (q8)", partReport(1)),
    fetchWithProgress(`./miotts/q8/${manifest.files.norms.name}`, "LM weights (q8)", partReport(2)),
  ]);

  // MioCodec: the same checkpoint URL and streaming loader as mio-codec.html.
  // STARTED here, before the ~1s q8 unpack and the GPU upload, so its network
  // time overlaps that CPU/GPU work; progress stays silent until we actually
  // wait on it (two writers on one status line would just flicker).
  let checkpointVisible = false;
  const checkpointPromise = fetchCheckpoint((stage, detail) => {
    if (checkpointVisible) report(stage, detail);
  });
  checkpointPromise.catch(() => {}); // surfaced at the await below, not as an unhandled rejection

  report("unpacking the q8 weights");
  await new Promise((resolve) => setTimeout(resolve, 0)); // let the status paint before ~1s of packing
  const weights: Qwen3WeightsQ8 = loadWeightsQ8({ manifest, codes, scales, norms }); // sha256 skipped — module doc

  report("uploading the LM to the GPU");
  const engine = await createGpuEngine(device, weights, { maxSeqLen: MAX_SEQ_LEN });

  checkpointVisible = true;
  report("downloading the checkpoint");
  const checkpoint = await checkpointPromise;
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

/* -------------------------------------------------------------------------- *
 * Voice cloning: reference audio -> the codec encoder -> a 128-dim embedding
 * -------------------------------------------------------------------------- */

/**
 * The encoder weights (117 MB, `export_encoder_weights.py`'s artifact, served
 * by serve.mjs), fetched lazily on the FIRST reference-voice encode: the
 * default voice never pays for them. One promise, kept across files and runs
 * — like the LM/codec loads, a failure drops the cache so the next attempt
 * retries instead of re-awaiting a forever-rejected promise.
 */
let encoderWeightsPromise: Promise<Weights> | null = null;

function loadEncoderWeights(report: Progress): Promise<Weights> {
  encoderWeightsPromise ??= (async () => {
    const buffer = await fetchWithProgress(
      "./miotts/encoder-weights.safetensors",
      "encoder weights",
      report,
    );
    report("parsing the encoder weights");
    return new Weights(Safetensors.parse(buffer));
  })().catch((error: unknown) => {
    encoderWeightsPromise = null;
    throw error;
  });
  return encoderWeightsPromise;
}

/**
 * Decode an audio file to 24 kHz mono.
 *
 * `decodeAudioData` resamples to its context's rate per spec, so decoding on a
 * 24 kHz OfflineAudioContext usually IS the resample — one pass through the
 * browser's sinc resampler. The render fallback covers an implementation that
 * hands back the file's native rate instead (an `AudioBufferSourceNode`
 * resamples its buffer to the context rate while rendering). Either way this
 * is the **browser's** resampler, not torchaudio's polyphase that produced the
 * golden's 24 kHz input — check-tts.mjs measures that skew rather than
 * assuming it away.
 */
async function toMono24k(bytes: ArrayBuffer): Promise<Float32Array> {
  const probe = new OfflineAudioContext(1, 1, SAMPLE_RATE_EXPECTED);
  const decoded = await probe.decodeAudioData(bytes);

  const channels = decoded.numberOfChannels;
  const mono = new Float32Array(decoded.length);
  for (let c = 0; c < channels; c += 1) {
    const data = decoded.getChannelData(c);
    for (let i = 0; i < mono.length; i += 1) mono[i] = mono[i]! + data[i]! / channels;
  }
  if (decoded.sampleRate === SAMPLE_RATE_EXPECTED) return mono;

  const frames = Math.ceil((mono.length * SAMPLE_RATE_EXPECTED) / decoded.sampleRate);
  const context = new OfflineAudioContext(1, frames, SAMPLE_RATE_EXPECTED);
  const buffer = context.createBuffer(1, mono.length, decoded.sampleRate);
  buffer.copyToChannel(mono, 0);
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(context.destination);
  source.start();
  const rendered = await context.startRendering();
  return rendered.getChannelData(0).slice();
}

interface VoiceEntry {
  embedding: Float32Array;
  encodeMs: number;
  /** What the encoder actually saw, kept for `window.__voiceWave`. */
  wave: Float32Array;
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
  voice: VoiceEntry | null,
  report: Progress,
): Promise<GenerateOutcome> {
  const { engine, tokenizer, normalize, codecWeights, codecBackend, fixture } = loaded;
  // The LM never sees the voice — speaker identity enters only at the decoder,
  // as AdaLN-Zero conditioning. Same ids whichever embedding is used.
  const globalEmbedding = voice ? voice.embedding : Float32Array.from(fixture.global_embedding);

  const promptIds = tokenizer.encodeChat(normalize(text));
  if (promptIds.length + 1 >= MAX_SEQ_LEN) {
    throw new Error(`prompt is ${promptIds.length} tokens; the engine was built for maxSeqLen=${MAX_SEQ_LEN}`);
  }
  const maxNew = maxNewFor(promptIds.length);
  const sampler: SamplerOptions =
    mode === "greedy"
      ? { mode: "greedy" }
      // The MioTTS reference server's own sampling: temperature 0.8, top_p 1.0.
      : { mode: "top-p", temperature: 0.8, topP: 1.0, rng: xorshift32(seed) };
  // Sampling goes through sampler.ts, not `llm/sampler.js` directly: upstream's
  // top-p path sorts all 164,480 logits every step, which capped sampled runs
  // at ~26 tok/s regardless of the GPU (ISSUE #120). Same ids, ~14x faster.
  const samplerStats = createSamplerStats();

  engine.reset();
  report("generating speech tokens");
  const lmStart = performance.now();
  // Prefill = the decode path, one token at a time (see gpu-engine.ts' module
  // doc — 15 tokens do not earn a batched prefill). Only the last token's
  // logits matter, so every earlier step skips the 164k-row lm_head matvec
  // and its 657 KB readback entirely.
  let lmSteps = 0;
  let logits: Float32Array | null = null;
  for (let i = 0; i < promptIds.length; i += 1) {
    logits = await engine.decodeStep(promptIds[i]!, { skipLogits: i + 1 < promptIds.length });
    lmSteps += 1;
  }
  const prefillMs = performance.now() - lmStart;

  const generatedIds: number[] = [];
  for (let step = 0; step < maxNew; step += 1) {
    const id = sampleNextTopK(logits!, generatedIds, sampler, { stats: samplerStats });
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
    globalEmbedding,
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
      voice: voice ? "reference" : "default",
      ...(voice ? { encodeMs: voice.encodeMs, embedding: Array.from(voice.embedding) } : {}),
      promptIds,
      generatedIds,
      speechIndices,
      nonSpeechIds,
      prefillMs,
      lmMs,
      lmSteps,
      lmTokensPerSec: lmSteps / (lmMs / 1000),
      decodeMs,
      totalMs,
      audioSeconds,
      rtf: totalMs / 1000 / audioSeconds,
      engineStats: { ...engine.stats },
      samplerFallbacks: samplerStats.fallbacks,
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
  const voiceSelect = element<HTMLSelectElement>("voice");
  const refAudio = element<HTMLInputElement>("refaudio");
  const voiceStatus = element<HTMLSpanElement>("voice-status");
  const bar = element<HTMLProgressElement>("bar");
  const status = element<HTMLParagraphElement>("status");
  const player = element<HTMLAudioElement>("player");
  const metrics = element<HTMLDListElement>("metrics");

  window.__result = null;
  window.__voice = null;
  window.__voiceWave = null;
  let loadedPromise: Promise<Loaded> | null = null;

  // Embeddings are cached per file (name + size + lastModified — enough to
  // tell one chosen file from another without hashing 5 MB of WAV; the mtime
  // is what separates a re-recorded take from the identically-named,
  // identically-sized one it replaced), so re-picking the same clip, or
  // re-running with it, never re-encodes.
  const voiceCache = new Map<string, VoiceEntry>();
  let activeVoice: VoiceEntry | null = null;
  /** The in-flight encode; a run with #voice=reference awaits it first. */
  let voicePromise: Promise<void> | null = null;
  /**
   * Bumped on every file choice, captured by that choice's encode.
   *
   * Encodes are slow (seconds) and a user can pick a second file while the
   * first is still running. Without this, a superseded encode would finish
   * later and overwrite `activeVoice` / `__voice` with ITS result — the page
   * would then synthesise in a voice the user had already replaced, and a late
   * failure would clobber a newer "ready". Every commit below checks that the
   * counter has not moved and bails if it has.
   */
  let voiceGeneration = 0;

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

  voiceSelect.addEventListener("change", () => {
    refAudio.classList.toggle("hidden", voiceSelect.value !== "reference");
  });

  refAudio.addEventListener("change", () => {
    const file = refAudio.files?.[0];
    if (!file) return;
    const key = `${file.name}:${file.size}:${file.lastModified}`;
    const generation = (voiceGeneration += 1);
    /** This encode is still the current selection — see `voiceGeneration`. */
    const current = () => generation === voiceGeneration;

    voicePromise = (async () => {
      const cached = voiceCache.get(key);
      if (cached) {
        // Reached synchronously (no await above), so nothing can have
        // superseded this choice yet — the guard is here anyway so that an
        // `await` creeping in above cannot silently reintroduce the race.
        if (!current()) return;
        activeVoice = cached;
        window.__voiceWave = () => Array.from(cached.wave);
        window.__voice = {
          status: "ready",
          key,
          encodeMs: cached.encodeMs,
          embedding: Array.from(cached.embedding),
        };
        voiceStatus.textContent = `${file.name}: encoded (cached, ${cached.encodeMs.toFixed(0)} ms)`;
        return;
      }

      activeVoice = null;
      window.__voice = { status: "encoding", key };
      voiceStatus.textContent = `${file.name}: encoding…`;
      bar.classList.remove("hidden");
      try {
        // The encode needs the shared GPU device, which arrives with
        // everything else — so choosing a file is also what triggers the
        // model loads if no run has yet. Same cache and retry story as the
        // run button's.
        loadedPromise ??= loadEverything(report).catch((error: unknown) => {
          loadedPromise = null;
          throw error;
        });
        const loaded = await loadedPromise;
        const weights = await loadEncoderWeights(report);

        report("decoding + resampling the reference audio");
        const decoded = await toMono24k(await file.arrayBuffer());

        // Trim to the reference server's own `max_reference_seconds = 20.0`
        // before anything expensive touches it — see MAX_REFERENCE_SECONDS.
        // The note is visible rather than silent: the embedding a user gets
        // from a 3-minute clip is the first 20 s of it, and that is something
        // they should be told, not left to infer from the voice.
        const limit = MAX_REFERENCE_SECONDS * SAMPLE_RATE_EXPECTED;
        const trimmed = decoded.length > limit;
        const wave = trimmed ? decoded.slice(0, limit) : decoded;
        const trimNote = trimmed ? ` (reference trimmed to ${MAX_REFERENCE_SECONDS} s)` : "";
        if (trimmed) {
          voiceStatus.textContent = `${file.name}: encoding…${trimNote}`;
        }

        report(`encoding the voice on ${loaded.codecBackend.name}${trimNote}`);
        await new Promise((resolve) => setTimeout(resolve, 0)); // let the status paint
        const started = performance.now();
        const embedding = await encodeGlobal(wave, weights, { backend: loaded.codecBackend });
        const encodeMs = performance.now() - started;

        const entry: VoiceEntry = { embedding, encodeMs, wave };
        // Cached under this file's own key whether or not it is still the
        // current choice — the work is done and correct for that file, and a
        // later re-pick should not repay for it.
        voiceCache.set(key, entry);
        // Everything past here is a COMMIT to the shared "current voice"
        // state, so a superseded encode stops at this line rather than
        // overwriting a newer selection.
        if (!current()) return;
        activeVoice = entry;
        window.__voiceWave = () => Array.from(wave);
        window.__voice = { status: "ready", key, encodeMs, embedding: Array.from(embedding) };
        voiceStatus.textContent = `${file.name}: encoded in ${encodeMs.toFixed(0)} ms${trimNote}`;
        status.textContent = `話者エンコード完了${trimNote}`;
      } catch (error) {
        // Same rule on the failure path: a stale encode's error must not
        // clobber a newer selection's "ready" (or its own in-flight status).
        // Still rethrown, so the promise a run awaits stays rejected.
        if (current()) {
          const message = error instanceof Error ? error.message : String(error);
          window.__voice = { status: "error", error: message };
          voiceStatus.textContent = `${file.name}: 失敗`;
          status.textContent = `話者エンコード失敗: ${message}`;
        }
        throw error;
      } finally {
        // The bar belongs to whatever is running NOW; a superseded encode
        // finishing must not hide the bar its successor is still using.
        if (current()) bar.classList.add("hidden");
      }
    })();
    voicePromise.catch(() => {}); // surfaced via __voice and at the next run's await
  });

  button.addEventListener("click", async () => {
    button.disabled = true;
    bar.classList.remove("hidden");
    metrics.classList.add("hidden");
    window.__result = null;
    try {
      // Loaded once, kept across runs: weights stay resident on the device and
      // a second click only pays the generation itself. On failure the cached
      // promise is dropped so the next click retries the load instead of
      // re-awaiting a forever-rejected promise.
      loadedPromise ??= loadEverything(report).catch((error: unknown) => {
        loadedPromise = null;
        throw error;
      });
      const loaded = await loadedPromise;

      const mode = modeSelect.value === "sample" ? "sample" : "greedy";
      // The input's value when it parses to a finite number (0 is a valid
      // seed); the default 42 only when it is empty or garbage.
      const seedText = seedInput.value.trim();
      const parsedSeed = Number(seedText);
      const seed = seedText !== "" && Number.isFinite(parsedSeed) ? parsedSeed : 42;

      let voice: VoiceEntry | null = null;
      if (voiceSelect.value === "reference") {
        if (!voicePromise) throw new Error("reference voice selected but no audio file chosen");
        await voicePromise; // finish (or surface) an in-flight encode first
        if (!activeVoice) throw new Error("the reference voice has no embedding");
        voice = activeVoice;
      }

      const { result, pcm, sampleRate } = await generate(loaded, textArea.value, mode, seed, voice, report);

      bar.classList.add("hidden");
      status.textContent = "完了";
      player.src = URL.createObjectURL(toWav(pcm, sampleRate));
      player.classList.remove("hidden");
      element("m-adapter").textContent = result.adapter;
      element("m-voice").textContent = result.voice === "reference" ? "reference audio" : "default (fixture)";
      element("m-encode-ms").textContent =
        result.encodeMs !== undefined ? `${result.encodeMs.toFixed(0)} ms (cached per file)` : "—";
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
