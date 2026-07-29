import {
  WorkerSynthesisEngine,
  ZeroVox,
  isZeroVoxError,
  toJapaneseReading,
  type LoadProgress,
  type PcmAudio,
} from "zerovox";

type EngineKind = "placeholder" | "chatterbox";

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) {
    throw new Error(`#${id} is missing from index.html`);
  }
  return found as T;
}

const textInput = element<HTMLTextAreaElement>("text");
const jaReadingInput = element<HTMLInputElement>("ja-reading");
const engineSelect = element<HTMLSelectElement>("engine");
const referenceInput = element<HTMLInputElement>("reference");
const referenceLabel = element<HTMLLabelElement>("reference-label");
const speakButton = element<HTMLButtonElement>("speak");
const statusOutput = element<HTMLPreElement>("status");
const player = element<HTMLAudioElement>("player");

function log(message: string): void {
  const time = new Date().toLocaleTimeString();
  statusOutput.textContent += `[${time}] ${message}\n`;
  statusOutput.scrollTop = statusOutput.scrollHeight;
}

/**
 * ZeroVox requires a reference voice before it can speak. The placeholder
 * engine only extracts pitch / brightness / loudness from it, so a synthetic
 * vowel-like tone is enough to exercise the whole pipeline without assets.
 */
function syntheticReference(sampleRate: number): PcmAudio {
  const samples = new Float32Array(sampleRate);
  for (let index = 0; index < samples.length; index += 1) {
    const t = index / sampleRate;
    samples[index] =
      0.6 * Math.sin(2 * Math.PI * 150 * t) +
      0.25 * Math.sin(2 * Math.PI * 300 * t) +
      0.1 * Math.sin(2 * Math.PI * 450 * t);
  }
  return { samples, sampleRate };
}

/** Log download progress once per file per 25% step to keep the log readable. */
function createProgressLogger(): (progress: Record<string, unknown>) => void {
  const lastStep = new Map<string, number>();
  return (raw) => {
    const progress = raw as unknown as LoadProgress;
    if (progress.status === "progress" && progress.file && progress.progress !== undefined) {
      const step = Math.floor(progress.progress / 25);
      if (step > (lastStep.get(progress.file) ?? -1)) {
        lastStep.set(progress.file, step);
        log(`Downloading ${progress.file}: ${progress.progress.toFixed(0)}%`);
      }
    } else if (progress.status === "done" && progress.file) {
      log(`Downloaded ${progress.file}`);
    }
  };
}

const instances = new Map<EngineKind, Promise<ZeroVox>>();
const clonedSource = new Map<EngineKind, File | "synthetic">();

function getInstance(kind: EngineKind): Promise<ZeroVox> {
  let instance = instances.get(kind);
  if (!instance) {
    instance = createInstance(kind);
    instance.catch(() => instances.delete(kind));
    instances.set(kind, instance);
  }
  return instance;
}

const MODEL_ID = "onnx-community/chatterbox-ONNX";
const MODEL_BASE = `https://huggingface.co/${MODEL_ID}/resolve/main/`;

/** WebGPU types ship separately, so declare the slice the dtype check needs. */
interface GpuNavigator {
  gpu?: {
    requestAdapter(): Promise<{ features: { has(name: string): boolean } } | null>;
  };
}

/** Mirrors the engine's plan: q4f16 needs the adapter to run f16 shaders. */
async function pickLanguageModelDtype(): Promise<"q4f16" | "q4"> {
  try {
    const adapter = await (navigator as GpuNavigator).gpu?.requestAdapter();
    return adapter?.features.has("shader-f16") ? "q4f16" : "q4";
  } catch {
    return "q4";
  }
}

/**
 * Fetch every model file into transformers-cache while the page is idle.
 *
 * `from_pretrained` initialises each ONNX session as soon as that session's
 * file arrives, and session init blocks the thread that is also consuming the
 * remaining download streams — the unread stream backs up until the server
 * resets it. Pre-warming the cache means the engine later loads everything
 * from cache and no live download is left to kill.
 */
async function prewarmModelCache(): Promise<void> {
  const dtype = await pickLanguageModelDtype();
  const files = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "onnx/embed_tokens.onnx",
    "onnx/embed_tokens.onnx_data",
    "onnx/speech_encoder.onnx",
    "onnx/speech_encoder.onnx_data",
    "onnx/conditional_decoder.onnx",
    "onnx/conditional_decoder.onnx_data",
    `onnx/language_model_${dtype}.onnx`,
    `onnx/language_model_${dtype}.onnx_data`,
  ];

  const cache = await caches.open("transformers-cache");
  for (const file of files) {
    const url = MODEL_BASE + file;
    if (await cache.match(url)) {
      continue;
    }
    log(`Fetching ${file}…`);
    // no-store bypasses the HTTP disk cache entirely: aborted downloads can
    // leave corrupt cache entries behind that wedge later fetches of the same
    // URL, and we persist into transformers-cache ourselves anyway.
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Downloading ${file} failed: HTTP ${response.status}`);
    }
    await cache.put(url, await withProgress(response, file));
  }
  log("All model files are in the browser cache.");
}

/** Re-materialise a response while logging download progress in 25% steps. */
async function withProgress(response: Response, file: string): Promise<Response> {
  const total = Number(response.headers.get("Content-Length") ?? 0);
  if (!response.body || !Number.isFinite(total) || total <= 0) {
    return response;
  }

  const reader = response.body.getReader();
  const parts: BlobPart[] = [];
  let received = 0;
  let lastStep = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    parts.push(value as BlobPart);
    received += value.length;
    const step = Math.floor((received / total) * 4);
    if (step > lastStep) {
      lastStep = step;
      log(`Fetching ${file}: ${Math.min(100, step * 25)}%`);
    }
  }

  return new Response(new Blob(parts), { status: 200, headers: response.headers });
}

async function createInstance(kind: EngineKind): Promise<ZeroVox> {
  if (kind === "chatterbox") {
    log("Preparing the Chatterbox model… (first run downloads ~1.5 GB)");
    await prewarmModelCache();
    // Inference runs in a Web Worker so the page stays responsive during
    // model load, cloning and synthesis.
    const worker = new Worker(new URL("./tts.worker.ts", import.meta.url), { type: "module" });
    const engine = new WorkerSynthesisEngine(worker, { onProgress: createProgressLogger() });
    const tts = await ZeroVox.create({ engine });
    log(`Chatterbox ready (device: ${tts.device})`);
    return tts;
  }

  const tts = await ZeroVox.create();
  log(`Placeholder engine ready (device: ${tts.device})`);
  return tts;
}

async function ensureVoice(kind: EngineKind, tts: ZeroVox): Promise<boolean> {
  if (kind === "chatterbox") {
    const file = referenceInput.files?.[0];
    if (!file) {
      log("Chatterbox needs reference audio. Please choose an audio file.");
      return false;
    }
    if (clonedSource.get(kind) !== file) {
      log(`Cloning a voice from "${file.name}"…`);
      await tts.cloneVoice(file);
      clonedSource.set(kind, file);
      log("Voice cloned");
    }
    return true;
  }

  if (clonedSource.get(kind) !== "synthetic") {
    await tts.cloneVoice(syntheticReference(tts.sampleRate));
    clonedSource.set(kind, "synthetic");
  }
  return true;
}

async function speak(): Promise<void> {
  const kind = engineSelect.value as EngineKind;
  let text = textInput.value.trim();
  if (!text) {
    log("Please enter some text.");
    return;
  }
  if (jaReadingInput.checked) {
    const reading = toJapaneseReading(text);
    if (reading !== text) {
      log(`Japanese reading: ${reading}`);
    }
    text = reading;
  }

  const tts = await getInstance(kind);
  if (!(await ensureVoice(kind, tts))) {
    return;
  }

  log(`Synthesizing… (${text.length} characters)`);
  const started = performance.now();
  const audio = await tts.speak(text);
  const elapsed = ((performance.now() - started) / 1000).toFixed(2);
  log(`Synthesized ${audio.duration.toFixed(2)}s of audio in ${elapsed}s`);

  if (player.src) {
    URL.revokeObjectURL(player.src);
  }
  player.src = URL.createObjectURL(audio.toBlob());
  player.hidden = false;

  log("Playing…");
  await audio.play();
  log("Playback finished");
}

const DEFAULT_SAMPLE = "Hello from ZeroVox! This is a browser text to speech demo.";
const JAPANESE_SAMPLE = "会議は3月4日の14:00からです。参加費は1,000円で、AIが50%の確率で答えます。";

jaReadingInput.addEventListener("change", () => {
  // Make the toggle self-explanatory: swap the untouched default sample for a
  // Japanese one (and back), but never overwrite text the user typed.
  if (jaReadingInput.checked && textInput.value === DEFAULT_SAMPLE) {
    textInput.value = JAPANESE_SAMPLE;
  } else if (!jaReadingInput.checked && textInput.value === JAPANESE_SAMPLE) {
    textInput.value = DEFAULT_SAMPLE;
  }
});

engineSelect.addEventListener("change", () => {
  const isChatterbox = engineSelect.value === "chatterbox";
  referenceLabel.hidden = !isChatterbox;
  referenceInput.hidden = !isChatterbox;
});

speakButton.addEventListener("click", () => {
  speakButton.disabled = true;
  speak()
    .catch((cause) => {
      if (isZeroVoxError(cause)) {
        log(`Error [${cause.code}] ${cause.message}`);
      } else {
        log(`Error: ${cause instanceof Error ? cause.message : String(cause)}`);
      }
      console.error(cause);
    })
    .finally(() => {
      speakButton.disabled = false;
    });
});

log("Ready. Type some text and press Speak.");
