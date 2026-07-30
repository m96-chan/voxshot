import {
  WorkerSynthesisEngine,
  VoxShot,
  isVoxShotError,
  toJapaneseReading,
  type LoadProgress,
  type PcmAudio,
} from "voxshot";

import { prewarmModelCache } from "./model-cache.js";

type EngineKind = "placeholder" | "chatterbox" | "chatterbox-multilingual";

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
 * VoxShot requires a reference voice before it can speak. The placeholder
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

const instances = new Map<EngineKind, Promise<VoxShot>>();
const clonedSource = new Map<EngineKind, File | "synthetic">();

function getInstance(kind: EngineKind): Promise<VoxShot> {
  let instance = instances.get(kind);
  if (!instance) {
    instance = createInstance(kind);
    instance.catch(() => instances.delete(kind));
    instances.set(kind, instance);
  }
  return instance;
}

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

const seconds = (fromMs: number) => ((performance.now() - fromMs) / 1000).toFixed(1);

async function createInstance(kind: EngineKind): Promise<VoxShot> {
  if (kind === "chatterbox") {
    log("Preparing the Chatterbox model… (first run downloads ~1.5 GB)");
    const downloadStart = performance.now();
    await prewarmModelCache({
      dtype: await pickLanguageModelDtype(),
      log,
      cacheStorage: globalThis.caches,
    });
    log(
      `Downloads / cache check took ${seconds(downloadStart)}s. Initializing the model (loading weights, compiling kernels)… this can take a minute or two.`,
    );
    const initStart = performance.now();
    // Inference runs in a Web Worker so the page stays responsive during
    // model load, cloning and synthesis.
    const worker = new Worker(new URL("./tts.worker.ts", import.meta.url), { type: "module" });
    const engine = new WorkerSynthesisEngine(worker, { onProgress: createProgressLogger() });
    const tts = await VoxShot.create({ engine });
    log(`Chatterbox ready (device: ${tts.device}, init ${seconds(initStart)}s)`);
    return tts;
  }

  if (kind === "chatterbox-multilingual") {
    // Probe the file the download script fetches last: Vite answers missing
    // paths with index.html (HTTP 200), which would otherwise surface as a
    // baffling "protobuf parsing failed" from ONNX Runtime.
    const probe = await fetch("/models/chatterbox-multilingual/onnx/language_model_q4.onnx_data", {
      method: "HEAD",
    });
    if (!probe.ok || (probe.headers.get("Content-Type") ?? "").includes("text/html")) {
      throw new Error(
        "Local multilingual model not found (or the download is still running). Run: bash scripts/download-multilingual.sh (in examples/browser) to completion, then reload.",
      );
    }
    log(
      "Loading the local Chatterbox Multilingual model (loading weights, compiling kernels)… this can take a minute or two.",
    );
    const initStart = performance.now();
    const worker = new Worker(new URL("./tts-multilingual.worker.ts", import.meta.url), {
      type: "module",
    });
    const engine = new WorkerSynthesisEngine(worker, { onProgress: createProgressLogger() });
    const tts = await VoxShot.create({ engine });
    log(`Chatterbox Multilingual ready (device: ${tts.device}, init ${seconds(initStart)}s)`);
    return tts;
  }

  const tts = await VoxShot.create();
  log(`Placeholder engine ready (device: ${tts.device})`);
  return tts;
}

async function ensureVoice(kind: EngineKind, tts: VoxShot): Promise<boolean> {
  if (kind !== "placeholder") {
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
  if (kind === "chatterbox" && /[^\x00-\x7F]/.test(text)) {
    log(
      "⚠ The English Chatterbox model cannot speak Japanese: kana map to unknown tokens and the output will be (near) silent. Use the Placeholder engine to hear the text pipeline — real Japanese speech is tracked in issue #25.",
    );
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

const DEFAULT_SAMPLE = "Hello from VoxShot! This is a browser text to speech demo.";
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
  const kind = engineSelect.value as EngineKind;
  const needsReference = kind !== "placeholder";
  referenceLabel.hidden = !needsReference;
  referenceInput.hidden = !needsReference;

  // Start loading the selected engine right away, so the minute of session
  // initialization overlaps with the user picking a reference file and
  // typing, instead of starting only when Speak is pressed.
  if (kind !== "placeholder") {
    getInstance(kind).catch((cause) => {
      log(`Error: ${cause instanceof Error ? cause.message : String(cause)}`);
      console.error(cause);
    });
  }
});

speakButton.addEventListener("click", () => {
  speakButton.disabled = true;
  speak()
    .catch((cause) => {
      if (isVoxShotError(cause)) {
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
