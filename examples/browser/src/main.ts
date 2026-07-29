import {
  ChatterboxEngine,
  ZeroVox,
  isZeroVoxError,
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
function createProgressLogger(): (progress: LoadProgress) => void {
  const lastStep = new Map<string, number>();
  return (progress) => {
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

async function createInstance(kind: EngineKind): Promise<ZeroVox> {
  if (kind === "chatterbox") {
    log("Loading the Chatterbox model… (first run downloads a few hundred MB)");
    const engine = new ChatterboxEngine({ onProgress: createProgressLogger() });
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
  const text = textInput.value.trim();
  if (!text) {
    log("Please enter some text.");
    return;
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
