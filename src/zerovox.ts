import { normalizePeak, resample, toMono, trimSilence } from "./audio/pcm.js";
import { SynthesizedAudio } from "./audio/synthesized-audio.js";
import { createBrowserPlatform } from "./browser-platform.js";
import type { DevicePreference, ResolvedDevice } from "./device.js";
import { resolveDevice } from "./device.js";
import { PlaceholderEngine } from "./engine/placeholder-engine.js";
import type { SynthesisEngine } from "./engine/types.js";
import { DisposedError, InvalidInputError, NoVoiceError, VoiceNotFoundError } from "./errors.js";
import type { PcmAudio, Platform } from "./platform.js";
import { normalizeText } from "./text/normalize.js";
import { splitSentences } from "./text/segment.js";
import { IndexedDbVoiceStore } from "./voice/indexeddb-store.js";
import { MemoryVoiceStore } from "./voice/memory-store.js";
import type { VoiceEmbedding, VoiceStore } from "./voice/types.js";

/** Model identifiers this version understands. */
const KNOWN_MODELS = new Set(["default"]);

/** Reference audio is normalized to this peak before embedding. */
const REFERENCE_PEAK = 0.95;

/** Anything that can be turned into reference audio. */
export type VoiceSource = ArrayBuffer | ArrayBufferView | Blob | PcmAudio;

export interface ZeroVoxOptions {
  /**
   * Which backend to run inference on.
   *
   * @defaultValue "auto"
   */
  device?: DevicePreference;
  /**
   * Model identifier. Only `"default"` exists today.
   *
   * @defaultValue "default"
   */
  model?: string;
  /** Replace the synthesis engine, e.g. with an ONNX Runtime Web backend. */
  engine?: SynthesisEngine;
  /** Replace voice persistence. Defaults to IndexedDB, or memory without it. */
  voiceStore?: VoiceStore;
  /** Replace the browser bindings. Mostly useful for tests and non browser hosts. */
  platform?: Platform;
  /**
   * Longest chunk, in characters, handed to the engine at once.
   *
   * @defaultValue 120
   */
  maxChunkLength?: number;
  /**
   * Shortest chunk, in characters. Shorter chunks are merged with their
   * neighbours, because very short prompts make neural TTS models unstable.
   * `0` disables merging.
   *
   * @defaultValue 0
   */
  minChunkLength?: number;
  /** Clock used for embedding timestamps. Injectable for deterministic tests. */
  now?: () => number;
}

export interface SpeakOptions {
  /**
   * Playback rate multiplier passed to the engine.
   *
   * @defaultValue 1
   */
  speed?: number;
}

/**
 * The ZeroVox facade: clone a voice, then speak with it.
 *
 * ```ts
 * const tts = await ZeroVox.create();
 * await tts.cloneVoice(referenceAudioFile);
 * const audio = await tts.speak("Hello!");
 * await audio.play();
 * ```
 */
export class ZeroVox {
  readonly #engine: SynthesisEngine;
  readonly #platform: Platform;
  readonly #store: VoiceStore;
  readonly #maxChunkLength: number;
  readonly #minChunkLength: number;
  readonly #now: () => number;
  readonly #device: ResolvedDevice;

  #voice: VoiceEmbedding | undefined;
  #disposed = false;

  private constructor(
    device: ResolvedDevice,
    engine: SynthesisEngine,
    platform: Platform,
    store: VoiceStore,
    maxChunkLength: number,
    minChunkLength: number,
    now: () => number,
  ) {
    this.#device = device;
    this.#engine = engine;
    this.#platform = platform;
    this.#store = store;
    this.#maxChunkLength = maxChunkLength;
    this.#minChunkLength = minChunkLength;
    this.#now = now;
  }

  /** Resolve the device, load the engine and return a ready instance. */
  static async create(options: ZeroVoxOptions = {}): Promise<ZeroVox> {
    const model = options.model ?? "default";
    if (!KNOWN_MODELS.has(model)) {
      throw new InvalidInputError(
        `Unknown model "${model}". Available models: ${[...KNOWN_MODELS].join(", ")}.`,
      );
    }

    const maxChunkLength = options.maxChunkLength ?? 120;
    if (!Number.isFinite(maxChunkLength) || maxChunkLength <= 0) {
      throw new InvalidInputError("maxChunkLength must be a positive finite number.");
    }
    const minChunkLength = options.minChunkLength ?? 0;
    if (!Number.isFinite(minChunkLength) || minChunkLength < 0) {
      throw new InvalidInputError("minChunkLength must be a non negative finite number.");
    }

    const platform = options.platform ?? createBrowserPlatform();
    const engine = options.engine ?? new PlaceholderEngine();
    const store = options.voiceStore ?? createDefaultVoiceStore();

    const device = await resolveDevice(options.device, platform.gpu);
    await engine.load(device);

    return new ZeroVox(
      device,
      engine,
      platform,
      store,
      maxChunkLength,
      minChunkLength,
      options.now ?? Date.now,
    );
  }

  /** The backend inference actually runs on. */
  get device(): ResolvedDevice {
    return this.#device;
  }

  /** Sample rate of the audio this instance produces. */
  get sampleRate(): number {
    return this.#engine.sampleRate;
  }

  /** The voice currently used for synthesis, if any. */
  get currentVoice(): VoiceEmbedding | undefined {
    return this.#voice;
  }

  /**
   * Extract a voice from reference audio and make it the active voice.
   *
   * Accepts an encoded file (`ArrayBuffer`, `Blob`, `File`, typed array) or
   * ready made mono PCM.
   */
  async cloneVoice(source: VoiceSource): Promise<VoiceEmbedding> {
    this.#assertUsable();

    const pcm = await this.#toPcm(source);
    const prepared = this.#prepareReference(pcm);
    const embedded = await this.#engine.embed(prepared);
    const { vector, tensors } =
      embedded instanceof Float32Array ? { vector: embedded, tensors: undefined } : embedded;

    this.#voice = {
      vector,
      sampleRate: prepared.sampleRate,
      createdAt: this.#now(),
      engine: this.#engine.name,
      ...(tensors ? { tensors } : {}),
    };
    return this.#voice;
  }

  /** Synthesize the whole text and return it as one audio result. */
  async speak(text: string, options: SpeakOptions = {}): Promise<SynthesizedAudio> {
    const chunks: SynthesizedAudio[] = [];
    for await (const chunk of this.stream(text, options)) {
      chunks.push(chunk);
    }
    return SynthesizedAudio.concat(chunks);
  }

  /**
   * Synthesize the text chunk by chunk, yielding each as soon as it is ready,
   * so playback can start before the whole utterance is rendered.
   */
  async *stream(text: string, options: SpeakOptions = {}): AsyncGenerator<SynthesizedAudio> {
    this.#assertUsable();

    if (typeof text !== "string") {
      throw new InvalidInputError("text must be a string.");
    }
    const voice = this.#voice;
    if (!voice) {
      throw new NoVoiceError();
    }

    const chunks = splitSentences(text, {
      maxLength: this.#maxChunkLength,
      minLength: this.#minChunkLength,
    });
    if (chunks.length === 0) {
      throw new InvalidInputError("text must contain at least one speakable character.");
    }

    const speed = options.speed ?? 1;
    for (const chunk of chunks) {
      const samples = await this.#engine.synthesize({ text: chunk, voice, speed });
      yield new SynthesizedAudio(samples, this.#engine.sampleRate, this.#platform.player);
    }
  }

  /** Persist the active voice under `name`. */
  async saveVoice(name: string): Promise<void> {
    this.#assertUsable();
    if (!this.#voice) {
      throw new NoVoiceError();
    }
    await this.#store.save(name, this.#voice);
  }

  /** Load a persisted voice and make it active. */
  async useVoice(name: string): Promise<VoiceEmbedding> {
    this.#assertUsable();

    const stored = await this.#store.load(name);
    if (!stored) {
      throw new VoiceNotFoundError(name);
    }
    this.#voice = stored;
    return stored;
  }

  /**
   * Remove a persisted voice. The active voice stays loaded even when it is
   * the one being deleted, so synthesis in flight keeps working.
   */
  async deleteVoice(name: string): Promise<boolean> {
    this.#assertUsable();
    return this.#store.delete(name);
  }

  /** Every persisted voice name. */
  async listVoices(): Promise<string[]> {
    this.#assertUsable();
    return this.#store.list();
  }

  /** Release the engine. The instance is unusable afterwards. */
  async dispose(): Promise<void> {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    this.#voice = undefined;
    await this.#engine.dispose();
  }

  async #toPcm(source: VoiceSource): Promise<{ channels: readonly Float32Array[]; sampleRate: number }> {
    if (isPcmAudio(source)) {
      return { channels: [source.samples], sampleRate: source.sampleRate };
    }

    const encoded = await toArrayBuffer(source);
    return this.#platform.decoder.decode(encoded);
  }

  /** Down-mix, trim, resample and level reference audio for the engine. */
  #prepareReference(pcm: { channels: readonly Float32Array[]; sampleRate: number }): PcmAudio {
    const mono = toMono(pcm.channels);
    const trimmed = trimSilence(mono);
    if (trimmed.length === 0) {
      throw new InvalidInputError("Reference audio contains no audible signal.");
    }

    const resampled = resample(trimmed, pcm.sampleRate, this.#engine.sampleRate);
    return {
      samples: normalizePeak(resampled, REFERENCE_PEAK),
      sampleRate: this.#engine.sampleRate,
    };
  }

  #assertUsable(): void {
    if (this.#disposed) {
      throw new DisposedError();
    }
  }
}

function createDefaultVoiceStore(): VoiceStore {
  return IndexedDbVoiceStore.isSupported() ? new IndexedDbVoiceStore() : new MemoryVoiceStore();
}

function isPcmAudio(source: VoiceSource): source is PcmAudio {
  return (
    typeof source === "object" &&
    source !== null &&
    "samples" in source &&
    (source as PcmAudio).samples instanceof Float32Array
  );
}

async function toArrayBuffer(source: VoiceSource): Promise<ArrayBuffer> {
  if (source instanceof ArrayBuffer) {
    return source;
  }
  if (ArrayBuffer.isView(source)) {
    return source.buffer.slice(
      source.byteOffset,
      source.byteOffset + source.byteLength,
    ) as ArrayBuffer;
  }
  if (typeof (source as Blob)?.arrayBuffer === "function") {
    return (source as Blob).arrayBuffer();
  }
  throw new InvalidInputError(
    "Unsupported audio source. Pass an ArrayBuffer, a typed array, a Blob/File, or { samples, sampleRate }.",
  );
}
