import { resample } from "../../audio/pcm.js";
import type { ResolvedDevice } from "../../device.js";
import { InvalidInputError, ZeroVoxError } from "../../errors.js";
import type { PcmAudio } from "../../platform.js";
import type { VoiceTensor, VoiceTensorType } from "../../voice/types.js";
import type { EmbedResult, SynthesisEngine, SynthesisRequest } from "../types.js";
import type { DtypeConfig, LoadPlan } from "./dtype-plan.js";
import { buildLoadPlans } from "./dtype-plan.js";
import type {
  ChatterboxModelLike,
  ChatterboxProcessorLike,
  LoadProgress,
  TensorLike,
  TransformersModule,
  TransformersModuleLoader,
} from "./transformers-module.js";

/**
 * Sample rate of Chatterbox output.
 *
 * Fixed by the S3Gen vocoder: upstream `chatterbox/models/s3gen/const.py`
 * defines `S3GEN_SR = 24000`.
 */
export const CHATTERBOX_SAMPLE_RATE = 24_000;

/**
 * The English repo, not the multilingual one: as of 2026-07,
 * `onnx-community/chatterbox-multilingual-ONNX` ships no `config.json` /
 * `preprocessor_config.json` (its `library_name` is the Python `chatterbox`),
 * so Transformers.js `from_pretrained` cannot load it.
 */
const DEFAULT_MODEL_ID = "onnx-community/chatterbox-ONNX";
const DEFAULT_MAX_NEW_TOKENS = 256;
const DEFAULT_EXAGGERATION = 0.5;
const MIN_SPEED = 0.25;
const MAX_SPEED = 4;

/** Speaker tensors `encode_speech` produces and `generate` consumes. */
const SPEAKER_TENSOR_NAMES = [
  "audio_features",
  "audio_tokens",
  "speaker_embeddings",
  "speaker_features",
] as const;

export interface ChatterboxEngineOptions {
  /**
   * Hugging Face model id.
   *
   * @defaultValue "onnx-community/chatterbox-ONNX"
   */
  modelId?: string;
  /** Override the per-session quantization chosen for the device. */
  dtype?: Partial<DtypeConfig>;
  /**
   * Generation cap. 256 tokens is roughly 5-10 seconds of speech; raising it
   * increases latency for the chunk being rendered.
   *
   * @defaultValue 256
   */
  maxNewTokens?: number;
  /**
   * Chatterbox's expressiveness control.
   *
   * @defaultValue 0.5
   */
  exaggeration?: number;
  /** Called with model download / load progress events. */
  onProgress?: (progress: LoadProgress) => void;
  /**
   * How to obtain `@huggingface/transformers`. Replace it in tests, or to
   * pin your own build of the library.
   */
  loadModule?: TransformersModuleLoader;
}

/**
 * Zero-shot TTS backed by Chatterbox ONNX through Transformers.js v4.
 *
 * The model is split into four ONNX sessions (`embed_tokens`,
 * `speech_encoder`, `language_model`, `conditional_decoder`). Cloning runs the
 * speech encoder once and keeps its four output tensors, so every later
 * `synthesize` call skips straight to generation.
 *
 * `@huggingface/transformers` is an optional peer dependency: it is imported
 * lazily on first `load()`, so applications that supply another engine never
 * pay for it.
 */
export class ChatterboxEngine implements SynthesisEngine {
  readonly name = "chatterbox";
  readonly sampleRate = CHATTERBOX_SAMPLE_RATE;
  readonly modelId: string;

  readonly #dtypeOverrides: Partial<DtypeConfig> | undefined;
  readonly #maxNewTokens: number;
  readonly #exaggeration: number;
  readonly #onProgress: ((progress: LoadProgress) => void) | undefined;
  readonly #loadModule: TransformersModuleLoader;

  #module: TransformersModule | undefined;
  #model: ChatterboxModelLike | undefined;
  #processor: ChatterboxProcessorLike | undefined;
  #plan: LoadPlan | undefined;

  constructor(options: ChatterboxEngineOptions = {}) {
    this.modelId = options.modelId ?? DEFAULT_MODEL_ID;
    this.#dtypeOverrides = options.dtype;
    this.#maxNewTokens = options.maxNewTokens ?? DEFAULT_MAX_NEW_TOKENS;
    this.#exaggeration = options.exaggeration ?? DEFAULT_EXAGGERATION;
    this.#onProgress = options.onProgress;
    this.#loadModule = options.loadModule ?? defaultModuleLoader;
  }

  /** The device / dtype combination that actually loaded, once `load` ran. */
  get loadedPlan(): LoadPlan | undefined {
    return this.#plan;
  }

  /**
   * Download (or read from the browser cache) and initialise the model.
   *
   * Each plan from {@link buildLoadPlans} is tried in turn, so a GPU without
   * fp16 support degrades instead of failing outright.
   */
  async load(device: ResolvedDevice): Promise<void> {
    if (this.#model) {
      return;
    }

    const transformers = (this.#module ??= await this.#importModule());
    const plans = buildLoadPlans(device, this.#dtypeOverrides);
    const failures: string[] = [];

    for (const plan of plans) {
      try {
        this.#model = await transformers.ChatterboxModel.from_pretrained(this.modelId, {
          device: plan.device,
          dtype: plan.dtype,
          ...(this.#onProgress ? { progress_callback: this.#onProgress } : {}),
        });
        this.#processor = await transformers.AutoProcessor.from_pretrained(this.modelId);
        this.#plan = plan;
        return;
      } catch (cause) {
        failures.push(
          `${plan.device}/${plan.dtype.model}: ${cause instanceof Error ? cause.message : String(cause)}`,
        );
      }
    }

    throw new ZeroVoxError(
      `Failed to load "${this.modelId}" on every candidate configuration. Attempts: ${failures.join("; ")}`,
    );
  }

  /** Run the speech encoder over reference audio and keep its speaker tensors. */
  async embed(audio: PcmAudio): Promise<EmbedResult> {
    const { model, transformers } = this.#requireLoaded();
    if (audio.samples.length === 0) {
      throw new InvalidInputError("Reference audio must not be empty.");
    }

    const audioValues = new transformers.Tensor("float32", audio.samples, [
      1,
      audio.samples.length,
    ]);
    const outputs = (await model.encode_speech(audioValues)) as unknown as Record<string, TensorLike>;

    const tensors: Record<string, VoiceTensor> = {};
    for (const name of SPEAKER_TENSOR_NAMES) {
      const tensor = outputs[name];
      if (!tensor) {
        throw new ZeroVoxError(`The speech encoder did not return "${name}".`);
      }
      tensors[name] = toVoiceTensor(tensor);
    }
    return {
      vector: Float32Array.from(tensors.speaker_embeddings?.data as Float32Array),
      tensors,
    };
  }

  /** Render one chunk of speech with the given voice. */
  async synthesize(request: SynthesisRequest): Promise<Float32Array> {
    const { model, processor, transformers } = this.#requireLoaded();
    const { text, voice, speed } = request;

    if (!Number.isFinite(speed) || speed < MIN_SPEED || speed > MAX_SPEED) {
      throw new InvalidInputError(`speed must be between ${MIN_SPEED} and ${MAX_SPEED}.`);
    }
    if (voice.engine !== undefined && voice.engine !== this.name) {
      throw new InvalidInputError(
        `This voice was produced by the "${voice.engine}" engine and cannot be used with "${this.name}". Clone the reference audio again.`,
      );
    }
    if (!voice.tensors) {
      throw new InvalidInputError(
        "This voice carries no speaker tensors. Clone the reference audio with this engine first.",
      );
    }

    const trimmed = text.trim();
    if (trimmed.length === 0) {
      return new Float32Array(0);
    }

    const inputs = await processor(trimmed);
    const speaker: Record<string, TensorLike> = {};
    for (const name of SPEAKER_TENSOR_NAMES) {
      const tensor = voice.tensors[name];
      if (!tensor) {
        throw new InvalidInputError(`This voice is missing the "${name}" tensor.`);
      }
      speaker[name] = new transformers.Tensor(tensor.type, tensor.data, [...tensor.dims]);
    }

    const waveform = await model.generate({
      ...inputs,
      ...speaker,
      exaggeration: this.#exaggeration,
      max_new_tokens: this.#maxNewTokens,
    });

    const samples = Float32Array.from(waveform.data as Float32Array);
    if (speed === 1) {
      return samples;
    }
    // Chatterbox exposes no duration control, so speed is applied to the
    // rendered waveform. Like changing playback rate, this shifts pitch.
    return resample(samples, this.sampleRate, this.sampleRate / speed);
  }

  /** Free the ONNX sessions. A later `load()` re-creates them. */
  async dispose(): Promise<void> {
    const model = this.#model;
    this.#model = undefined;
    this.#processor = undefined;
    this.#plan = undefined;
    await model?.dispose();
  }

  async #importModule(): Promise<TransformersModule> {
    try {
      return await this.#loadModule();
    } catch (cause) {
      throw new ZeroVoxError(
        "ChatterboxEngine needs the optional peer dependency \"@huggingface/transformers\" (v4). Install it, or pass a custom loadModule.",
        "UNKNOWN",
        { cause },
      );
    }
  }

  #requireLoaded(): {
    model: ChatterboxModelLike;
    processor: ChatterboxProcessorLike;
    transformers: TransformersModule;
  } {
    if (!this.#model || !this.#processor || !this.#module) {
      throw new ZeroVoxError("ChatterboxEngine is not loaded. Call load() first.");
    }
    return { model: this.#model, processor: this.#processor, transformers: this.#module };
  }
}

/** Copy a library tensor into a storable, structured-cloneable one. */
function toVoiceTensor(tensor: TensorLike): VoiceTensor {
  const type: VoiceTensorType = tensor.data instanceof BigInt64Array ? "int64" : "float32";
  return {
    type,
    dims: [...tensor.dims],
    data:
      tensor.data instanceof BigInt64Array
        ? BigInt64Array.from(tensor.data)
        : Float32Array.from(tensor.data),
  };
}

/**
 * Lazily import the optional peer dependency.
 *
 * The `as string` cast stops TypeScript from resolving the specifier at build
 * time — the package is optional, so a consumer who brings their own engine
 * must not need it installed. The emitted JavaScript still contains the plain
 * literal, so bundlers resolve and code-split it as usual.
 */
const defaultModuleLoader: TransformersModuleLoader = async () =>
  (await import("@huggingface/transformers" as string)) as TransformersModule;
