import { resample } from "../../audio/pcm.js";
import type { ResolvedDevice } from "../../device.js";
import { InvalidInputError, LoadStalledError, VoxShotError } from "../../errors.js";
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
/**
 * How long loading may stay silent before it is declared stalled.
 *
 * Deliberately generous: session creation legitimately emits nothing while it
 * runs, measured at roughly 35 s on an idle machine and over two minutes on a
 * loaded one, so a tighter default would abandon healthy loads. This is a
 * backstop against hanging forever, not a latency target.
 */
const DEFAULT_STALL_TIMEOUT_MS = 300_000;
const MIN_SPEED = 0.25;
const MAX_SPEED = 4;

/** Speaker tensors `encode_speech` produces and `generate` consumes. */
const SPEAKER_TENSOR_NAMES = [
  "audio_features",
  "audio_tokens",
  "speaker_embeddings",
  "speaker_features",
] as const;

/**
 * Engine-level load milestones, emitted on the same channel as the file
 * progress forwarded from Transformers.js.
 *
 * The `load-` prefix keeps them distinct from the library's own statuses
 * (`initiate` / `download` / `progress` / `progress_total` / `done` / `ready`)
 * and from the generic `ready` that {@link exposeEngine} emits for any engine.
 *
 * Note there is deliberately no "downloads finished / compiling now" event.
 * Detecting that transition needs a trustworthy count of the files a load will
 * touch, and Transformers.js cannot supply one for this model: `chatterbox` is
 * absent from its `MODEL_TYPE_MAPPING`, so its expected-file list resolves to a
 * single-file encoder-only layout that does not exist in the repo. Emitting a
 * milestone inferred from silence would just move the consumer's guess into the
 * library and dress it up as a fact.
 */
export interface ChatterboxLifecycleEvent {
  readonly status: "load-start" | "load-fallback" | "load-ready";
  /** The plan this event concerns, as `device/dtype` — e.g. `webgpu/q4f16`. */
  readonly plan: string;
  readonly device: ResolvedDevice;
  /** Quantization of the language model, the only session that varies. */
  readonly dtype: string;
  /** Why the plan was abandoned. Only present on `load-fallback`. */
  readonly reason?: string;
}

/** Everything {@link ChatterboxEngineOptions.onProgress} may receive. */
export type ChatterboxLoadEvent = LoadProgress | ChatterboxLifecycleEvent;

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
  /**
   * Called with model download progress forwarded from Transformers.js, and
   * with the engine's own {@link ChatterboxLifecycleEvent} milestones.
   */
  onProgress?: (progress: ChatterboxLoadEvent) => void;
  /**
   * Reject {@link ChatterboxEngine.load} when it produces no progress for this
   * long. The clock is reset by every progress event, so it measures silence
   * rather than total elapsed time — a total cap would abandon healthy loads,
   * because a 1.5 GB download legitimately takes minutes.
   *
   * A hung transfer never rejects on its own, so without this `load()` stays
   * pending forever and the caller has no error to catch. Set `0` to wait
   * indefinitely.
   *
   * @defaultValue 300000
   */
  stallTimeoutMs?: number;
  /**
   * How to obtain `@huggingface/transformers`. Replace it in tests, or to
   * pin your own build of the library.
   */
  loadModule?: TransformersModuleLoader;
  /**
   * Whether the WebGPU adapter can run f16 shaders. Only consulted when the
   * resolved device is `webgpu`. Defaults to asking `navigator.gpu` for an
   * adapter and checking `features.has("shader-f16")` — an f16 model on a
   * device without it loads fine and then fails at the first inference, so
   * the check must happen before the load plans are built.
   */
  supportsFp16?: () => boolean | Promise<boolean>;
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
  readonly #onProgress: ((progress: ChatterboxLoadEvent) => void) | undefined;
  readonly #loadModule: TransformersModuleLoader;
  readonly #supportsFp16: () => boolean | Promise<boolean>;
  readonly #stallTimeoutMs: number;

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
    this.#supportsFp16 = options.supportsFp16 ?? webGpuSupportsFp16;
    this.#stallTimeoutMs = options.stallTimeoutMs ?? DEFAULT_STALL_TIMEOUT_MS;
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
    const fp16 = device === "webgpu" ? await this.#supportsFp16() : true;
    const plans = buildLoadPlans(device, this.#dtypeOverrides, fp16);
    const failures: string[] = [];

    for (const plan of plans) {
      this.#announce("load-start", plan);
      try {
        await this.#withStallGuard(async (notice) => {
          this.#model = await transformers.ChatterboxModel.from_pretrained(this.modelId, {
            device: plan.device,
            dtype: plan.dtype,
            // Always supplied, even without an `onProgress` consumer: the
            // stall guard needs these events to know the transfer is alive.
            progress_callback: (progress: LoadProgress) => {
              notice();
              this.#onProgress?.(progress);
            },
          });
          this.#processor = await transformers.AutoProcessor.from_pretrained(this.modelId);
        });
        this.#plan = plan;
        this.#announce("load-ready", plan);
        return;
      } catch (cause) {
        // A stall is a transfer problem, not a device problem, so the
        // remaining plans would only stall in turn — give up immediately
        // rather than burning one timeout per plan.
        if (cause instanceof LoadStalledError) {
          throw cause;
        }
        const reason = cause instanceof Error ? cause.message : String(cause);
        this.#announce("load-fallback", plan, reason);
        failures.push(`${describePlan(plan)}: ${reason}`);
      }
    }

    throw new VoxShotError(
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
        throw new VoxShotError(`The speech encoder did not return "${name}".`);
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

  /** Emit one engine-level milestone, if anyone is listening. */
  #announce(status: ChatterboxLifecycleEvent["status"], plan: LoadPlan, reason?: string): void {
    this.#onProgress?.({
      status,
      plan: describePlan(plan),
      device: plan.device,
      dtype: plan.dtype.language_model,
      ...(reason === undefined ? {} : { reason }),
    });
  }

  /**
   * Run `operation`, rejecting with {@link LoadStalledError} if it goes quiet.
   *
   * `operation` receives a `notice` callback and must invoke it whenever the
   * work makes observable progress; each call restarts the clock. The
   * operation itself cannot be cancelled — Transformers.js exposes no handle
   * for that — so on a stall it is abandoned rather than aborted, and any
   * rejection it produces later is swallowed instead of surfacing as an
   * unhandled rejection.
   */
  async #withStallGuard<T>(operation: (notice: () => void) => Promise<T>): Promise<T> {
    if (this.#stallTimeoutMs <= 0) {
      return operation(() => {});
    }

    let timer: ReturnType<typeof setTimeout> | undefined;
    let declareStalled: ((reason: unknown) => void) | undefined;

    const restart = (): void => {
      if (timer !== undefined) {
        clearTimeout(timer);
      }
      timer = setTimeout(() => {
        declareStalled?.(new LoadStalledError(this.#stallTimeoutMs));
      }, this.#stallTimeoutMs);
    };

    const stalled = new Promise<never>((_, reject) => {
      declareStalled = reject;
    });

    restart();
    const running = operation(restart);
    running.catch(() => {});

    try {
      return await Promise.race([running, stalled]);
    } finally {
      if (timer !== undefined) {
        clearTimeout(timer);
      }
    }
  }

  async #importModule(): Promise<TransformersModule> {
    try {
      return await this.#loadModule();
    } catch (cause) {
      throw new VoxShotError(
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
      throw new VoxShotError("ChatterboxEngine is not loaded. Call load() first.");
    }
    return { model: this.#model, processor: this.#processor, transformers: this.#module };
  }
}

/** Render a plan as `device/dtype`, the form used in events and error text. */
function describePlan(plan: LoadPlan): string {
  return `${plan.device}/${plan.dtype.language_model}`;
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
 * Structural subset of `navigator.gpu`, declared locally because WebGPU types
 * ship in a separate package (same approach as `browser-platform.ts`).
 */
interface GpuNavigator {
  gpu?: {
    requestAdapter(): Promise<{ features?: { has(name: string): boolean } } | null>;
  };
}

/**
 * Default f16 probe, mirroring Transformers.js' internal
 * `isWebGpuFp16Supported`. When `navigator.gpu` is absent the environment is
 * not a browser running real WebGPU (tests, custom hosts), so the plans are
 * left untouched rather than silently degraded.
 */
async function webGpuSupportsFp16(): Promise<boolean> {
  const gpu = (globalThis.navigator as GpuNavigator | undefined)?.gpu;
  if (!gpu) {
    return true;
  }
  try {
    const adapter = await gpu.requestAdapter();
    return adapter?.features?.has("shader-f16") ?? false;
  } catch {
    return false;
  }
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
