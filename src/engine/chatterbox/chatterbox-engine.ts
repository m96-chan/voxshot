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
  InterruptableStoppingCriteriaLike,
  LoadProgress,
  StoppingCriteriaLike,
  PretrainedConfigLike,
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

/**
 * The architecture name Transformers.js registers Chatterbox under.
 *
 * Its `MODEL_TYPE_MAPPING` is keyed by class name (`ChatterboxModel`), but the
 * repo's `config.json` declares only `model_type: "chatterbox"` and no
 * `architectures`, and `resolve_model_type` consults exactly those two fields.
 * The lookup therefore misses and falls back to an encoder-only, single-file
 * layout — which is why loads used to probe `onnx/model_quantized.onnx` and
 * 404 twice before the real component files were fetched.
 *
 * Naming the architecture on the config we hand to `from_pretrained` makes the
 * lookup succeed. Beyond silencing the 404s and the warning, it repairs the
 * expected-file list that drives download totals: with the wrong layout,
 * Transformers.js seeds its progress denominator with `config.json` alone.
 */
const CHATTERBOX_ARCHITECTURE = "ChatterboxModel";
/**
 * Speech tokens a chunk needs, per character of text.
 *
 * Measured on `onnx-community/chatterbox-ONNX` with a speech-shaped reference,
 * generating with a cap high enough not to interfere:
 *
 * ```
 * chars   30    60    90   120   160
 * tokens  92   185   257   331   403
 * ```
 *
 * That is close to linear at roughly 2.4 tokens per character plus a fixed
 * ~20. The old fixed cap of 256 therefore ran out at about 89 characters —
 * inside the 120-character chunks the splitter produces by default, so a
 * full-length chunk was truncated by construction.
 *
 * Dense text (digits, CJK, heavy punctuation) costs more per character, which
 * is what {@link TOKEN_BUDGET_SAFETY} is for; the truncation report is the
 * backstop for when even that is not enough.
 */
const TOKENS_PER_CHARACTER = 2.4;
const TOKEN_BUDGET_BASE = 20;
const TOKEN_BUDGET_SAFETY = 1.25;
/** Enough for a very short prompt to finish naturally. */
const MIN_TOKEN_BUDGET = 96;


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

/**
 * Generation stopped because it reached its token budget rather than because
 * the utterance had finished, so the audio ends early.
 *
 * Carried on the same channel as the load milestones because that is the only
 * channel there is. A declared variant rather than something cast into place,
 * so consumers can narrow to it.
 */
export interface ChatterboxTruncationEvent {
  readonly status: "synthesize-truncated";
  /** The chunk that did not fit. */
  readonly text: string;
  /** The budget it ran out of. */
  readonly tokens: number;
}

/** Everything {@link ChatterboxEngineOptions.onProgress} may receive. */
export type ChatterboxLoadEvent =
  | LoadProgress
  | ChatterboxLifecycleEvent
  | ChatterboxTruncationEvent;

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
   * Fixed generation cap, honoured as written for every chunk.
   *
   * Leave it unset — the default — and the cap is sized to each chunk from its
   * length instead, so a full-length chunk is not truncated and a short one
   * does not pay for tokens it will never use. A fixed cap that is too small
   * for the text ends the audio mid-sentence; see {@link TOKENS_PER_CHARACTER}
   * for the measured relationship, and the `synthesize-truncated` event for
   * when it happens anyway.
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
  readonly #maxNewTokens: number | undefined;
  readonly #exaggeration: number;
  readonly #onProgress: ((progress: ChatterboxLoadEvent) => void) | undefined;
  readonly #loadModule: TransformersModuleLoader;
  readonly #supportsFp16: () => boolean | Promise<boolean>;
  readonly #stallTimeoutMs: number;

  #module: TransformersModule | undefined;
  /**
   * The engine's whole lifetime, as one value.
   *
   * Held in separate fields, "model built but processor not yet" and "loaded
   * but no plan" were both representable and both reachable — a load can be
   * abandoned while its operation is still running and still writing. One
   * value means a load either lands whole or not at all, and every state
   * carries what is needed to release whatever it holds.
   */
  #state: EngineState = { kind: "idle" };

  constructor(options: ChatterboxEngineOptions = {}) {
    this.modelId = options.modelId ?? DEFAULT_MODEL_ID;
    this.#dtypeOverrides = options.dtype;
    this.#maxNewTokens = options.maxNewTokens;
    this.#exaggeration = options.exaggeration ?? DEFAULT_EXAGGERATION;
    this.#onProgress = options.onProgress;
    this.#loadModule = options.loadModule ?? defaultModuleLoader;
    this.#supportsFp16 = options.supportsFp16 ?? webGpuSupportsFp16;
    this.#stallTimeoutMs = options.stallTimeoutMs ?? DEFAULT_STALL_TIMEOUT_MS;
  }

  /** The device / dtype combination that actually loaded, once `load` ran. */
  get loadedPlan(): LoadPlan | undefined {
    return this.#state.kind === "loaded" ? this.#state.plan : undefined;
  }

  /**
   * Download (or read from the browser cache) and initialise the model.
   *
   * Each plan from {@link buildLoadPlans} is tried in turn, so a GPU without
   * fp16 support degrades instead of failing outright.
   */
  async load(device: ResolvedDevice): Promise<void> {
    if (this.#state.kind === "loaded") {
      return;
    }
    if (this.#state.kind === "loading") {
      // Two callers racing would each build a full set of sessions, and one
      // set would end up with nothing referring to it and nothing to dispose
      // it. They share the first attempt.
      return this.#state.settled;
    }
    let abandoned = false;
    const settled = this.#load(device, () => abandoned).finally(() => {
      // Compare by identity, not by kind: an abandoned load finishing after a
      // newer one has taken over would otherwise hand the engine back to
      // `idle` while that newer load is still running, and the next caller
      // would start a third.
      if (this.#state === loading) {
        this.#state = { kind: "idle" };
      }
    });
    // Mark the stored promise observed. A concurrent caller receives this same
    // promise and its rejection, because sharing an attempt has to mean
    // sharing its outcome — resolving would leave them believing the engine is
    // ready.
    settled.catch(() => undefined);
    const loading: EngineState = {
      kind: "loading",
      settled,
      abandon: () => {
        abandoned = true;
      },
    };
    this.#state = loading;
    return settled;
  }

  async #load(device: ResolvedDevice, abandoned: () => boolean): Promise<void> {

    const transformers = (this.#module ??= await this.#importModule());
    // Resolved once, outside the plan loop: the config does not vary by plan,
    // and re-reading it per attempt would repeat work on every fallback.
    const config = await this.#resolveConfig(transformers);
    const fp16 = device === "webgpu" ? await this.#supportsFp16() : true;
    const plans = buildLoadPlans(device, this.#dtypeOverrides, fp16);
    const failures: string[] = [];

    for (const plan of plans) {
      this.#announce("load-start", plan);
      try {
        const loaded = await this.#withStallGuard(async (notice) => {
          const model = await transformers.ChatterboxModel.from_pretrained(this.modelId, {
            config,
            device: plan.device,
            dtype: plan.dtype,
            // Always supplied, even without an `onProgress` consumer: the
            // stall guard needs these events to know the transfer is alive.
            progress_callback: (progress: LoadProgress) => {
              notice();
              this.#onProgress?.(progress);
            },
          });
          try {
            const processor = await transformers.AutoProcessor.from_pretrained(this.modelId);
            return { kind: "loaded" as const, model, processor, plan };
          } catch (cause) {
            // The model exists and nothing else refers to it yet. Releasing it
            // here is the only chance — the caller never sees it.
            await model.dispose().catch(() => undefined);
            throw cause;
          }
        }, release);

        if (abandoned()) {
          // Disposed while this was in flight. The result belongs to nobody.
          await release(loaded);
          return;
        }
        this.#state = loaded;
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
    // No upper bound: the usable range is not documented upstream, and the
    // model accepts any float. This only rejects values that cannot be a
    // setting at all.
    const expressiveness = request.expressiveness ?? this.#exaggeration;
    if (!Number.isFinite(expressiveness) || expressiveness < 0) {
      throw new InvalidInputError("expressiveness must be a non negative finite number.");
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

    // `ChatterboxModel.generate` spreads its params straight into the base
    // `generate`, so a stopping criterion reaches the token loop and ends it
    // at the next step. Without this an abandoned render keeps the worker's
    // single execution slot busy for its full length (#67).
    const stopper = request.signal ? this.#interrupter(transformers, request.signal) : undefined;
    const counter = this.#counter(transformers);
    const criteria = [stopper, counter?.criterion].filter((c) => c !== undefined);

    // A cap the caller set is honoured as written; otherwise it is sized to
    // this chunk, so a full-length one is not truncated and a short one does
    // not pay for tokens it will never use.
    const budget = this.#maxNewTokens ?? tokenBudgetFor(trimmed);

    const waveform = await model.generate({
      ...inputs,
      ...speaker,
      // Chatterbox's own name for the knob the request calls expressiveness.
      exaggeration: expressiveness,
      max_new_tokens: budget,
      ...(criteria.length > 0 ? { stopping_criteria: criteria } : {}),
    });

    // Interruption leaves a truncated waveform behind; nobody is waiting for
    // it, and returning it would be indistinguishable from a short utterance.
    request.signal?.throwIfAborted();

    const samples = Float32Array.from(waveform.data as Float32Array);

    // Counted, not inferred. The waveform's length depends on the reference
    // voice as well as the token count, so recovering one from it was never
    // sound — see #90.
    //
    // The criteria run once per token, after it is appended, so a run stopped
    // by the cap ends on exactly `budget` steps. An utterance that finishes
    // naturally on its very last allowed token lands there too and is also
    // reported; the two are indistinguishable from here, and over-reporting at
    // the boundary is the safer way round.
    if (counter !== undefined && counter.steps >= budget) {
      this.#onProgress?.({
        status: "synthesize-truncated",
        text: trimmed,
        tokens: budget,
      });
    }

    if (speed === 1) {
      return samples;
    }
    // Chatterbox exposes no duration control, so speed is applied to the
    // rendered waveform. Like changing playback rate, this shifts pitch.
    return resample(samples, this.sampleRate, this.sampleRate / speed);
  }

  /** Free the ONNX sessions. A later `load()` re-creates them. */
  async dispose(): Promise<void> {
    const previous = this.#state;
    // Back to `idle`, not a terminal state: a reload after dispose is part of
    // the contract. What must not survive is the *result* of a load that was
    // in flight, which `abandon` takes care of.
    this.#state = { kind: "idle" };
    if (previous.kind === "loading") {
      // Tell the load its result is unwanted, so whatever it builds is
      // released rather than assigned to an engine that has gone.
      previous.abandon();
      return;
    }
    if (previous.kind === "loaded") {
      await previous.model.dispose();
    }
  }

  /**
   * Read the model's config and name its architecture if the repo omits it.
   *
   * See {@link CHATTERBOX_ARCHITECTURE} for why this is necessary. An
   * architecture the repo already declares is left alone: it knows its own
   * layout better than this default does.
   */
  async #resolveConfig(transformers: TransformersModule): Promise<PretrainedConfigLike> {
    const config = await transformers.AutoConfig.from_pretrained(this.modelId);
    if (config.architectures?.length) {
      return config;
    }
    return { ...config, architectures: [CHATTERBOX_ARCHITECTURE] };
  }

  /**
   * Build a stopping criterion wired to `signal`, when the library offers one.
   *
   * Returns undefined on builds without `InterruptableStoppingCriteria`, in
   * which case cancellation degrades to "the caller stops waiting" rather than
   * failing the render outright.
   */
  #interrupter(
    transformers: TransformersModule,
    signal: AbortSignal,
  ): InterruptableStoppingCriteriaLike | undefined {
    const Criteria = transformers.InterruptableStoppingCriteria;
    if (!Criteria) {
      return undefined;
    }
    const stopper = new Criteria();
    if (signal.aborted) {
      stopper.interrupt();
    } else {
      signal.addEventListener("abort", () => stopper.interrupt(), { once: true });
    }
    return stopper;
  }

  /**
   * A stopping criterion that stops nothing and counts everything.
   *
   * `StoppingCriteriaList` invokes each criterion once per generated token, so
   * the call count is exactly what generation produced — independent of the
   * reference voice, which is what the waveform-derived version was not.
   * Returns undefined on builds without `StoppingCriteria`, in which case
   * truncation goes unreported rather than reported from a number that cannot
   * support the claim.
   */
  #counter(
    transformers: TransformersModule,
  ): { criterion: StoppingCriteriaLike; readonly steps: number } | undefined {
    const Base = transformers.StoppingCriteria;
    if (!Base) {
      return undefined;
    }
    const state = { steps: 0 };
    class Counter extends (Base as new () => StoppingCriteriaLike) {
      override _call(inputIds: number[][]): boolean[] {
        state.steps += 1;
        return new Array(inputIds.length).fill(false);
      }
    }
    return {
      criterion: new Counter(),
      get steps() {
        return state.steps;
      },
    };
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
  async #withStallGuard<T>(
    operation: (notice: () => void) => Promise<T>,
    discard?: (value: T) => Promise<void> | void,
  ): Promise<T> {
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
    let abandoned = false;
    const running = operation(restart);
    running.then(
      (value) => {
        // A stalled transfer can recover after the guard has given up. What it
        // produced is unreachable, so whoever owns the result has to be given
        // the chance to release it.
        if (abandoned) {
          void discard?.(value);
        }
      },
      () => undefined,
    );

    try {
      return await Promise.race([running, stalled]);
    } catch (cause) {
      abandoned = true;
      throw cause;
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
    if (this.#state.kind !== "loaded" || !this.#module) {
      throw new VoxShotError("ChatterboxEngine is not loaded. Call load() first.");
    }
    return {
      model: this.#state.model,
      processor: this.#state.processor,
      transformers: this.#module,
    };
  }
}

/**
 * How many speech tokens `text` is likely to need.
 *
 * See {@link TOKENS_PER_CHARACTER} for the measurement this comes from.
 */
function tokenBudgetFor(text: string): number {
  const estimate = TOKENS_PER_CHARACTER * text.length + TOKEN_BUDGET_BASE;
  return Math.max(MIN_TOKEN_BUDGET, Math.ceil(estimate * TOKEN_BUDGET_SAFETY));
}

/**
 * Every state the engine can be in, and what each one owns.
 *
 * `loading` carries the handles needed to disown a load in flight; `loaded`
 * carries everything a synthesis needs. No other combination is expressible.
 */
type EngineState =
  | { readonly kind: "idle" }
  | { readonly kind: "loading"; readonly settled: Promise<void>; readonly abandon: () => void }
  | {
      readonly kind: "loaded";
      readonly model: ChatterboxModelLike;
      readonly processor: ChatterboxProcessorLike;
      readonly plan: LoadPlan;
    };

/** Release sessions nobody is waiting for any more. */
async function release(loaded: { model: ChatterboxModelLike }): Promise<void> {
  await loaded.model.dispose().catch(() => undefined);
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
