/**
 * The slice of `@huggingface/transformers` that VoxShot depends on.
 *
 * Declaring it structurally keeps the package an *optional* peer dependency:
 * consumers who bring their own engine never have to install it, and tests can
 * run the whole engine against a double instead of a 1 GB model download.
 *
 * Shapes below were read from `@huggingface/transformers@4.2.0`
 * (`src/models/chatterbox/*.js`, `src/models/session_config.js`).
 */

/** Minimal `Tensor` surface: element type, payload and shape. */
export interface TensorLike {
  readonly type: string;
  readonly data: Float32Array | BigInt64Array;
  readonly dims: readonly number[];
  /**
   * Release the backing buffer.
   *
   * Optional for the same reason as the criteria classes: an older build or a
   * test double without it still satisfies the type, and disposal degrades to
   * whatever the garbage collector manages.
   *
   * It matters most on WebGPU, where the buffer is not something the collector
   * can reason about or feel pressure from — dropping the last reference to a
   * GPU-backed tensor is not the same as freeing it.
   */
  dispose?(): void;
}

/** Speaker data returned by `ChatterboxModel.encode_speech`. */
export interface SpeechEncoderOutputs {
  readonly audio_features: TensorLike;
  readonly audio_tokens: TensorLike;
  readonly speaker_embeddings: TensorLike;
  readonly speaker_features: TensorLike;
}

/** Progress events emitted by `from_pretrained`. */
export interface LoadProgress {
  readonly status: string;
  readonly file?: string;
  readonly name?: string;
  readonly loaded?: number;
  readonly total?: number;
  readonly progress?: number;
}

/**
 * A model's `config.json`, as returned by `AutoConfig.from_pretrained`.
 *
 * Only `architectures` is named because that is the field the engine has to
 * repair; everything else is carried through untouched.
 */
export interface PretrainedConfigLike {
  architectures?: string[];
  [key: string]: unknown;
}

export interface FromPretrainedOptions {
  device?: string;
  dtype?: Record<string, string> | string;
  progress_callback?: (progress: LoadProgress) => void;
  [option: string]: unknown;
}

export interface ChatterboxModelLike {
  encode_speech(audioValues: TensorLike): Promise<SpeechEncoderOutputs>;
  /** Runs the language model and the conditional decoder; returns a waveform. */
  generate(params: Record<string, unknown>): Promise<TensorLike>;
  dispose(): Promise<unknown>;
}

/** `Processor` instances are callable (`Callable` base class). */
export type ChatterboxProcessorLike = (
  text: string,
  audio?: unknown,
) => Promise<Record<string, unknown>>;

/** Stopping criterion whose `interrupt()` ends generation at the next token. */
export interface InterruptableStoppingCriteriaLike {
  interrupt(): void;
}

/**
 * Base class for stopping criteria, invoked once per generated token.
 *
 * Subclassing it is how the engine counts what generation actually produced.
 * `generate` returns only a waveform, and a waveform's length depends on the
 * reference voice as well as the token count, so it cannot be used to recover
 * one.
 */
export interface StoppingCriteriaLike {
  _call(inputIds: number[][], scores: unknown): boolean[];
}

export interface TransformersModule {
  readonly Tensor: new (
    type: string,
    data: Float32Array | BigInt64Array,
    dims: number[],
  ) => TensorLike;
  readonly ChatterboxModel: {
    from_pretrained(
      modelId: string,
      options: FromPretrainedOptions,
    ): Promise<ChatterboxModelLike>;
  };
  /**
   * Optional so an older build, or a test double, still satisfies the type.
   * Without it a cancelled render cannot be interrupted — the caller stops
   * waiting, but the work runs to completion.
   */
  readonly InterruptableStoppingCriteria?: new () => InterruptableStoppingCriteriaLike;
  /**
   * Optional for the same reason as above: an older build or a test double
   * without it still satisfies the type, and truncation simply goes
   * unreported rather than being reported from a number that cannot support
   * the claim.
   */
  readonly StoppingCriteria?: new () => StoppingCriteriaLike;
  /**
   * Optional for the same reason as the two above, and with the same shape of
   * consequence: `loadModule` is public API, so a custom module written before
   * this member existed must keep working. Without it the architecture goes
   * unnamed and the load falls back to the layout it used before #45 — two
   * 404s and a download denominator seeded from `config.json` alone — which is
   * a worse load, not a failed one.
   */
  readonly AutoConfig?: {
    from_pretrained(
      modelId: string,
      options?: FromPretrainedOptions,
    ): Promise<PretrainedConfigLike>;
  };
  readonly AutoProcessor: {
    from_pretrained(
      modelId: string,
      options?: FromPretrainedOptions,
    ): Promise<ChatterboxProcessorLike>;
  };
}

/** How the engine gets hold of the library. Replaceable in tests. */
export type TransformersModuleLoader = () => Promise<TransformersModule>;
