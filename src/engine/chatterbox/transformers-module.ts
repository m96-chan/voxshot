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
  readonly AutoProcessor: {
    from_pretrained(
      modelId: string,
      options?: FromPretrainedOptions,
    ): Promise<ChatterboxProcessorLike>;
  };
}

/** How the engine gets hold of the library. Replaceable in tests. */
export type TransformersModuleLoader = () => Promise<TransformersModule>;
