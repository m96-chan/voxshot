import type { ResolvedDevice } from "../device.js";
import type { PcmAudio } from "../platform.js";
import type { VoiceEmbedding, VoiceTensor } from "../voice/types.js";

/** Everything an engine needs to render one chunk of speech. */
export interface SynthesisRequest {
  /** Already normalized text — engines do not normalize again. */
  readonly text: string;
  /** Speaker to render the text with. */
  readonly voice: VoiceEmbedding;
  /** Playback rate multiplier; `1` is the engine's natural pace. */
  readonly speed: number;
}

/**
 * What an engine extracts from reference audio.
 *
 * Engines whose speaker representation is a single vector may return that
 * `Float32Array` directly; models that need several tensors (Chatterbox keeps
 * four) return them alongside it.
 */
export interface EmbedResult {
  readonly vector: Float32Array;
  readonly tensors?: Readonly<Record<string, VoiceTensor>>;
}

/**
 * The seam between the VoxShot facade and an actual model.
 *
 * Swapping in an ONNX Runtime Web backend means implementing this interface —
 * nothing above it needs to change.
 */
export interface SynthesisEngine {
  /** Stable identifier, useful for logs and cache keys. */
  readonly name: string;
  /** Sample rate of the audio {@link SynthesisEngine.synthesize} returns. */
  readonly sampleRate: number;
  /** Prepare weights for the given device. Called once before use. */
  load(device: ResolvedDevice): Promise<void>;
  /** Extract a speaker embedding from mono reference audio. */
  embed(audio: PcmAudio): Promise<Float32Array | EmbedResult>;
  /** Render one chunk of speech. */
  synthesize(request: SynthesisRequest): Promise<Float32Array>;
  /** Release any resources held by the engine. */
  dispose(): Promise<void>;
}
