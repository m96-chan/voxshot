import type { SynthesisRequest } from "../types.js";
import { InvalidInputError } from "../../errors.js";
import type { VoiceEmbedding } from "../../voice/types.js";
import { MAX_NEW_TOKENS, MAX_SEQ_LEN } from "./constants.js";
import type { MioTtsWeightSource } from "./weights.js";

/**
 * What a caller can set on the MioTTS engine, and how a request's controls map
 * onto the model's own.
 *
 * The mapping is the interesting part, because `SynthesisRequest` is written
 * for engines in general and MioTTS answers two of its three controls with
 * "no such thing".
 */

/** Recorded on every voice this engine extracts. */
export const MIOTTS_ENGINE_NAME = "miotts";

/** MioCodec decodes at 24 kHz; nothing about it is configurable. */
export const MIOTTS_SAMPLE_RATE = 24_000;

/** Width of MioCodec's `global_embedding`, the whole speaker representation. */
export const SPEAKER_EMBEDDING_SIZE = 128;

/** The MioTTS reference server's own sampling settings. */
const DEFAULT_TEMPERATURE = 0.8;
const DEFAULT_TOP_P = 1;

/**
 * Default RNG seed.
 *
 * Fixed rather than drawn from a clock, and re-seeded per request rather than
 * carried across an engine's lifetime, because VoxShot caches renders: the
 * synthesis cache hands back the first result it saw for a (voice, text, …)
 * key. An engine whose second render of the same sentence differed would make
 * that cache a liar — it would be serving audio the engine no longer produces.
 * A caller who wants variety changes `seed`, which is a thing they can see.
 */
const DEFAULT_SEED = 42;

export interface MioTtsEngineOptions {
  /** Where the weights come from. See {@link MioTtsWeightSource}. */
  readonly weights: MioTtsWeightSource;
  /**
   * Run on a device the caller already owns, instead of requesting one.
   *
   * For an application that keeps another model resident — a local LLM, say —
   * a second device means a second set of GPU objects and no shared budget.
   */
  readonly device?: GPUDevice;
  /**
   * Sequence budget the KV cache is sized from.
   *
   * The cost is linear: 768 tokens is ~176 MiB of KV, 256 is ~59 MiB. A caller
   * synthesising one sentence per call does not need room for the model's full
   * 700-token generation, and the difference matters when something else is
   * sharing the GPU.
   *
   * @defaultValue 768
   */
  readonly maxSeqLen?: number;
  /** Hard cap on generated speech tokens. @defaultValue 700 */
  readonly maxNewTokens?: number;
  /**
   * Sampling temperature. Also what {@link SynthesisRequest.expressiveness}
   * maps onto. @defaultValue 0.8
   */
  readonly temperature?: number;
  /** Nucleus sampling mass. @defaultValue 1 */
  readonly topP?: number;
  /** RNG seed — see {@link DEFAULT_SEED} for why it is fixed. @defaultValue 42 */
  readonly seed?: number;
}

/** {@link MioTtsEngineOptions} with every default filled in. */
export interface ResolvedMioTtsOptions {
  readonly weights: MioTtsWeightSource;
  readonly device: GPUDevice | undefined;
  readonly maxSeqLen: number;
  readonly maxNewTokens: number;
  readonly temperature: number;
  readonly topP: number;
  readonly seed: number;
}

/** The parameters one `synthesize` call generates under. */
export interface GenerationParams {
  readonly temperature: number;
  readonly topP: number;
  readonly seed: number;
  readonly maxNewTokens: number;
}

function assertPositiveInteger(value: number, name: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new InvalidInputError(`${name} must be a positive integer.`);
  }
}

/**
 * Reject a temperature that would mean argmax decoding.
 *
 * Not caution: greedy decoding on this model does not reliably reach eos. Some
 * inputs cycle to the 700-token cap, and the reference implementation does the
 * same in float32, so it is the checkpoint's behaviour and not this port's.
 * Someone reaching for temperature 0 almost always wants a reproducible
 * render, which `seed` already gives — so the error says so instead of
 * handing them a call that takes half a minute and returns babble.
 */
function assertTemperature(value: number, name: string): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new InvalidInputError(
      `${name} must be a positive finite number. Greedy decoding (0) is not offered: this model ` +
        "does not reliably emit an end token without sampling, and generation runs to the cap " +
        "instead. For a reproducible render, set `seed` rather than removing sampling.",
    );
  }
}

export function resolveEngineOptions(options: MioTtsEngineOptions): ResolvedMioTtsOptions {
  if (typeof options?.weights?.load !== "function") {
    throw new InvalidInputError(
      "weights must be a MioTtsWeightSource — an object with a `load(part)` method returning the part's bytes.",
    );
  }

  const maxSeqLen = options.maxSeqLen ?? MAX_SEQ_LEN;
  assertPositiveInteger(maxSeqLen, "maxSeqLen");

  const maxNewTokens = options.maxNewTokens ?? MAX_NEW_TOKENS;
  assertPositiveInteger(maxNewTokens, "maxNewTokens");

  const temperature = options.temperature ?? DEFAULT_TEMPERATURE;
  assertTemperature(temperature, "temperature");

  const topP = options.topP ?? DEFAULT_TOP_P;
  if (!Number.isFinite(topP) || topP <= 0 || topP > 1) {
    throw new InvalidInputError("topP must be greater than 0 and at most 1.");
  }

  const seed = options.seed ?? DEFAULT_SEED;
  if (!Number.isInteger(seed)) {
    throw new InvalidInputError("seed must be an integer.");
  }

  return {
    weights: options.weights,
    device: options.device,
    maxSeqLen,
    maxNewTokens,
    temperature,
    topP,
    seed,
  };
}

/**
 * Turn one request into generation parameters.
 *
 * `speed` is absent by design. MioTTS generates speech tokens at a fixed
 * 25 Hz and has no pace control, and `SynthesisEngine` allows an engine to
 * ignore a control it does not have. It is ignored outright rather than
 * approximated by resampling the PCM, which would change the pitch of a voice
 * the caller cloned specifically to keep.
 *
 * `expressiveness` maps onto the sampling temperature. It is the only control
 * this model has that changes how much the delivery varies from one render to
 * the next, which is what the name asks for.
 */
export function resolveGeneration(
  request: SynthesisRequest,
  engine: ResolvedMioTtsOptions,
): GenerationParams {
  const temperature = request.expressiveness ?? engine.temperature;
  assertTemperature(temperature, "expressiveness");

  return {
    temperature,
    topP: engine.topP,
    seed: engine.seed,
    maxNewTokens: engine.maxNewTokens,
  };
}

/** Refuse a voice this engine cannot condition its decoder on. */
export function assertUsableVoice(voice: VoiceEmbedding): void {
  // `undefined` passes: an embedding saved before the field existed, or a
  // constant a caller extracted once and now ships as 128 numbers, is a
  // perfectly good voice that simply never recorded where it came from.
  if (voice.engine !== undefined && voice.engine !== MIOTTS_ENGINE_NAME) {
    throw new InvalidInputError(
      `This voice was produced by the "${voice.engine}" engine and cannot be used with ` +
        `"${MIOTTS_ENGINE_NAME}". Clone the reference audio again.`,
    );
  }
  if (voice.vector.length !== SPEAKER_EMBEDDING_SIZE) {
    throw new InvalidInputError(
      `This voice has ${voice.vector.length} dimensions; "${MIOTTS_ENGINE_NAME}" conditions its ` +
        `decoder on MioCodec's ${SPEAKER_EMBEDDING_SIZE}-dimension global embedding.`,
    );
  }
}
