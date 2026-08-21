import { describe, expect, it } from "vitest";

import {
  MIOTTS_ENGINE_NAME,
  MIOTTS_SAMPLE_RATE,
  SPEAKER_EMBEDDING_SIZE,
  assertUsableVoice,
  resolveEngineOptions,
  resolveGeneration,
  type MioTtsEngineOptions,
} from "../../../src/engine/miotts/options.js";
import type { MioTtsWeightSource } from "../../../src/engine/miotts/weights.js";
import { InvalidInputError } from "../../../src/errors.js";
import type { SynthesisRequest } from "../../../src/engine/types.js";
import type { VoiceEmbedding } from "../../../src/voice/types.js";

const weights: MioTtsWeightSource = { load: async () => new ArrayBuffer(0) };

const options = (overrides: Partial<MioTtsEngineOptions> = {}): MioTtsEngineOptions => ({
  weights,
  ...overrides,
});

function voice(overrides: Partial<VoiceEmbedding> = {}): VoiceEmbedding {
  return {
    vector: new Float32Array(SPEAKER_EMBEDDING_SIZE),
    sampleRate: MIOTTS_SAMPLE_RATE,
    createdAt: 0,
    engine: MIOTTS_ENGINE_NAME,
    ...overrides,
  };
}

function request(overrides: Partial<SynthesisRequest> = {}): SynthesisRequest {
  return { text: "こんにちは", voice: voice(), speed: 1, ...overrides };
}

describe("resolveEngineOptions", () => {
  it("defaults to the reference server's own sampling settings", () => {
    const resolved = resolveEngineOptions(options());

    expect(resolved.temperature).toBe(0.8);
    expect(resolved.topP).toBe(1);
    expect(resolved.maxNewTokens).toBe(700);
    expect(resolved.maxSeqLen).toBe(768);
  });

  it("requires a weight source", () => {
    expect(() => resolveEngineOptions({} as MioTtsEngineOptions)).toThrow(InvalidInputError);
    expect(() => resolveEngineOptions({ weights: {} } as MioTtsEngineOptions)).toThrow(/load/);
  });

  it("accepts a smaller sequence budget", () => {
    // A caller synthesising one sentence at a time does not need room for 700
    // tokens, and the KV cache is sized from this: 768 costs 176 MiB, 256
    // costs 59 MiB, which matters when something else shares the GPU.
    expect(resolveEngineOptions(options({ maxSeqLen: 256 })).maxSeqLen).toBe(256);
  });

  it("rejects a sequence budget that cannot hold a prompt and a generation", () => {
    expect(() => resolveEngineOptions(options({ maxSeqLen: 0 }))).toThrow(InvalidInputError);
    expect(() => resolveEngineOptions(options({ maxSeqLen: 12.5 }))).toThrow(InvalidInputError);
  });

  it("rejects greedy decoding rather than letting a caller discover it hangs", () => {
    // Measured, and reproduced in the reference implementation itself: argmax
    // decoding on this model never emits eos for some inputs and runs to the
    // 700-token cap. Someone reaching for temperature 0 wants reproducibility,
    // which `seed` already gives them.
    const error = (() => {
      try {
        resolveEngineOptions(options({ temperature: 0 }));
      } catch (thrown: unknown) {
        return thrown as Error;
      }
      return null;
    })();

    expect(error).toBeInstanceOf(InvalidInputError);
    expect(error?.message).toMatch(/seed/);
  });

  it("rejects a top-p outside (0, 1]", () => {
    expect(() => resolveEngineOptions(options({ topP: 0 }))).toThrow(InvalidInputError);
    expect(() => resolveEngineOptions(options({ topP: 1.5 }))).toThrow(InvalidInputError);
    expect(resolveEngineOptions(options({ topP: 1 })).topP).toBe(1);
  });

  it("rejects a non-integer seed but accepts zero", () => {
    expect(() => resolveEngineOptions(options({ seed: 1.5 }))).toThrow(InvalidInputError);
    expect(resolveEngineOptions(options({ seed: 0 })).seed).toBe(0);
  });

  it("rejects a token budget that leaves nothing to generate", () => {
    expect(() => resolveEngineOptions(options({ maxNewTokens: 0 }))).toThrow(InvalidInputError);
  });
});

describe("resolveGeneration", () => {
  const engine = resolveEngineOptions(options());

  it("carries the engine's defaults through when the request says nothing", () => {
    expect(resolveGeneration(request(), engine)).toEqual({
      temperature: 0.8,
      topP: 1,
      seed: 42,
      maxNewTokens: 700,
    });
  });

  it("maps expressiveness onto the sampling temperature", () => {
    // MioTTS has no expressiveness dial of its own; temperature is the one
    // control that changes how much the delivery varies.
    expect(resolveGeneration(request({ expressiveness: 1.2 }), engine).temperature).toBe(1.2);
  });

  it("ignores speed, because the model has no speed control", () => {
    // 25 Hz token rate, fixed. The interface allows an engine to ignore what
    // it cannot do; this pins that it is ignored rather than half-applied.
    const slow = resolveGeneration(request({ speed: 0.5 }), engine);
    const fast = resolveGeneration(request({ speed: 2 }), engine);

    expect(slow).toEqual(fast);
  });

  it("rejects an expressiveness that would mean greedy decoding", () => {
    expect(() => resolveGeneration(request({ expressiveness: 0 }), engine)).toThrow(
      InvalidInputError,
    );
  });

  it("produces the same parameters for the same request every time", () => {
    // VoxShot's synthesis cache returns the first render for a repeated
    // (voice, text) pair. If generation were seeded from a clock or a running
    // stream, the cache would be handing back audio the engine would no longer
    // produce — so the seed is per request, not per engine lifetime.
    expect(resolveGeneration(request(), engine)).toEqual(resolveGeneration(request(), engine));
  });
});

describe("assertUsableVoice", () => {
  it("accepts a voice this engine produced", () => {
    expect(() => assertUsableVoice(voice())).not.toThrow();
  });

  it("accepts a voice with no engine recorded", () => {
    // Saved before the field existed, or hand-built from a constant the caller
    // extracted once — alibi-ai's case, where the speaker never changes.
    expect(() => assertUsableVoice(voice({ engine: undefined }))).not.toThrow();
  });

  it("rejects a voice another engine produced", () => {
    expect(() => assertUsableVoice(voice({ engine: "chatterbox" }))).toThrow(/chatterbox/);
  });

  it("rejects a vector that is not the codec's global embedding", () => {
    // 128 floats, from MioCodec's encoder. A vector of another width is a
    // voice from somewhere else that forgot to say so, and would otherwise
    // reach the decoder's conditioning and read past its end.
    expect(() => assertUsableVoice(voice({ vector: new Float32Array(256) }))).toThrow(
      /128/,
    );
  });
});
