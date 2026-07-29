import { beforeEach, describe, expect, it, vi } from "vitest";

import { CHATTERBOX_SAMPLE_RATE, ChatterboxEngine } from "../../../src/engine/chatterbox/chatterbox-engine.js";
import type {
  TensorLike,
  TransformersModule,
} from "../../../src/engine/chatterbox/transformers-module.js";
import { InvalidInputError, ZeroVoxError } from "../../../src/errors.js";
import type { VoiceEmbedding } from "../../../src/voice/types.js";
import { toArray } from "../../helpers/tensor.js";

/** Stand-in for `@huggingface/transformers`' Tensor. */
class FakeTensor implements TensorLike {
  constructor(
    readonly type: string,
    readonly data: Float32Array | BigInt64Array,
    readonly dims: number[],
  ) {}
}

interface ModuleHarness {
  module: TransformersModule;
  attempts: { device: string; dtype: Record<string, string> }[];
  encodeCalls: TensorLike[];
  generateCalls: Record<string, unknown>[];
  processorCalls: string[];
  disposeCalls: number;
  failOn: (device: string, modelDtype: string) => boolean;
  waveform: Float32Array;
  progressEvents: unknown[];
}

function createModule(): ModuleHarness {
  const harness: ModuleHarness = {
    attempts: [],
    encodeCalls: [],
    generateCalls: [],
    processorCalls: [],
    disposeCalls: 0,
    failOn: () => false,
    waveform: Float32Array.from([0, 0.5, -0.5, 0.25]),
    progressEvents: [],
    module: undefined as unknown as TransformersModule,
  };

  const speakerOutputs = () => ({
    audio_features: new FakeTensor("float32", Float32Array.from([1, 2]), [1, 1, 2]),
    audio_tokens: new FakeTensor("int64", BigInt64Array.from([3n, 4n]), [1, 2]),
    speaker_embeddings: new FakeTensor("float32", Float32Array.from([0.1, 0.2, 0.3]), [1, 3]),
    speaker_features: new FakeTensor("float32", Float32Array.from([5, 6]), [1, 1, 2]),
  });

  harness.module = {
    Tensor: FakeTensor as unknown as TransformersModule["Tensor"],
    ChatterboxModel: {
      async from_pretrained(modelId, options) {
        const dtype = options.dtype as Record<string, string>;
        harness.attempts.push({ device: options.device as string, dtype });
        options.progress_callback?.({ status: "progress", file: modelId });
        if (harness.failOn(options.device as string, dtype.language_model as string)) {
          throw new Error(`no ${options.device} support`);
        }
        return {
          async encode_speech(audioValues: TensorLike) {
            harness.encodeCalls.push(audioValues);
            return speakerOutputs();
          },
          async generate(params: Record<string, unknown>) {
            harness.generateCalls.push(params);
            return new FakeTensor("float32", harness.waveform, [1, harness.waveform.length]);
          },
          async dispose() {
            harness.disposeCalls += 1;
          },
        };
      },
    },
    AutoProcessor: {
      async from_pretrained() {
        return async (text: string) => {
          harness.processorCalls.push(text);
          return {
            input_ids: new FakeTensor("int64", BigInt64Array.from([1n, 2n]), [1, 2]),
            attention_mask: new FakeTensor("int64", BigInt64Array.from([1n, 1n]), [1, 2]),
          };
        };
      },
    },
  };
  return harness;
}

const audio = (length = 16_000, sampleRate = 16_000) => ({
  samples: Float32Array.from({ length }, (_, index) => Math.sin(index / 8) * 0.5),
  sampleRate,
});

describe("ChatterboxEngine", () => {
  let harness: ModuleHarness;
  let engine: ChatterboxEngine;

  const createEngine = (overrides: Record<string, unknown> = {}) =>
    new ChatterboxEngine({ loadModule: async () => harness.module, ...overrides });

  beforeEach(() => {
    harness = createModule();
    engine = createEngine();
  });

  describe("metadata", () => {
    it("reports the Chatterbox output sample rate", () => {
      expect(engine.sampleRate).toBe(24_000);
      expect(CHATTERBOX_SAMPLE_RATE).toBe(24_000);
    });

    it("names itself after the engine, not the model id", () => {
      expect(engine.name).toBe("chatterbox");
    });

    it("defaults to the English ONNX model, the only repo from_pretrained can load", () => {
      expect(engine.modelId).toBe("onnx-community/chatterbox-ONNX");
    });

    it("accepts a custom model id", () => {
      expect(createEngine({ modelId: "onnx-community/chatterbox-multilingual-ONNX" }).modelId).toBe(
        "onnx-community/chatterbox-multilingual-ONNX",
      );
    });
  });

  describe("load", () => {
    it("loads the model and the processor once", async () => {
      await engine.load("webgpu");
      await engine.load("webgpu");

      expect(harness.attempts).toHaveLength(1);
    });

    it("uses the WebGPU plan first", async () => {
      await engine.load("webgpu");

      expect(harness.attempts[0]).toEqual({
        device: "webgpu",
        dtype: {
          embed_tokens: "fp32",
          speech_encoder: "fp32",
          language_model: "q4f16",
          conditional_decoder: "fp32",
        },
      });
    });

    it("falls back through the plans when a device fails", async () => {
      harness.failOn = (device, modelDtype) => device === "webgpu" && modelDtype === "q4f16";

      await engine.load("webgpu");

      expect(harness.attempts.map((attempt) => `${attempt.device}:${attempt.dtype.language_model}`)).toEqual([
        "webgpu:q4f16",
        "webgpu:q4",
      ]);
      expect(engine.loadedPlan?.device).toBe("webgpu");
    });

    it("falls back to wasm when WebGPU cannot load at all", async () => {
      harness.failOn = (device) => device === "webgpu";

      await engine.load("webgpu");

      expect(engine.loadedPlan?.device).toBe("wasm");
    });

    it("skips f16 plans when the injected probe denies shader-f16", async () => {
      await createEngine({ supportsFp16: async () => false }).load("webgpu");

      expect(harness.attempts[0]).toEqual({
        device: "webgpu",
        dtype: {
          embed_tokens: "fp32",
          speech_encoder: "fp32",
          language_model: "q4",
          conditional_decoder: "fp32",
        },
      });
    });

    it("keeps f16 plans when the injected probe confirms shader-f16", async () => {
      await createEngine({ supportsFp16: () => true }).load("webgpu");

      expect(harness.attempts[0]?.dtype.language_model).toBe("q4f16");
    });

    it("never probes f16 support for a wasm device", async () => {
      const supportsFp16 = vi.fn(async () => false);
      await createEngine({ supportsFp16 }).load("wasm");

      expect(supportsFp16).not.toHaveBeenCalled();
      expect(harness.attempts[0]?.dtype.language_model).toBe("q4");
    });

    describe("default f16 probe", () => {
      const loadWithAdapter = async (navigator: unknown) => {
        vi.stubGlobal("navigator", navigator);
        try {
          await createEngine().load("webgpu");
        } finally {
          vi.unstubAllGlobals();
        }
        return harness.attempts[0]?.dtype.language_model;
      };

      it("asks the adapter for shader-f16 and keeps f16 plans when present", async () => {
        const dtype = await loadWithAdapter({
          gpu: { requestAdapter: async () => ({ features: new Set(["shader-f16"]) }) },
        });

        expect(dtype).toBe("q4f16");
      });

      it("drops f16 plans when the adapter lacks shader-f16", async () => {
        const dtype = await loadWithAdapter({
          gpu: { requestAdapter: async () => ({ features: new Set() }) },
        });

        expect(dtype).toBe("q4");
      });

      it("drops f16 plans when the adapter cannot be requested", async () => {
        const dtype = await loadWithAdapter({
          gpu: {
            requestAdapter: async () => {
              throw new Error("gpu unavailable");
            },
          },
        });

        expect(dtype).toBe("q4");
      });

      it("leaves plans untouched when navigator.gpu does not exist", async () => {
        const dtype = await loadWithAdapter({});

        expect(dtype).toBe("q4f16");
      });
    });

    it("throws when every plan fails", async () => {
      harness.failOn = () => true;

      await expect(engine.load("wasm")).rejects.toBeInstanceOf(ZeroVoxError);
    });

    it("reports load progress", async () => {
      const onProgress = vi.fn();
      await createEngine({ onProgress }).load("wasm");

      expect(onProgress).toHaveBeenCalledWith({
        status: "progress",
        file: "onnx-community/chatterbox-ONNX",
      });
    });

    it("surfaces a missing transformers package clearly", async () => {
      const broken = new ChatterboxEngine({
        loadModule: async () => {
          throw new Error("Cannot find module '@huggingface/transformers'");
        },
      });

      await expect(broken.load("wasm")).rejects.toThrow(/@huggingface\/transformers/);
    });
  });

  describe("embed", () => {
    beforeEach(async () => {
      await engine.load("wasm");
    });

    it("passes the reference audio as a [1, n] float32 tensor", async () => {
      const reference = audio(1_000);

      await engine.embed(reference);

      const tensor = harness.encodeCalls[0] as FakeTensor;
      expect(tensor.type).toBe("float32");
      expect(tensor.dims).toEqual([1, 1_000]);
      expect(tensor.data).toHaveLength(1_000);
    });

    it("returns the speaker embeddings as the primary vector", async () => {
      const { vector } = await engine.embed(audio(1_000));

      expect(Array.from(vector)).toEqual([
        expect.closeTo(0.1, 6),
        expect.closeTo(0.2, 6),
        expect.closeTo(0.3, 6),
      ]);
    });

    it("returns every speaker tensor for reuse", async () => {
      const { tensors } = await engine.embed(audio(1_000));

      expect(Object.keys(tensors ?? {}).sort()).toEqual([
        "audio_features",
        "audio_tokens",
        "speaker_embeddings",
        "speaker_features",
      ]);
      expect(tensors?.audio_tokens?.type).toBe("int64");
      expect(toArray(tensors?.audio_tokens?.data)).toEqual([3n, 4n]);
    });

    it("requires the engine to be loaded", async () => {
      await expect(createEngine().embed(audio(1_000))).rejects.toBeInstanceOf(ZeroVoxError);
    });

    it("fails clearly when the speech encoder omits a tensor", async () => {
      const partial = new ChatterboxEngine({
        loadModule: async () => ({
          ...harness.module,
          ChatterboxModel: {
            async from_pretrained() {
              return {
                async encode_speech() {
                  return { speaker_embeddings: new FakeTensor("float32", Float32Array.from([1]), [1, 1]) };
                },
                async generate() {
                  return new FakeTensor("float32", Float32Array.from([0]), [1, 1]);
                },
                async dispose() {},
              };
            },
          },
        }) as unknown as TransformersModule,
      });
      await partial.load("wasm");

      await expect(partial.embed(audio(1_000))).rejects.toThrow(/audio_features/);
    });

    it("rejects empty reference audio", async () => {
      await expect(
        engine.embed({ samples: new Float32Array(0), sampleRate: 16_000 }),
      ).rejects.toBeInstanceOf(InvalidInputError);
    });
  });

  describe("synthesize", () => {
    let voice: VoiceEmbedding;

    beforeEach(async () => {
      await engine.load("wasm");
      const embedded = await engine.embed(audio(1_000));
      voice = {
        vector: embedded.vector,
        sampleRate: 24_000,
        createdAt: 0,
        engine: "chatterbox",
        tensors: embedded.tensors,
      };
    });

    it("tokenizes the text and generates a waveform", async () => {
      const samples = await engine.synthesize({ text: "こんにちは", voice, speed: 1 });

      expect(harness.processorCalls).toEqual(["こんにちは"]);
      expect(Array.from(samples)).toEqual([0, 0.5, -0.5, 0.25]);
    });

    it("passes the cached speaker tensors back to generate", async () => {
      await engine.synthesize({ text: "hi", voice, speed: 1 });
      const params = harness.generateCalls[0] as Record<string, FakeTensor>;

      expect(params.input_ids).toBeDefined();
      expect(params.attention_mask).toBeDefined();
      expect(params.speaker_embeddings?.dims).toEqual([1, 3]);
      expect(params.audio_tokens?.type).toBe("int64");
      expect(toArray(params.audio_tokens?.data)).toEqual([3n, 4n]);
    });

    it("applies the configured generation parameters", async () => {
      const configured = createEngine({ maxNewTokens: 512, exaggeration: 0.7 });
      await configured.load("wasm");
      await configured.embed(audio(1_000));

      await configured.synthesize({ text: "hi", voice, speed: 1 });

      const params = harness.generateCalls[0] as Record<string, unknown>;
      expect(params.max_new_tokens).toBe(512);
      expect(params.exaggeration).toBe(0.7);
    });

    it("resamples the waveform when a non default speed is requested", async () => {
      const samples = await engine.synthesize({ text: "hi", voice, speed: 2 });

      expect(samples).toHaveLength(2);
    });

    it("rejects an embedding produced by another engine", async () => {
      await expect(
        engine.synthesize({ text: "hi", voice: { ...voice, engine: "placeholder" }, speed: 1 }),
      ).rejects.toBeInstanceOf(InvalidInputError);
    });

    it("rejects an embedding without speaker tensors", async () => {
      const { tensors, ...withoutTensors } = voice;

      await expect(
        engine.synthesize({ text: "hi", voice: withoutTensors, speed: 1 }),
      ).rejects.toBeInstanceOf(InvalidInputError);
      expect(tensors).toBeDefined();
    });

    it("rejects a voice that is missing one of the speaker tensors", async () => {
      const { audio_tokens, ...rest } = voice.tensors as Record<string, never>;

      await expect(
        engine.synthesize({ text: "hi", voice: { ...voice, tensors: rest }, speed: 1 }),
      ).rejects.toThrow(/audio_tokens/);
      expect(audio_tokens).toBeDefined();
    });

    it("returns nothing for empty text without calling the model", async () => {
      const samples = await engine.synthesize({ text: "  ", voice, speed: 1 });

      expect(samples).toHaveLength(0);
      expect(harness.generateCalls).toHaveLength(0);
    });

    it("rejects a speed outside the supported range", async () => {
      await expect(
        engine.synthesize({ text: "hi", voice, speed: 0 }),
      ).rejects.toBeInstanceOf(InvalidInputError);
    });

    it("requires the engine to be loaded", async () => {
      await expect(
        createEngine().synthesize({ text: "hi", voice, speed: 1 }),
      ).rejects.toBeInstanceOf(ZeroVoxError);
    });
  });

  describe("dispose", () => {
    it("releases the model and allows a reload", async () => {
      await engine.load("wasm");

      await engine.dispose();
      await engine.load("wasm");

      expect(harness.disposeCalls).toBe(1);
      expect(harness.attempts).toHaveLength(2);
    });

    it("tolerates dispose before load", async () => {
      await expect(engine.dispose()).resolves.toBeUndefined();
    });
  });
});
