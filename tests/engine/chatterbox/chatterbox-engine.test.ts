import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CHATTERBOX_SAMPLE_RATE, ChatterboxEngine } from "../../../src/engine/chatterbox/chatterbox-engine.js";
import type {
  ChatterboxLifecycleEvent,
  ChatterboxLoadEvent,
} from "../../../src/engine/chatterbox/chatterbox-engine.js";
import type {
  TensorLike,
  TransformersModule,
} from "../../../src/engine/chatterbox/transformers-module.js";
import { InvalidInputError, LoadStalledError, VoxShotError } from "../../../src/errors.js";
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
  /** When true for a plan, `from_pretrained` returns a promise that never settles. */
  hangOn: (device: string, modelDtype: string) => boolean;
  /** Last `progress_callback` handed to `from_pretrained`, so tests can drive it. */
  lastProgressCallback: ((progress: Record<string, unknown>) => void) | undefined;
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
    hangOn: () => false,
    lastProgressCallback: undefined,
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
        harness.lastProgressCallback = options.progress_callback as
          | ((progress: Record<string, unknown>) => void)
          | undefined;
        options.progress_callback?.({ status: "progress", file: modelId });
        if (harness.hangOn(options.device as string, dtype.language_model as string)) {
          // Models the reported bug: a dangling transfer leaves `from_pretrained`
          // pending forever — it neither resolves nor rejects.
          return new Promise(() => {});
        }
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

      await expect(engine.load("wasm")).rejects.toBeInstanceOf(VoxShotError);
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

    /**
     * #44: the winning plan used to be invisible. `buildLoadPlans` silently
     * degrades q4f16 -> q4 -> WASM, and a consumer had no way to tell which
     * plan won, or that a fallback had happened at all.
     */
    describe("lifecycle events", () => {
      const capture = () => {
        const events: ChatterboxLoadEvent[] = [];
        return { events, onProgress: (event: ChatterboxLoadEvent) => events.push(event) };
      };

      const isLifecycle = (event: ChatterboxLoadEvent): event is ChatterboxLifecycleEvent =>
        event.status.startsWith("load-");

      const lifecycle = (events: ChatterboxLoadEvent[]) => events.filter(isLifecycle);

      it("announces the plan before each attempt", async () => {
        const { events, onProgress } = capture();

        await createEngine({ onProgress }).load("wasm");

        expect(lifecycle(events)[0]).toMatchObject({
          status: "load-start",
          plan: "wasm/q4",
          device: "wasm",
          dtype: "q4",
        });
      });

      it("reports the winning plan when the model is up", async () => {
        const { events, onProgress } = capture();

        await createEngine({ onProgress }).load("webgpu");

        expect(lifecycle(events).at(-1)).toMatchObject({
          status: "load-ready",
          plan: "webgpu/q4f16",
          device: "webgpu",
          dtype: "q4f16",
        });
      });

      it("surfaces each abandoned plan with the reason it failed", async () => {
        harness.failOn = (device) => device === "webgpu";
        const { events, onProgress } = capture();

        await createEngine({ onProgress }).load("webgpu");

        expect(lifecycle(events).map((event) => [event.status, event.plan])).toEqual([
          ["load-start", "webgpu/q4f16"],
          ["load-fallback", "webgpu/q4f16"],
          ["load-start", "webgpu/q4"],
          ["load-fallback", "webgpu/q4"],
          ["load-start", "wasm/q4"],
          ["load-ready", "wasm/q4"],
        ]);
      });

      it("explains why a plan was abandoned", async () => {
        harness.failOn = (device) => device === "webgpu";
        const { events, onProgress } = capture();

        await createEngine({ onProgress }).load("webgpu");

        const fallback = lifecycle(events).find((event) => event.status === "load-fallback");
        expect(fallback?.reason).toContain("no webgpu support");
      });

      it("never claims readiness when every plan fails", async () => {
        harness.failOn = () => true;
        const { events, onProgress } = capture();

        await expect(createEngine({ onProgress }).load("wasm")).rejects.toBeInstanceOf(VoxShotError);

        expect(lifecycle(events).map((event) => event.status)).toEqual([
          "load-start",
          "load-fallback",
        ]);
      });

      it("keeps forwarding the library's own file progress alongside them", async () => {
        const { events, onProgress } = capture();

        await createEngine({ onProgress }).load("wasm");

        expect(events.map((event) => event.status)).toEqual([
          "load-start",
          "progress",
          "load-ready",
        ]);
      });

      it("loads normally when nobody is listening", async () => {
        await expect(createEngine().load("wasm")).resolves.toBeUndefined();
      });
    });

    /**
     * Regression coverage for #43: a transfer that hangs used to leave `load()`
     * pending forever, so consumers had no error to catch and no way to tell
     * "still loading" from "dead".
     */
    describe("stall timeout", () => {
      beforeEach(() => {
        vi.useFakeTimers();
      });

      afterEach(() => {
        vi.useRealTimers();
      });

      it("rejects with a LoadStalledError when a transfer produces no progress", async () => {
        harness.hangOn = () => true;
        const stalling = createEngine({ stallTimeoutMs: 1000 });

        const load = stalling.load("wasm");
        const assertion = expect(load).rejects.toBeInstanceOf(LoadStalledError);
        await vi.advanceTimersByTimeAsync(1000);

        await assertion;
      });

      it("tags the stall error with a machine readable code", async () => {
        harness.hangOn = () => true;
        const stalling = createEngine({ stallTimeoutMs: 1000 });

        const load = stalling.load("wasm");
        const assertion = expect(load).rejects.toMatchObject({ code: "LOAD_STALLED" });
        await vi.advanceTimersByTimeAsync(1000);

        await assertion;
      });

      it("keeps waiting while progress events keep arriving", async () => {
        harness.hangOn = () => true;
        const stalling = createEngine({ stallTimeoutMs: 1000 });
        const load = stalling.load("wasm");
        const settled = vi.fn();
        load.then(settled, settled);

        // Three quiet stretches, each shorter than the timeout, separated by
        // progress. A total-elapsed cap would have fired by now; a stall
        // timer must not.
        for (let tick = 0; tick < 3; tick += 1) {
          await vi.advanceTimersByTimeAsync(900);
          harness.lastProgressCallback?.({ status: "progress", file: "weights" });
        }
        await vi.advanceTimersByTimeAsync(0);

        expect(settled).not.toHaveBeenCalled();

        // Drain the still-pending load so the rejection is observed.
        await vi.advanceTimersByTimeAsync(1000);
        await expect(load).rejects.toBeInstanceOf(LoadStalledError);
      });

      it("abandons the whole load instead of stalling once per remaining plan", async () => {
        harness.hangOn = () => true;
        const stalling = createEngine({ stallTimeoutMs: 1000 });

        const load = stalling.load("webgpu");
        const assertion = expect(load).rejects.toBeInstanceOf(LoadStalledError);
        await vi.advanceTimersByTimeAsync(1000);
        await assertion;

        // A hang is a transfer problem, not a device problem, so retrying the
        // remaining plans would only burn another timeout each.
        expect(harness.attempts).toHaveLength(1);
      });

      it("still falls through the plans when a device genuinely fails", async () => {
        harness.failOn = (device) => device === "webgpu";
        const stalling = createEngine({ stallTimeoutMs: 1000 });

        await stalling.load("webgpu");

        expect(harness.attempts.map((attempt) => attempt.device)).toEqual([
          "webgpu",
          "webgpu",
          "wasm",
        ]);
      });

      it("waits indefinitely when the timeout is disabled", async () => {
        harness.hangOn = () => true;
        const stalling = createEngine({ stallTimeoutMs: 0 });
        const settled = vi.fn();
        stalling.load("wasm").then(settled, settled);

        await vi.advanceTimersByTimeAsync(10 * 60 * 1000);

        expect(settled).not.toHaveBeenCalled();
      });

      it("does not fire once the model has loaded", async () => {
        const stalling = createEngine({ stallTimeoutMs: 1000 });
        await stalling.load("wasm");

        await vi.advanceTimersByTimeAsync(10 * 60 * 1000);

        // The timer must be cleared on success; a late fire would reject a
        // promise nobody is holding any more.
        expect(stalling.loadedPlan?.device).toBe("wasm");
      });

      it("guards against a load that is silent from the very first moment", async () => {
        // No progress callback is ever invoked here, so the timer only has
        // load() itself as its starting point.
        harness.module.ChatterboxModel.from_pretrained = (async () =>
          new Promise(() => {})) as unknown as typeof harness.module.ChatterboxModel.from_pretrained;
        const stalling = createEngine({ stallTimeoutMs: 1000 });

        const load = stalling.load("wasm");
        const assertion = expect(load).rejects.toBeInstanceOf(LoadStalledError);
        await vi.advanceTimersByTimeAsync(1000);

        await assertion;
      });
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
      await expect(createEngine().embed(audio(1_000))).rejects.toBeInstanceOf(VoxShotError);
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
      ).rejects.toBeInstanceOf(VoxShotError);
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
