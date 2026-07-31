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
  attempts: { device: string; dtype: Record<string, string>; config?: unknown }[];
  /** What the repo's `config.json` contains before the engine touches it. */
  repoConfig: Record<string, unknown>;
  configLoads: number;
  /** When true, `AutoConfig.from_pretrained` rejects. */
  configFails: boolean;
  /** When true, `AutoConfig.from_pretrained` never settles. */
  configHangs: boolean;
  encodeCalls: TensorLike[];
  generateCalls: Record<string, unknown>[];
  processorCalls: string[];
  /** Runs inside the processor call, so a test can abort mid-preprocessing. */
  onProcess: (() => Promise<void> | void) | undefined;
  disposeCalls: number;
  failOn: (device: string, modelDtype: string) => boolean;
  /** When true for a plan, `from_pretrained` returns a promise that never settles. */
  hangOn: (device: string, modelDtype: string) => boolean;
  /** Resolve a hung `from_pretrained`, as a stalled transfer that recovers. */
  releaseLoad: (() => void) | undefined;
  /** When true, `AutoProcessor.from_pretrained` throws after the model is built. */
  failProcessor: boolean;
  /** Models handed out, so leaks show up as created minus disposed. */
  modelsCreated: number;
  /** Last `progress_callback` handed to `from_pretrained`, so tests can drive it. */
  lastProgressCallback: ((progress: Record<string, unknown>) => void) | undefined;
  /** Every stopping criterion the engine constructed. */
  stoppers: { interrupted: boolean }[];
  /** Tokens the fake should claim to have produced; "cap" means it ran out. */
  tokensUsed: number | "cap" | undefined;
  waveform: Float32Array;
  progressEvents: unknown[];
}

function createModule(): ModuleHarness {
  const harness: ModuleHarness = {
    attempts: [],
    // Mirrors the real repo: model_type only, no architectures — which is
    // exactly why resolve_model_type falls back (#45).
    repoConfig: { model_type: "chatterbox", text_config: { hidden: 1 } },
    configLoads: 0,
    configFails: false,
    configHangs: false,
    encodeCalls: [],
    generateCalls: [],
    processorCalls: [],
    onProcess: undefined,
    disposeCalls: 0,
    failOn: () => false,
    hangOn: () => false,
    releaseLoad: undefined,
    failProcessor: false,
    modelsCreated: 0,
    lastProgressCallback: undefined,
    stoppers: [],
    tokensUsed: undefined,
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
        harness.attempts.push({
          device: options.device as string,
          dtype,
          config: options.config,
        });
        harness.lastProgressCallback = options.progress_callback as
          | ((progress: Record<string, unknown>) => void)
          | undefined;
        options.progress_callback?.({ status: "progress", file: modelId });
        if (harness.hangOn(options.device as string, dtype.language_model as string)) {
          // A dangling transfer: neither resolves nor rejects, unless the test
          // later lets it recover the way a stalled socket can.
          return new Promise((resolve) => {
            harness.releaseLoad = () => resolve(makeModel());
          });
        }
        if (harness.failOn(options.device as string, dtype.language_model as string)) {
          throw new Error(`no ${options.device} support`);
        }
        return makeModel();

        function makeModel() {
          harness.modelsCreated += 1;
          return {
          async encode_speech(audioValues: TensorLike) {
            harness.encodeCalls.push(audioValues);
            return speakerOutputs();
          },
          async generate(params: Record<string, unknown>) {
            harness.generateCalls.push(params);

            // Drive the stopping criteria the way StoppingCriteriaList does:
            // once per generated token, after the token would have been
            // appended. That is what the engine counts.
            const steps =
              harness.tokensUsed === "cap"
                ? (params.max_new_tokens as number)
                : (harness.tokensUsed ?? 0);
            // Called as a function, which is how StoppingCriteriaList does it.
            const criteria = params.stopping_criteria as
              | ((ids: number[][], scores: unknown) => boolean[])[]
              | undefined;
            for (let step = 0; step < steps; step += 1) {
              for (const criterion of criteria ?? []) {
                criterion([[0]], null);
              }
            }

            // Deliberately unrelated to any constant the engine uses. A fake
            // that reproduced the engine's own arithmetic could only ever
            // confirm the engine agrees with itself.
            return new FakeTensor("float32", harness.waveform, [1, harness.waveform.length]);
          },
          async dispose() {
            harness.disposeCalls += 1;
          },
          };
        }
      },
    },
    // Mirrors the library's `Callable`: the constructor returns a *closure*
    // whose prototype is the subclass, so instances are invoked as functions
    // rather than through `._call`. The engine's counter subclasses this, and
    // a base that were an ordinary class would let a counter that is not
    // callable pass here while failing against the real build.
    StoppingCriteria: class {
      constructor() {
        const closure = function (this: unknown, ...args: unknown[]): unknown {
          return (closure as unknown as { _call: (...a: unknown[]) => unknown })._call(...args);
        };
        return Object.setPrototypeOf(closure, new.target.prototype) as unknown as never;
      }
      _call(_inputIds: number[][], _scores: unknown): boolean[] {
        return [false];
      }
    },
    InterruptableStoppingCriteria: class {
      interrupted = false;
      constructor() {
        harness.stoppers.push(this);
      }
      interrupt() {
        this.interrupted = true;
      }
    },
    AutoConfig: {
      async from_pretrained() {
        harness.configLoads += 1;
        if (harness.configHangs) {
          return new Promise<never>(() => {});
        }
        if (harness.configFails) {
          throw new Error("config.json is unreachable");
        }
        return { ...harness.repoConfig };
      },
    },
    AutoProcessor: {
      async from_pretrained() {
        if (harness.failProcessor) {
          // The model is already built by now; whatever holds it has to
          // release it on this path too.
          throw new Error("preprocessor_config.json is missing");
        }
        return async (text: string) => {
          harness.processorCalls.push(text);
          await harness.onProcess?.();
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

      expect(harness.attempts[0]).toMatchObject({
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

      expect(harness.attempts[0]).toMatchObject({
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
     * #45: the repo's config.json carries `model_type: "chatterbox"` and no
     * `architectures`, but Transformers.js registers the type under the class
     * name `ChatterboxModel`. Its `resolve_model_type` only consults
     * `architectures` and `model_type`, so it fell back to an encoder-only
     * single-file layout, probed `onnx/model_quantized.onnx`, and 404'd twice.
     */
    describe("model architecture", () => {
      const configOf = (attempt: { config?: unknown }) =>
        attempt.config as { architectures?: string[]; model_type?: string } | undefined;

      it("names the architecture so the model type resolves", async () => {
        await engine.load("wasm");

        expect(configOf(harness.attempts[0]!)?.architectures).toEqual(["ChatterboxModel"]);
      });

      it("keeps the rest of the repo config intact", async () => {
        await engine.load("wasm");

        expect(configOf(harness.attempts[0]!)).toMatchObject({
          model_type: "chatterbox",
          text_config: { hidden: 1 },
        });
      });

      it("leaves an architecture the repo already declares alone", async () => {
        harness.repoConfig = { model_type: "chatterbox", architectures: ["SomethingElse"] };

        await engine.load("wasm");

        expect(configOf(harness.attempts[0]!)?.architectures).toEqual(["SomethingElse"]);
      });

      it("replaces an empty architecture list", async () => {
        harness.repoConfig = { model_type: "chatterbox", architectures: [] };

        await engine.load("wasm");

        expect(configOf(harness.attempts[0]!)?.architectures).toEqual(["ChatterboxModel"]);
      });

      it("reads the config once and reuses it across plan fallbacks", async () => {
        harness.failOn = (device) => device === "webgpu";

        await engine.load("webgpu");

        expect(harness.attempts).toHaveLength(3);
        expect(harness.configLoads).toBe(1);
      });
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

/**
 * #67: an abandoned render used to run to completion, holding the worker's
 * single execution slot. `ChatterboxModel.generate` spreads its params into
 * the base `generate`, so a stopping criterion reaches the token loop.
 */
describe("ChatterboxEngine cancellation", () => {
  let harness: ModuleHarness;
  let engine: ChatterboxEngine;

  beforeEach(async () => {
    harness = createModule();
    engine = new ChatterboxEngine({ loadModule: async () => harness.module });
    await engine.load("wasm");
    await engine.embed(audio());
  });

  const voice = (): VoiceEmbedding => ({
    vector: Float32Array.from([0.1]),
    sampleRate: CHATTERBOX_SAMPLE_RATE,
    createdAt: 0,
    engine: "chatterbox",
    tensors: {
      audio_features: { type: "float32", dims: [1], data: Float32Array.from([1]) },
      audio_tokens: { type: "int64", dims: [1], data: BigInt64Array.from([1n]) },
      speaker_embeddings: { type: "float32", dims: [1], data: Float32Array.from([1]) },
      speaker_features: { type: "float32", dims: [1], data: Float32Array.from([1]) },
    },
  });

  it("passes a stopping criterion to generate when a signal is supplied", async () => {
    const controller = new AbortController();

    await engine.synthesize({
      text: "hello",
      voice: voice(),
      speed: 1,
      signal: controller.signal,
    });

    // Criteria travel as an array now: the interrupter alongside the counter.
    expect(harness.generateCalls[0]?.stopping_criteria).toContain(harness.stoppers[0]);
  });

  it("builds no interrupter when no signal is supplied", async () => {
    await engine.synthesize({ text: "hello", voice: voice(), speed: 1 });

    // The counter is always present — truncation is reported regardless of
    // whether anyone can cancel. Only the interrupter depends on a signal.
    expect(harness.stoppers).toHaveLength(0);
  });

  it("interrupts the criterion when the caller aborts", async () => {
    const controller = new AbortController();
    const rendering = engine.synthesize({
      text: "cut me",
      voice: voice(),
      speed: 1,
      signal: controller.signal,
    });
    controller.abort();

    await expect(rendering).rejects.toThrow();
    expect(harness.stoppers[0]?.interrupted).toBe(true);
  });

  it("starts the criterion interrupted when the abort lands during preprocessing", async () => {
    // The gap between the up-front abort check and the criterion being built:
    // `processor(text)` is awaited in between, so a signal can go from live to
    // aborted while the render is still committing to happen. The criterion
    // has to notice, or nothing will stop the token loop.
    const controller = new AbortController();
    harness.onProcess = () => controller.abort();

    await expect(
      engine.synthesize({ text: "gone", voice: voice(), speed: 1, signal: controller.signal }),
    ).rejects.toThrow();
    expect(harness.stoppers[0]?.interrupted).toBe(true);
  });

  it("still renders on a build without InterruptableStoppingCriteria", async () => {
    const bare = createModule();
    delete (bare.module as { InterruptableStoppingCriteria?: unknown })
      .InterruptableStoppingCriteria;
    const plain = new ChatterboxEngine({ loadModule: async () => bare.module });
    await plain.load("wasm");
    await plain.embed(audio());

    const samples = await plain.synthesize({
      text: "no criteria here",
      voice: voice(),
      speed: 1,
      signal: new AbortController().signal,
    });

    expect(samples.length).toBeGreaterThan(0);
    expect(bare.stoppers).toHaveLength(0);
  });
});

/**
 * #69: expressiveness was fixed when the engine was constructed, so changing
 * it meant reloading a 1.5 GB model. The model accepts a value per call.
 */
describe("ChatterboxEngine expressiveness", () => {
  let harness: ModuleHarness;

  const voice = (): VoiceEmbedding => ({
    vector: Float32Array.from([0.1]),
    sampleRate: CHATTERBOX_SAMPLE_RATE,
    createdAt: 0,
    engine: "chatterbox",
    tensors: {
      audio_features: { type: "float32", dims: [1], data: Float32Array.from([1]) },
      audio_tokens: { type: "int64", dims: [1], data: BigInt64Array.from([1n]) },
      speaker_embeddings: { type: "float32", dims: [1], data: Float32Array.from([1]) },
      speaker_features: { type: "float32", dims: [1], data: Float32Array.from([1]) },
    },
  });

  const ready = async (overrides: Record<string, unknown> = {}) => {
    harness = createModule();
    const engine = new ChatterboxEngine({ loadModule: async () => harness.module, ...overrides });
    await engine.load("wasm");
    await engine.embed(audio());
    return engine;
  };

  it("uses the request's value in preference to the constructor default", async () => {
    const engine = await ready({ exaggeration: 0.5 });

    await engine.synthesize({ text: "hi", voice: voice(), speed: 1, expressiveness: 0.9 });

    expect(harness.generateCalls[0]?.exaggeration).toBe(0.9);
  });

  it("falls back to the constructor default when the request omits it", async () => {
    const engine = await ready({ exaggeration: 0.25 });

    await engine.synthesize({ text: "hi", voice: voice(), speed: 1 });

    expect(harness.generateCalls[0]?.exaggeration).toBe(0.25);
  });

  it("honours a request value of zero rather than treating it as absent", async () => {
    const engine = await ready({ exaggeration: 0.5 });

    await engine.synthesize({ text: "hi", voice: voice(), speed: 1, expressiveness: 0 });

    expect(harness.generateCalls[0]?.exaggeration).toBe(0);
  });

  it("rejects a value that is not a finite non-negative number", async () => {
    const engine = await ready();

    await expect(
      engine.synthesize({ text: "hi", voice: voice(), speed: 1, expressiveness: -1 }),
    ).rejects.toBeInstanceOf(InvalidInputError);
    await expect(
      engine.synthesize({ text: "hi", voice: voice(), speed: 1, expressiveness: Number.NaN }),
    ).rejects.toBeInstanceOf(InvalidInputError);
  });
});

/**
 * #65: the chunk budget is in characters and the generation cap is in tokens,
 * and nothing related the two. Measured on this checkpoint, a chunk needs
 * roughly 2.4 tokens per character, so the 256-token default runs out at about
 * 89 characters — well inside the 120-character chunks the splitter produces.
 * Generation then stopped at the cap and the audio simply ended early, with no
 * error and no event.
 */
describe("ChatterboxEngine generation budget", () => {
  let harness: ModuleHarness;

  const voice = (): VoiceEmbedding => ({
    vector: Float32Array.from([0.1]),
    sampleRate: CHATTERBOX_SAMPLE_RATE,
    createdAt: 0,
    engine: "chatterbox",
    tensors: {
      audio_features: { type: "float32", dims: [1], data: Float32Array.from([1]) },
      audio_tokens: { type: "int64", dims: [1], data: BigInt64Array.from([1n]) },
      speaker_embeddings: { type: "float32", dims: [1], data: Float32Array.from([1]) },
      speaker_features: { type: "float32", dims: [1], data: Float32Array.from([1]) },
    },
  });

  const ready = async (overrides: Record<string, unknown> = {}) => {
    harness = createModule();
    const engine = new ChatterboxEngine({ loadModule: async () => harness.module, ...overrides });
    await engine.load("wasm");
    await engine.embed(audio());
    return engine;
  };

  const capFor = (text: string) => harness.generateCalls[0]?.max_new_tokens as number;

  it("gives a longer chunk a larger budget", async () => {
    const engine = await ready();

    await engine.synthesize({ text: "a".repeat(30), voice: voice(), speed: 1 });
    const small = capFor("");
    harness.generateCalls.length = 0;
    await engine.synthesize({ text: "a".repeat(120), voice: voice(), speed: 1 });
    const large = harness.generateCalls[0]?.max_new_tokens as number;

    expect(large).toBeGreaterThan(small);
  });

  it("covers what a full-length chunk actually needs", async () => {
    const engine = await ready();

    // 120 characters measured at 331 tokens on this checkpoint; the budget has
    // to clear that, or the splitter's own maximum truncates by construction.
    await engine.synthesize({ text: "a".repeat(120), voice: voice(), speed: 1 });

    expect(capFor("")).toBeGreaterThan(331);
  });

  it("uses an explicitly configured cap verbatim", async () => {
    const engine = await ready({ maxNewTokens: 200 });

    await engine.synthesize({ text: "a".repeat(120), voice: voice(), speed: 1 });

    // Someone who set a cap meant it, even where the derived one would differ.
    expect(capFor("")).toBe(200);
  });

  it("reports when generation stopped because it ran out of budget", async () => {
    const events: { status: string }[] = [];
    const engine = await ready({ onProgress: (e: { status: string }) => events.push(e) });
    harness.tokensUsed = "cap";

    await engine.synthesize({ text: "a".repeat(120), voice: voice(), speed: 1 });

    const truncated = events.filter((e) => e.status === "synthesize-truncated");
    expect(truncated).toHaveLength(1);
    expect(truncated[0]).toMatchObject({ text: "a".repeat(120) });
  });

  it("says nothing when generation finished on its own", async () => {
    const events: { status: string }[] = [];
    const engine = await ready({ onProgress: (e: { status: string }) => events.push(e) });
    harness.tokensUsed = 40;

    await engine.synthesize({ text: "a".repeat(120), voice: voice(), speed: 1 });

    expect(events.filter((e) => e.status === "synthesize-truncated")).toHaveLength(0);
  });
});

/**
 * #86: the loaded state lives in four fields assigned at different moments, so
 * a load that is abandoned, duplicated, or disposed underneath can leave the
 * engine half-loaded or leave a full set of ONNX sessions with nothing left to
 * release them.
 */
describe("ChatterboxEngine load lifetime", () => {
  let harness: ModuleHarness;

  const createEngine = (overrides: Record<string, unknown> = {}) =>
    new ChatterboxEngine({ loadModule: async () => harness.module, ...overrides });

  beforeEach(() => {
    harness = createModule();
  });

  const leaked = () => harness.modelsCreated - harness.disposeCalls;

  it("releases the model when the processor fails after it is built", async () => {
    harness.failProcessor = true;
    const engine = createEngine();

    await expect(engine.load("wasm")).rejects.toBeInstanceOf(VoxShotError);

    // One model was built and the load failed. Nothing else refers to it.
    expect(harness.modelsCreated).toBeGreaterThan(0);
    expect(leaked()).toBe(0);
  });

  it("builds the model once for two concurrent loads", async () => {
    const engine = createEngine();

    await Promise.all([engine.load("wasm"), engine.load("wasm")]);

    expect(harness.modelsCreated).toBe(1);
    expect(engine.loadedPlan?.device).toBe("wasm");
  });

  it("does not come back to life when disposed during a load", async () => {
    harness.hangOn = () => true;
    const engine = createEngine({ stallTimeoutMs: 0 });

    const loading = engine.load("wasm").catch(() => undefined);
    // The load reaches `from_pretrained` only after several awaits; releasing
    // before it gets there would resolve nothing.
    await vi.waitFor(() => expect(harness.releaseLoad).toBeDefined());
    await engine.dispose();

    // The transfer completes after the owner has gone.
    harness.releaseLoad?.();
    await loading;
    await Promise.resolve();

    expect(engine.loadedPlan).toBeUndefined();
    expect(leaked()).toBe(0);
  });
});

/**
 * #86, self-review. Two ways the state can be handed to the wrong owner.
 */
describe("ChatterboxEngine load state ownership", () => {
  let harness: ModuleHarness;

  const createEngine = (overrides: Record<string, unknown> = {}) =>
    new ChatterboxEngine({ loadModule: async () => harness.module, ...overrides });

  beforeEach(() => {
    harness = createModule();
  });

  it("does not let an abandoned load reset a newer one", async () => {
    harness.hangOn = () => true;
    const engine = createEngine({ stallTimeoutMs: 0 });

    const first = engine.load("wasm").catch(() => undefined);
    await vi.waitFor(() => expect(harness.releaseLoad).toBeDefined());
    const releaseFirst = harness.releaseLoad as () => void;
    harness.releaseLoad = undefined;

    await engine.dispose();

    // A second load takes ownership.
    const second = engine.load("wasm").catch(() => undefined);
    await vi.waitFor(() => expect(harness.releaseLoad).toBeDefined());

    // The abandoned first load finishes now. Its cleanup must not hand the
    // engine back to `idle` while the second one still owns it — that would
    // let a third caller start yet another load.
    releaseFirst();
    await first;
    await Promise.resolve();

    const third = engine.load("wasm").catch(() => undefined);
    // Re-read after the wait: assigning undefined above narrows the type, and
    // the waitFor that repopulates it is invisible to the checker.
    const releaseSecond = harness.releaseLoad as (() => void) | undefined;
    releaseSecond?.();
    await Promise.all([second, third]);

    expect(harness.modelsCreated - harness.disposeCalls).toBe(1);
  });

  it("gives a concurrent caller the same failure, not a false success", async () => {
    harness.failOn = () => true;
    const engine = createEngine();

    const first = engine.load("wasm");
    const second = engine.load("wasm");

    await expect(first).rejects.toBeInstanceOf(VoxShotError);
    // Sharing the attempt must share its outcome. Resolving here would leave
    // the caller believing the engine is ready.
    await expect(second).rejects.toBeInstanceOf(VoxShotError);
  });
});

/**
 * #90: truncation used to be recovered from the waveform's length, using
 * constants fitted against one reference voice. The waveform also depends on
 * the speaker tokens, so the reading moved with the voice.
 */
describe("ChatterboxEngine truncation detection", () => {
  let harness: ModuleHarness;

  const voiceWith = (speakerTokens: number): VoiceEmbedding => ({
    vector: Float32Array.from([0.1]),
    sampleRate: CHATTERBOX_SAMPLE_RATE,
    createdAt: 0,
    engine: "chatterbox",
    tensors: {
      audio_features: { type: "float32", dims: [1], data: Float32Array.from([1]) },
      // Stands in for a reference voice of a different length: the old check
      // folded this into a constant, so a different voice moved its answer.
      audio_tokens: {
        type: "int64",
        dims: [1, speakerTokens],
        data: BigInt64Array.from(Array.from({ length: speakerTokens }, () => 1n)),
      },
      speaker_embeddings: { type: "float32", dims: [1], data: Float32Array.from([1]) },
      speaker_features: { type: "float32", dims: [1], data: Float32Array.from([1]) },
    },
  });

  const ready = async (overrides: Record<string, unknown> = {}) => {
    harness = createModule();
    const events: ChatterboxLoadEvent[] = [];
    const engine = new ChatterboxEngine({
      loadModule: async () => harness.module,
      onProgress: (event) => events.push(event),
      ...overrides,
    });
    await engine.load("wasm");
    await engine.embed(audio());
    return { engine, events };
  };

  const truncations = (events: ChatterboxLoadEvent[]) =>
    events.filter((event) => event.status === "synthesize-truncated");

  it("reports truncation the same way whatever the reference voice", async () => {
    for (const speakerTokens of [1, 400]) {
      const { engine, events } = await ready();
      harness.tokensUsed = "cap";

      await engine.synthesize({ text: "a".repeat(60), voice: voiceWith(speakerTokens), speed: 1 });

      expect(truncations(events)).toHaveLength(1);
    }
  });

  it("stays silent for a short render whatever the reference voice", async () => {
    for (const speakerTokens of [1, 400]) {
      const { engine, events } = await ready();
      harness.tokensUsed = 5;

      await engine.synthesize({ text: "a".repeat(60), voice: voiceWith(speakerTokens), speed: 1 });

      expect(truncations(events)).toHaveLength(0);
    }
  });

  it("carries the chunk and the budget it ran out of", async () => {
    const { engine, events } = await ready({ maxNewTokens: 40 });
    harness.tokensUsed = "cap";

    await engine.synthesize({ text: "too long", voice: voiceWith(1), speed: 1 });

    expect(truncations(events)[0]).toMatchObject({ text: "too long", tokens: 40 });
  });

  it("says nothing on a build without StoppingCriteria", async () => {
    const bare = createModule();
    delete (bare.module as { StoppingCriteria?: unknown }).StoppingCriteria;
    const events: ChatterboxLoadEvent[] = [];
    const engine = new ChatterboxEngine({
      loadModule: async () => bare.module,
      onProgress: (event) => events.push(event),
    });
    await engine.load("wasm");
    await engine.embed(audio());
    bare.tokensUsed = "cap";

    const samples = await engine.synthesize({ text: "hi", voice: voiceWith(1), speed: 1 });

    // Degrades to silence rather than to a number it cannot justify.
    expect(samples.length).toBeGreaterThan(0);
    expect(truncations(events)).toHaveLength(0);
  });
});

describe("ChatterboxEngine option validation", () => {
  const build = (options: Record<string, unknown>) => () =>
    new ChatterboxEngine({ loadModule: async () => createModule().module, ...options });

  it.each([
    ["under a millisecond, which setTimeout rounds up and fires at once", 0.4],
    ["negative", -1],
    ["not a number at all", Number.NaN],
    // Every comparison against a non-number is false, so one slips past a
    // check written as a range and reaches setTimeout, which reads it as NaN
    // and fires in 1 ms. The same shape of failure as Infinity, from JavaScript
    // rather than from the timer.
    ["not a number even in type", {} as unknown as number],
    ["a string that looks like one", "500" as unknown as number],
  ])("refuses a stall timeout that is %s", (_why, stallTimeoutMs) => {
    expect(build({ stallTimeoutMs })).toThrow(InvalidInputError);
  });

  it("waits forever when the stall timeout is Infinity", async () => {
    // The intuitive spelling of "never give up". It used to arm the guard and
    // reject in about a millisecond, because setTimeout coerces Infinity to 1.
    const harness = createModule();
    harness.hangOn = () => true;
    const engine = new ChatterboxEngine({
      loadModule: async () => harness.module,
      stallTimeoutMs: Number.POSITIVE_INFINITY,
    });

    const settled = vi.fn();
    void engine.load("wasm").then(settled, settled);
    await new Promise((resolve) => setTimeout(resolve, 20));

    expect(settled).not.toHaveBeenCalled();
  });

  it("reads null as unset, the way ?? did before these checks existed", () => {
    // Not reachable from TypeScript, but `?? DEFAULT` treated null and
    // undefined alike and a config deserialised from JSON says null for
    // "unset". Rejecting it would be a change nobody asked for.
    expect(
      build({ stallTimeoutMs: null as unknown as number, maxNewTokens: null as unknown as number }),
    ).not.toThrow();
  });

  it.each([
    ["zero, which reaches generate as a cap of nothing", 0],
    ["fractional", 12.5],
    ["negative", -8],
  ])("refuses a token cap that is %s", (_why, maxNewTokens) => {
    expect(build({ maxNewTokens })).toThrow(InvalidInputError);
  });
});

describe("ChatterboxEngine budget ceiling", () => {
  let harness: ModuleHarness;

  const voice = (): VoiceEmbedding => ({
    vector: Float32Array.from([0.1]),
    sampleRate: CHATTERBOX_SAMPLE_RATE,
    createdAt: 0,
    engine: "chatterbox",
    tensors: {
      audio_features: { type: "float32", dims: [1], data: Float32Array.from([1]) },
      audio_tokens: { type: "int64", dims: [1], data: BigInt64Array.from([1n]) },
      speaker_embeddings: { type: "float32", dims: [1], data: Float32Array.from([1]) },
      speaker_features: { type: "float32", dims: [1], data: Float32Array.from([1]) },
    },
  });

  const ready = async (overrides: Record<string, unknown> = {}) => {
    harness = createModule();
    const engine = new ChatterboxEngine({ loadModule: async () => harness.module, ...overrides });
    await engine.load("wasm");
    await engine.embed(audio());
    return engine;
  };

  it("caps the budget it derives, however long the chunk", async () => {
    // `synthesize` is public and takes any length, and maxChunkLength is the
    // caller's to set, so the derived budget has to stop somewhere: minutes of
    // audio in one call is not a request anyone meant to make.
    const engine = await ready();

    await engine.synthesize({ text: "a".repeat(20_000), voice: voice(), speed: 1 });

    const budget = harness.generateCalls[0]?.max_new_tokens as number;
    expect(budget).toBeLessThan(2_000);
  });

  it("still honours a cap the caller named, above the ceiling or not", async () => {
    // Documented as honoured as written. Someone who names a number means it.
    const engine = await ready({ maxNewTokens: 8_000 });

    await engine.synthesize({ text: "hi", voice: voice(), speed: 1 });

    expect(harness.generateCalls[0]?.max_new_tokens).toBe(8_000);
  });
});

describe("ChatterboxEngine config resolution", () => {
  // Naming the architecture is what silences the two 404s and repairs the
  // download denominator (#45). None of that is worth failing a load over: an
  // engine that cannot read config.json still loads, exactly as it did before
  // #45 existed. So absent, failing and hung are all the same answer.

  it("loads on a build that does not export AutoConfig", async () => {
    const bare = createModule();
    delete (bare.module as { AutoConfig?: unknown }).AutoConfig;
    const engine = new ChatterboxEngine({ loadModule: async () => bare.module });

    await engine.load("wasm");

    expect(engine.loadedPlan).toBeDefined();
    expect(bare.attempts[0]?.config).toBeUndefined();
  });

  it("loads when the config cannot be read", async () => {
    const harness = createModule();
    harness.configFails = true;
    const engine = new ChatterboxEngine({ loadModule: async () => harness.module });

    await engine.load("wasm");

    expect(engine.loadedPlan).toBeDefined();
    expect(harness.attempts[0]?.config).toBeUndefined();
  });

  it("gives up on a config that never arrives instead of waiting forever", async () => {
    // The first network fetch of the load, and it used to sit outside the
    // stall guard entirely: a hung config.json left load() pending with no
    // timer to end it.
    const harness = createModule();
    harness.configHangs = true;
    const engine = new ChatterboxEngine({
      loadModule: async () => harness.module,
      stallTimeoutMs: 20,
    });

    await engine.load("wasm");

    expect(engine.loadedPlan).toBeDefined();
    expect(harness.attempts[0]?.config).toBeUndefined();
  });

  it("still names the architecture when the config is readable", async () => {
    const harness = createModule();
    const engine = new ChatterboxEngine({ loadModule: async () => harness.module });

    await engine.load("wasm");

    expect((harness.attempts[0]?.config as { architectures?: string[] })?.architectures).toEqual([
      "ChatterboxModel",
    ]);
  });
});

describe("ChatterboxEngine stall guard aftermath", () => {
  it("goes quiet once it has declared the load stalled", async () => {
    // The abandoned transfer keeps running and keeps emitting. A consumer that
    // has already handled LoadStalledError and moved on should not start
    // receiving download progress again.
    const harness = createModule();
    harness.hangOn = () => true;
    const events: ChatterboxLoadEvent[] = [];
    const engine = new ChatterboxEngine({
      loadModule: async () => harness.module,
      stallTimeoutMs: 20,
      onProgress: (event) => events.push(event),
    });

    await expect(engine.load("wasm")).rejects.toBeInstanceOf(LoadStalledError);
    const before = events.length;
    harness.lastProgressCallback?.({ status: "progress", file: "model.onnx", progress: 40 });

    expect(events).toHaveLength(before);
  });

  it("leaves no timer behind when the operation throws on the spot", async () => {
    // `#resolveConfig` hands the guard a plain arrow, so a module whose
    // from_pretrained throws synchronously never reaches the guard's own
    // try/finally. The timer it armed would outlive the call.
    vi.useFakeTimers();
    try {
      const harness = createModule();
      (harness.module as unknown as { AutoConfig: unknown }).AutoConfig = {
        from_pretrained: () => {
          throw new Error("thrown before any promise exists");
        },
      };
      const engine = new ChatterboxEngine({ loadModule: async () => harness.module });

      await engine.load("wasm");

      expect(vi.getTimerCount()).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("ChatterboxEngine per-render cleanup", () => {
  const voice = (): VoiceEmbedding => ({
    vector: Float32Array.from([0.1]),
    sampleRate: CHATTERBOX_SAMPLE_RATE,
    createdAt: 0,
    engine: "chatterbox",
    tensors: {
      audio_features: { type: "float32", dims: [1], data: Float32Array.from([1]) },
      audio_tokens: { type: "int64", dims: [1], data: BigInt64Array.from([1n]) },
      speaker_embeddings: { type: "float32", dims: [1], data: Float32Array.from([1]) },
      speaker_features: { type: "float32", dims: [1], data: Float32Array.from([1]) },
    },
  });

  const ready = async () => {
    const harness = createModule();
    const engine = new ChatterboxEngine({ loadModule: async () => harness.module });
    await engine.load("wasm");
    await engine.embed(audio());
    return { engine, harness };
  };

  it("lets go of the signal after every render, not just the aborted one", async () => {
    // SpeechPlayback uses one controller for a whole utterance, so a 100-chunk
    // render put 100 listeners on one signal, each holding a stopping
    // criterion that will never be asked to stop anything.
    const { engine } = await ready();
    const controller = new AbortController();
    const live = new Set<unknown>();
    const signal = controller.signal;
    const add = signal.addEventListener.bind(signal);
    const remove = signal.removeEventListener.bind(signal);
    signal.addEventListener = ((type: string, fn: never, options?: never) => {
      live.add(fn);
      add(type, fn, options);
    }) as unknown as typeof signal.addEventListener;
    signal.removeEventListener = ((type: string, fn: never, options?: never) => {
      live.delete(fn);
      remove(type, fn, options);
    }) as unknown as typeof signal.removeEventListener;

    for (let chunk = 0; chunk < 5; chunk += 1) {
      await engine.synthesize({ text: `chunk ${chunk}`, voice: voice(), speed: 1, signal });
    }

    expect(live.size).toBe(0);
  });

  it("does no work at all for a request that is already aborted", async () => {
    // The check used to run only after `generate`, so an abandoned request
    // still paid for the processor call, the tensors, a forward pass and the
    // whole vocoder pass before anyone looked at the signal.
    const { engine, harness } = await ready();
    const controller = new AbortController();
    controller.abort();

    await expect(
      engine.synthesize({ text: "never mind", voice: voice(), speed: 1, signal: controller.signal }),
    ).rejects.toThrow();

    expect(harness.processorCalls).toHaveLength(0);
    expect(harness.generateCalls).toHaveLength(0);
  });

  it("still answers empty text before it looks at the signal", async () => {
    // Ordering: nothing was going to be rendered anyway, so an aborted caller
    // gets the same empty answer as anyone else rather than an abort.
    const { engine } = await ready();
    const controller = new AbortController();
    controller.abort();

    const samples = await engine.synthesize({
      text: "   ",
      voice: voice(),
      speed: 1,
      signal: controller.signal,
    });

    expect(samples).toHaveLength(0);
  });
});

describe("ChatterboxEngine requiresGpu", () => {
  it("fails rather than finishing on wasm when every webgpu plan fails", async () => {
    const harness = createModule();
    harness.failOn = (device) => device === "webgpu";
    const engine = new ChatterboxEngine({
      loadModule: async () => harness.module,
      requiresGpu: true,
    });

    await expect(engine.load("webgpu")).rejects.toBeInstanceOf(VoxShotError);
    expect(harness.attempts.map((a) => a.device)).not.toContain("wasm");
  });

  it("still finishes on wasm when the engine did not ask", async () => {
    const harness = createModule();
    harness.failOn = (device) => device === "webgpu";
    const engine = new ChatterboxEngine({ loadModule: async () => harness.module });

    await engine.load("webgpu");

    expect(engine.loadedPlan?.device).toBe("wasm");
  });

  it("reports the requirement so a caller can check before loading", () => {
    expect(new ChatterboxEngine({ requiresGpu: true }).requiresGpu).toBe(true);
    expect(new ChatterboxEngine({}).requiresGpu).toBe(false);
  });
});

describe("ChatterboxEngine loadedDevice", () => {
  it("reports the device the winning plan actually used", async () => {
    const harness = createModule();
    harness.failOn = (device) => device === "webgpu";
    const engine = new ChatterboxEngine({ loadModule: async () => harness.module });

    await engine.load("webgpu");

    // Asked for webgpu, every webgpu plan failed, landed on wasm. Anything
    // that says "webgpu" here is describing a load that did not happen.
    expect(engine.loadedDevice).toBe("wasm");
  });

  it("says nothing before a load and after a dispose", async () => {
    const harness = createModule();
    const engine = new ChatterboxEngine({ loadModule: async () => harness.module });

    expect(engine.loadedDevice).toBeUndefined();
    await engine.load("wasm");
    expect(engine.loadedDevice).toBe("wasm");
    await engine.dispose();
    expect(engine.loadedDevice).toBeUndefined();
  });
});

describe("ChatterboxEngine compile milestone", () => {
  const compiling = (events: ChatterboxLoadEvent[]) =>
    events.filter((e) => (e as { status: string }).status === "load-compiling");

  /**
   * Drives progress while the load is still in flight. It has to be in flight:
   * once the attempt settles the engine deliberately stops listening (#81), so
   * events pushed afterwards are ignored — correctly, but invisibly.
   */
  const during = async (drive: (emit: (p: Record<string, unknown>) => void) => void) => {
    const harness = createModule();
    harness.hangOn = () => true;
    const events: ChatterboxLoadEvent[] = [];
    const engine = new ChatterboxEngine({
      loadModule: async () => harness.module,
      onProgress: (event) => events.push(event),
    });

    const loading = engine.load("wasm");
    // Wait for the transfer to actually be in flight; the engine imports the
    // module and resolves the config before it gets there.
    for (let tick = 0; tick < 50 && harness.releaseLoad === undefined; tick += 1) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    drive((p) => harness.lastProgressCallback?.(p));
    const mid = compiling(events).length;
    harness.releaseLoad?.();
    await loading;
    return { events, mid };
  };

  /** Aggregate as upstream emits it once the expected-file list resolved. */
  const seeded = (loaded: number, total: number) => ({
    status: "progress_total",
    loaded,
    total,
    files: { "config.json": {}, "onnx/language_model_q4.onnx": {}, "onnx/embed_tokens.onnx": {} },
  });

  it("announces compiling when the aggregate download completes", async () => {
    // Pre-populated from the expected-file list: the total is known before any
    // byte arrives and never moves.
    const { mid } = await during((emit) => {
      emit(seeded(0, 1000));
      emit(seeded(600, 1000));
      emit(seeded(1000, 1000));
    });

    expect(mid).toBe(1);
  });

  it("says nothing until the aggregate is actually complete", async () => {
    const { mid } = await during((emit) => {
      emit(seeded(0, 1000));
      emit(seeded(999, 1000));
    });

    expect(mid).toBe(0);
  });

  it("announces it once, not on every further event", async () => {
    const { mid } = await during((emit) => {
      emit(seeded(0, 1000));
      emit(seeded(1000, 1000));
      emit(seeded(1000, 1000));
    });

    expect(mid).toBe(1);
  });

  it("stays silent when the aggregate only covers the file in flight", async () => {
    // What an unresolved expected-file list produces: the aggregate is
    // accumulated from files as they start, so it hits 100% the moment the
    // first one finishes, with everything else unstarted. That is the guess
    // this milestone was withheld to avoid — and the map has grown by the time
    // it could be detected after the fact, so it is judged on the first event.
    const { mid } = await during((emit) => {
      emit({ status: "progress_total", loaded: 0, total: 400, files: { "a.onnx": {} } });
      emit({ status: "progress_total", loaded: 400, total: 400, files: { "a.onnx": {} } });
      emit({ status: "progress_total", loaded: 400, total: 900, files: { "a.onnx": {}, "b.onnx": {} } });
      emit({ status: "progress_total", loaded: 900, total: 900, files: { "a.onnx": {}, "b.onnx": {} } });
    });

    expect(mid).toBe(0);
  });

  it("stays silent when no aggregate arrives at all", async () => {
    const { mid } = await during((emit) => {
      emit({ status: "progress", file: "a.onnx", loaded: 10, total: 10 });
    });

    expect(mid).toBe(0);
  });
});
