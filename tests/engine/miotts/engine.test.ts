import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  MioTtsEngine,
  createMioTtsEngine,
  requestBrowserGpuDevice,
} from "../../../src/engine/miotts/engine.js";
import {
  MIOTTS_ENGINE_NAME,
  MIOTTS_SAMPLE_RATE,
  SPEAKER_EMBEDDING_SIZE,
} from "../../../src/engine/miotts/options.js";
import type {
  MioTtsWeightPart,
  MioTtsWeightSource,
} from "../../../src/engine/miotts/weights.js";
import { MIOCODEC_24K } from "../../../src/engine/miotts/codec/decoder.js";
import { DeviceUnavailableError, InvalidInputError } from "../../../src/errors.js";
import type { VoiceEmbedding } from "../../../src/voice/types.js";
import { createFakeGpu, installGpuGlobals, type FakeGpu } from "../../helpers/fake-gpu.js";
import { buildSyntheticQ8 } from "../../helpers/q8.js";
import { buildTinyDecoderCheckpoint } from "../../helpers/codec.js";
import { buildSafetensors } from "../../helpers/safetensors.js";

/**
 * The `SynthesisEngine` face of MioTTS.
 *
 * Two things are checked here and nowhere else. The first is the division of
 * responsibility this whole design rests on: the engine asks for bytes by part
 * name, one part at a time, and never asks for the voice encoder unless
 * someone clones a voice. The second is the gate — an environment that cannot
 * run the model finds out at construction, not part-way through a load.
 *
 * A synthetic language model and codec stand in for the real ones. That means
 * the audio these tests produce is meaningless, and they never look at it;
 * whether the engine produces MioTTS' audio is `npm run test:models`' question
 * and the demo's. What they look at is which parts were requested, when, and
 * what happens when one is missing.
 */

installGpuGlobals();

/**
 * The real vocabulary size.
 *
 * Everything else about the synthetic language model is tiny, but this cannot
 * be: `speechIndexOf` maps ids 151669 and up onto codebook indices, so a small
 * vocabulary can never produce a speech token and `synthesize` would only ever
 * reach its "no speech tokens" error. At full width the lm_head is also
 * chunked at 65,535 rows, which is the path the real model takes.
 */
const VOCAB_SIZE = 164_480;

/**
 * A codec checkpoint at the real 24 kHz rung's shape.
 *
 * Unlike the decoder's own tests, this one cannot shrink the config: the
 * engine hard-codes `MIOCODEC_24K` — correctly, since the rung is a property
 * of the model and not a caller's choice — so the checkpoint has to match it.
 * Random weights at real shapes come to 138 MB and build in ~70 ms, which is
 * affordable exactly once, so it is built here and shared.
 *
 * The FFN and upsample widths are still free: nothing reads them from a
 * config, only from the weights themselves, so they stay small.
 */
const CODEC_CHECKPOINT = buildTinyDecoderCheckpoint(MIOCODEC_24K);

/** A weight source backed by synthetic artifacts, recording what was asked for. */
function fakeSource(overrides: Partial<Record<MioTtsWeightPart, () => ArrayBuffer>> = {}) {
  const q8 = buildSyntheticQ8({ config: { vocabSize: VOCAB_SIZE } });
  const encode = (value: unknown) => new TextEncoder().encode(JSON.stringify(value)).buffer;
  const table: Record<MioTtsWeightPart, () => ArrayBuffer> = {
    tokenizer: () =>
      encode({
        added_tokens: [
          { id: 151643, content: "<|endoftext|>" },
          { id: 151644, content: "<|im_start|>" },
          { id: 151645, content: "<|im_end|>" },
        ],
        model: {
          vocab: Object.fromEntries(
            [...Array.from({ length: 0x7e - 0x21 + 1 }, (_, i) => String.fromCharCode(0x21 + i)), "Ġ", "Ċ"].map(
              (piece, index) => [piece, index],
            ),
          ),
          merges: [],
        },
      }) as ArrayBuffer,
    "lm-manifest": () => encode(q8.manifest) as ArrayBuffer,
    "lm-codes": () => q8.codes,
    "lm-scales": () => q8.scales,
    "lm-norms": () => q8.norms,
    "codec-decoder": () => CODEC_CHECKPOINT,
    // Never reached by synthesize(); a clone would need a real encoder
    // checkpoint, which is far too large to build here.
    "codec-encoder": () => buildSafetensors({}),
    ...overrides,
  };
  const asked: MioTtsWeightPart[] = [];
  const source: MioTtsWeightSource = {
    load: vi.fn(async (part: MioTtsWeightPart) => {
      asked.push(part);
      return table[part]();
    }),
  };
  return { source, asked };
}

function voice(overrides: Partial<VoiceEmbedding> = {}): VoiceEmbedding {
  return {
    vector: new Float32Array(SPEAKER_EMBEDDING_SIZE),
    sampleRate: MIOTTS_SAMPLE_RATE,
    createdAt: 0,
    engine: MIOTTS_ENGINE_NAME,
    ...overrides,
  };
}

/**
 * Logits the fake device hands back.
 *
 * Zeros would leave every id equally likely, which is fine for tests about the
 * shape of the work but leaves a decode loop drawing uniformly over 164,480
 * ids — speech tokens are 7.8% of those, so a short generation would sometimes
 * produce none and the test would fail on the draw rather than on the code.
 * This tilts the distribution towards the speech range instead. It is a
 * stand-in for a model that produces speech tokens, not a model.
 */
function tiltTowardsSpeech(backing: ArrayBuffer): void {
  const view = new Float32Array(backing);
  for (let i = 0; i < view.length; i += 1) {
    // Deterministic, and unrelated to anything the engine computes.
    view[i] = ((i * 2_654_435_761) % 1_000) / 1_000;
  }
}

let gpu: FakeGpu;

beforeEach(() => {
  gpu = createFakeGpu({ onMap: tiltTowardsSpeech });
});

const build = async (overrides: Parameters<typeof createMioTtsEngine>[0] | null = null) => {
  const { source, asked } = fakeSource();
  const engine = await createMioTtsEngine({
    weights: source,
    device: gpu.device,
    maxSeqLen: 64,
    // Short on purpose: each step samples over 164,480 logits, and the real
    // 700-token cap would make this suite seconds slower for nothing.
    maxNewTokens: 32,
    ...overrides,
  });
  return { engine, source, asked };
};

describe("metadata", () => {
  it("names itself, its rate, and its refusal to run without a GPU", async () => {
    const { engine } = await build();

    expect(engine.name).toBe(MIOTTS_ENGINE_NAME);
    expect(engine.sampleRate).toBe(24_000);
    // Not caution: the model streams ~600 MiB per token and a CPU run would
    // take minutes per sentence. VoxShot.create checks this BEFORE load(), so
    // an unsuitable machine never pays for the download.
    expect(engine.requiresGpu).toBe(true);
  });

  it("reports no device until it has loaded, and none again after disposal", async () => {
    const { engine } = await build();
    expect(engine.loadedDevice).toBeUndefined();

    await engine.load("webgpu");
    expect(engine.loadedDevice).toBe("webgpu");

    await engine.dispose();
    expect(engine.loadedDevice).toBeUndefined();
  });
});

describe("the gate", () => {
  it("refuses at construction where there is no WebGPU", async () => {
    // The point of failing here: the alternative is failing after a caller has
    // arranged 1.1 GB of weights for a machine that was never going to run it.
    const { source } = fakeSource();

    await expect(createMioTtsEngine({ weights: source })).rejects.toThrow(DeviceUnavailableError);
  });

  it("asks the weight source for nothing while refusing", async () => {
    const { source, asked } = fakeSource();

    await createMioTtsEngine({ weights: source }).catch(() => undefined);

    expect(asked).toEqual([]);
  });

  it("refuses to load onto anything but a GPU", async () => {
    const { engine } = await build();

    await expect(engine.load("wasm")).rejects.toThrow(DeviceUnavailableError);
  });

  it("rejects a weight source that is not one", async () => {
    await expect(
      createMioTtsEngine({ weights: {} as MioTtsWeightSource, device: gpu.device }),
    ).rejects.toThrow(InvalidInputError);
  });
});

describe("load", () => {
  it("asks for the language model and the codec decoder, and nothing else", async () => {
    // The heart of the lazy-parts design: a caller who synthesises with an
    // embedding they already hold must never pay for the encoder's 117 MB.
    const { engine, asked } = await build();

    await engine.load("webgpu");

    expect(new Set(asked)).toEqual(
      new Set(["tokenizer", "lm-manifest", "lm-codes", "lm-scales", "lm-norms", "codec-decoder"]),
    );
    expect(asked).not.toContain("codec-encoder");
  });

  it("asks for each part exactly once", async () => {
    const { engine, asked } = await build();

    await engine.load("webgpu");

    expect(new Set(asked).size).toBe(asked.length);
  });

  it("is idempotent", async () => {
    const { engine, asked } = await build();

    await engine.load("webgpu");
    const afterFirst = asked.length;
    await engine.load("webgpu");

    expect(asked).toHaveLength(afterFirst);
  });

  it("surfaces a weight source failure with the part that failed", async () => {
    const { source } = fakeSource({
      "lm-scales": () => {
        throw new Error("network down");
      },
    });
    const engine = await createMioTtsEngine({ weights: source, device: gpu.device, maxSeqLen: 64, maxNewTokens: 32 });

    await expect(engine.load("webgpu")).rejects.toThrow(/lm-scales/);
  });
});

describe("synthesize", () => {
  it("renders audio through the whole pipeline", async () => {
    // The audio is meaningless — the models are synthetic — so this asserts
    // that a chunk goes in and finite samples come out, not what they sound
    // like.
    const { engine } = await build();
    await engine.load("webgpu");

    const samples = await engine.synthesize({ text: "abc", voice: voice(), speed: 1 });

    expect(samples).toBeInstanceOf(Float32Array);
    expect(samples.every(Number.isFinite)).toBe(true);
  });

  it("never asks for the voice encoder", async () => {
    // Stated as its own test because it is the promise the part-by-part
    // design exists to keep, and it would be easy to break by loading
    // everything eagerly "for simplicity".
    const { engine, asked } = await build();
    await engine.load("webgpu");

    await engine.synthesize({ text: "abc", voice: voice(), speed: 1 });

    expect(asked).not.toContain("codec-encoder");
  });

  it("returns nothing for text with no speakable content", async () => {
    const { engine } = await build();
    await engine.load("webgpu");

    expect(await engine.synthesize({ text: "   ", voice: voice(), speed: 1 })).toHaveLength(0);
  });

  it("refuses a voice another engine produced", async () => {
    const { engine } = await build();
    await engine.load("webgpu");

    await expect(
      engine.synthesize({ text: "abc", voice: voice({ engine: "chatterbox" }), speed: 1 }),
    ).rejects.toThrow(/chatterbox/);
  });

  it("refuses a chunk longer than the sequence budget", async () => {
    // Names the two ways out rather than just refusing.
    const { source } = fakeSource();
    const engine = await createMioTtsEngine({ weights: source, device: gpu.device, maxSeqLen: 8, maxNewTokens: 4 });
    await engine.load("webgpu");

    await expect(
      engine.synthesize({ text: "abcdefghijklmnopqrstuvwxyz", voice: voice(), speed: 1 }),
    ).rejects.toThrow(/maxChunkLength|maxSeqLen/);
  });

  it("stops before doing any work when the request is already aborted", async () => {
    const { engine } = await build();
    await engine.load("webgpu");
    gpu.resetCounters();

    await expect(
      engine.synthesize({
        text: "abc",
        voice: voice(),
        speed: 1,
        signal: AbortSignal.abort(),
      }),
    ).rejects.toThrow();
    expect(gpu.log.submits).toBe(0);
  });

  it("stops mid-generation when the caller aborts", async () => {
    // A generation is up to 700 decode steps. A caller who has stopped
    // listening should not wait for the rest of them.
    const { engine } = await build();
    await engine.load("webgpu");
    const controller = new AbortController();

    const promise = engine.synthesize({
      text: "abc",
      voice: voice(),
      speed: 1,
      signal: controller.signal,
    });
    controller.abort();

    await expect(promise).rejects.toThrow();
  });

  it("renders the same samples for the same request", async () => {
    // VoxShot's synthesis cache returns the first render for a repeated
    // (voice, text) key, so an engine that drifted would make it a liar.
    const { engine } = await build();
    await engine.load("webgpu");

    const first = await engine.synthesize({ text: "abc", voice: voice(), speed: 1 });
    const second = await engine.synthesize({ text: "abc", voice: voice(), speed: 1 });

    expect(Array.from(second)).toEqual(Array.from(first));
  });

  it("ignores speed, as documented", async () => {
    const { engine } = await build();
    await engine.load("webgpu");

    const slow = await engine.synthesize({ text: "abc", voice: voice(), speed: 0.5 });
    const fast = await engine.synthesize({ text: "abc", voice: voice(), speed: 2 });

    expect(Array.from(fast)).toEqual(Array.from(slow));
  });

  it("refuses before it is loaded", async () => {
    const { engine } = await build();

    await expect(engine.synthesize({ text: "abc", voice: voice(), speed: 1 })).rejects.toThrow(
      /has not been loaded/,
    );
  });
});

describe("embed", () => {
  it("refuses reference audio at any other rate", async () => {
    // VoxShot resamples to `engine.sampleRate` before calling, so this only
    // fires for a direct caller — where silently encoding a 44.1 kHz clip as
    // if it were 24 kHz would produce a voice nobody could explain.
    const { engine } = await build();
    await engine.load("webgpu");

    await expect(
      engine.embed({ samples: new Float32Array(1000), sampleRate: 44_100 }),
    ).rejects.toThrow(InvalidInputError);
  });

  it("only asks for the encoder when a voice is actually cloned", async () => {
    // The encoder checkpoint here is empty, so the encode itself fails — which
    // is exactly what shows the part was requested at this point and not at
    // load time.
    const { engine, asked } = await build();
    await engine.load("webgpu");
    expect(asked).not.toContain("codec-encoder");

    await engine
      .embed({ samples: new Float32Array(24_000), sampleRate: MIOTTS_SAMPLE_RATE })
      .catch(() => undefined);

    expect(asked).toContain("codec-encoder");
  });

  it("retries the encoder download after a failure instead of caching the rejection", async () => {
    let attempts = 0;
    const { source } = fakeSource({
      "codec-encoder": () => {
        attempts += 1;
        throw new Error("network down");
      },
    });
    const engine = await createMioTtsEngine({ weights: source, device: gpu.device, maxSeqLen: 64, maxNewTokens: 32 });
    await engine.load("webgpu");
    const audio = { samples: new Float32Array(24_000), sampleRate: MIOTTS_SAMPLE_RATE };

    await engine.embed(audio).catch(() => undefined);
    await engine.embed(audio).catch(() => undefined);

    expect(attempts).toBe(2);
  });

  it("refuses before it is loaded", async () => {
    const { engine } = await build();

    await expect(
      engine.embed({ samples: new Float32Array(10), sampleRate: MIOTTS_SAMPLE_RATE }),
    ).rejects.toThrow(/has not been loaded/);
  });

  it("cuts a long reference down to the reference server's own limit", async () => {
    // WavLM's attention is O(T²); the MioTTS server trims references to 20 s
    // and this matches it. The encode itself fails here (the encoder
    // checkpoint is empty), which is fine — the trim happens before it, and
    // what a caller needs to know is that they got 20 seconds of their clip
    // and not all 25.
    const { engine } = await build();
    await engine.load("webgpu");

    await engine
      .embed({ samples: new Float32Array(25 * MIOTTS_SAMPLE_RATE), sampleRate: MIOTTS_SAMPLE_RATE })
      .catch(() => undefined);

    expect(engine.referenceWasTrimmed).toBe(true);
    expect(engine.lastReferenceSeconds).toBe(20);
  });

  it("leaves a short reference whole and says so", async () => {
    const { engine } = await build();
    await engine.load("webgpu");

    await engine
      .embed({ samples: new Float32Array(3 * MIOTTS_SAMPLE_RATE), sampleRate: MIOTTS_SAMPLE_RATE })
      .catch(() => undefined);

    expect(engine.referenceWasTrimmed).toBe(false);
    expect(engine.lastReferenceSeconds).toBe(3);
  });
});

describe("requestBrowserGpuDevice", () => {
  it("refuses where there is no navigator.gpu", async () => {
    await expect(requestBrowserGpuDevice()).rejects.toThrow(/no navigator\.gpu/);
  });

  it("refuses when the browser has WebGPU but no adapter", async () => {
    // A machine whose GPU is blocklisted, or a headless browser without a
    // software fallback. Different cause, same answer, different message.
    const navigatorStub = { gpu: { requestAdapter: async () => null } };
    vi.stubGlobal("navigator", navigatorStub);

    await expect(requestBrowserGpuDevice()).rejects.toThrow(/no WebGPU adapter/);

    vi.unstubAllGlobals();
  });

  it("asks for the adapter's own limits, not the defaults", async () => {
    // The packed embedding table alone is ~168 MiB of storage and the largest
    // lm_head chunk binding is ~67 MiB, past some adapters' defaults — so a
    // device requested at defaults fails part-way through the upload.
    const requestDevice = vi.fn(async () => ({ label: "device" }) as unknown as GPUDevice);
    vi.stubGlobal("navigator", {
      gpu: {
        requestAdapter: async () => ({
          limits: { maxStorageBufferBindingSize: 4_000, maxBufferSize: 8_000 },
          info: { vendor: "acme", architecture: "rev1", description: "" },
          requestDevice,
        }),
      },
    });

    const { adapterInfo } = await requestBrowserGpuDevice();

    expect(requestDevice).toHaveBeenCalledWith({
      requiredLimits: { maxStorageBufferBindingSize: 4_000, maxBufferSize: 8_000 },
    });
    expect(adapterInfo).toBe("acme rev1");

    vi.unstubAllGlobals();
  });

  it("falls back to a placeholder when the adapter will not identify itself", async () => {
    vi.stubGlobal("navigator", {
      gpu: {
        requestAdapter: async () => ({
          limits: { maxStorageBufferBindingSize: 1, maxBufferSize: 1 },
          requestDevice: async () => ({}) as unknown as GPUDevice,
        }),
      },
    });

    expect((await requestBrowserGpuDevice()).adapterInfo).toBe("unknown adapter");

    vi.unstubAllGlobals();
  });
});

describe("dispose", () => {
  it("releases the language model's buffers", async () => {
    const { engine } = await build();
    await engine.load("webgpu");

    await engine.dispose();

    expect(gpu.log.buffers.some((buffer) => buffer.destroyed)).toBe(true);
  });

  it("leaves a caller-supplied device alone", async () => {
    // An application keeping another model resident on the same device would
    // lose it otherwise. Ours to destroy only if ours to create.
    const destroy = vi.fn();
    const shared = { ...gpu.device, destroy } as unknown as GPUDevice;
    const { source } = fakeSource();
    const engine = await createMioTtsEngine({ weights: source, device: shared, maxSeqLen: 64, maxNewTokens: 32 });
    await engine.load("webgpu");

    await engine.dispose();

    expect(destroy).not.toHaveBeenCalled();
  });

  it("is safe to call before loading, and twice", async () => {
    const { engine } = await build();

    await engine.dispose();
    await engine.dispose();
  });
});

describe("the constructor", () => {
  it("is not the way in", async () => {
    // `new MioTtsEngine(...)` needs a device that `createMioTtsEngine` is what
    // obtains — the gate is the factory, and this is what keeps it there.
    expect(MioTtsEngine.length).toBe(2);
  });
});

describe("SPEAKER_EMBEDDING_SIZE", () => {
  it("is MioCodec's global embedding width", () => {
    expect(SPEAKER_EMBEDDING_SIZE).toBe(128);
  });
});
