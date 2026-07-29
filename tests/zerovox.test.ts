import { beforeEach, describe, expect, it, vi } from "vitest";

import { SynthesizedAudio } from "../src/audio/synthesized-audio.js";
import type { SynthesisEngine, SynthesisRequest } from "../src/engine/types.js";
import {
  DeviceUnavailableError,
  DisposedError,
  InvalidInputError,
  NoVoiceError,
  VoiceNotFoundError,
} from "../src/errors.js";
import type { AudioPlayer, DecodedAudio, PcmAudio, Platform } from "../src/platform.js";
import { MemoryVoiceStore } from "../src/voice/memory-store.js";
import type { VoiceTensor } from "../src/voice/types.js";
import { toArray } from "./helpers/tensor.js";
import { ZeroVox } from "../src/zerovox.js";

/** Engine double that records what it was asked to do. */
class FakeEngine implements SynthesisEngine {
  readonly name = "fake";
  readonly sampleRate = 16_000;

  loadedOn: string | undefined;
  disposed = false;
  embedded: PcmAudio[] = [];
  requests: SynthesisRequest[] = [];
  embedResult: { vector: Float32Array; tensors?: Record<string, VoiceTensor> } | undefined;

  async load(device: string): Promise<void> {
    this.loadedOn = device;
  }

  async embed(audio: PcmAudio): Promise<Float32Array | { vector: Float32Array }> {
    this.embedded.push(audio);
    return this.embedResult ?? Float32Array.from([audio.samples.length, audio.sampleRate]);
  }

  async synthesize(request: SynthesisRequest): Promise<Float32Array> {
    this.requests.push(request);
    // One sample per character keeps assertions about chunking readable.
    return new Float32Array([...request.text].map((_, index) => (index % 2 === 0 ? 0.5 : -0.5)));
  }

  async dispose(): Promise<void> {
    this.disposed = true;
  }
}

function speechAudio(length = 1_000): Float32Array {
  const samples = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    samples[index] = 0.5 * Math.sin(index / 4);
  }
  return samples;
}

interface Harness {
  engine: FakeEngine;
  platform: Platform;
  player: AudioPlayer & { calls: PcmAudio[] };
  decoded: { value: DecodedAudio };
  gpuAvailable: { value: boolean };
}

function createHarness(): Harness {
  const decoded = { value: { channels: [speechAudio()], sampleRate: 16_000 } as DecodedAudio };
  const gpuAvailable = { value: false };
  const player = {
    calls: [] as PcmAudio[],
    async play(audio: PcmAudio) {
      this.calls.push(audio);
    },
  };
  const platform: Platform = {
    decoder: { decode: async () => decoded.value },
    player,
    gpu: { isAvailable: async () => gpuAvailable.value },
  };
  return { engine: new FakeEngine(), platform, player, decoded, gpuAvailable };
}

describe("ZeroVox.create", () => {
  let harness: Harness;

  beforeEach(() => {
    harness = createHarness();
  });

  const create = (overrides: Record<string, unknown> = {}) =>
    ZeroVox.create({
      engine: harness.engine,
      platform: harness.platform,
      voiceStore: new MemoryVoiceStore(),
      ...overrides,
    });

  it("resolves the device and loads the engine onto it", async () => {
    const tts = await create();

    expect(tts.device).toBe("wasm");
    expect(harness.engine.loadedOn).toBe("wasm");
  });

  it("uses webgpu when it is available", async () => {
    harness.gpuAvailable.value = true;

    expect((await create()).device).toBe("webgpu");
  });

  it("propagates an unavailable device request", async () => {
    await expect(create({ device: "webgpu" })).rejects.toBeInstanceOf(DeviceUnavailableError);
  });

  it("exposes the engine sample rate", async () => {
    expect((await create()).sampleRate).toBe(16_000);
  });

  it("starts without an active voice", async () => {
    expect((await create()).currentVoice).toBeUndefined();
  });

  it("rejects a non positive chunk length", async () => {
    await expect(create({ maxChunkLength: 0 })).rejects.toBeInstanceOf(InvalidInputError);
  });

  it("merges chunks shorter than minChunkLength", async () => {
    const tts = await create({ minChunkLength: 10 });
    await tts.cloneVoice(new ArrayBuffer(16));

    await tts.speak("はい。なぜですか?それは仕様だからです。");

    expect(harness.engine.requests.map((request) => request.text)).toEqual([
      "はい。なぜですか?",
      "それは仕様だからです。",
    ]);
  });

  it("honours a custom chunk length", async () => {
    const tts = await create({ maxChunkLength: 4 });
    await tts.cloneVoice(new ArrayBuffer(16));

    await tts.speak("abcdefgh");

    expect(harness.engine.requests.map((request) => request.text)).toEqual(["abcd", "efgh"]);
  });

  it("rejects an unknown model name", async () => {
    await expect(create({ model: "nonexistent" })).rejects.toBeInstanceOf(InvalidInputError);
  });

  it("accepts the default model name", async () => {
    await expect(create({ model: "default" })).resolves.toBeInstanceOf(ZeroVox);
  });
});

describe("ZeroVox", () => {
  let harness: Harness;
  let tts: ZeroVox;
  let store: MemoryVoiceStore;

  beforeEach(async () => {
    harness = createHarness();
    store = new MemoryVoiceStore();
    tts = await ZeroVox.create({
      engine: harness.engine,
      platform: harness.platform,
      voiceStore: store,
      now: () => 1_700_000_000_000,
    });
  });

  describe("cloneVoice", () => {
    it("decodes an ArrayBuffer, embeds it and activates the voice", async () => {
      const voice = await tts.cloneVoice(new ArrayBuffer(16));

      expect(voice.vector).toHaveLength(2);
      expect(voice.createdAt).toBe(1_700_000_000_000);
      expect(tts.currentVoice).toEqual(voice);
    });

    it("accepts a Blob", async () => {
      const blob = new Blob([new Uint8Array([1, 2, 3])]);

      await expect(tts.cloneVoice(blob)).resolves.toBeDefined();
    });

    it("accepts a typed array view", async () => {
      const view = new Uint8Array([1, 2, 3, 4]);
      const decode = vi.spyOn(harness.platform.decoder, "decode");

      await tts.cloneVoice(view);

      expect(decode).toHaveBeenCalledOnce();
      expect((decode.mock.calls[0]?.[0] as ArrayBuffer).byteLength).toBe(4);
    });

    it("accepts ready made PCM without touching the decoder", async () => {
      const decode = vi.spyOn(harness.platform.decoder, "decode");

      await tts.cloneVoice({ samples: speechAudio(), sampleRate: 16_000 });

      expect(decode).not.toHaveBeenCalled();
    });

    it("down-mixes stereo reference audio to mono", async () => {
      await tts.cloneVoice(new ArrayBuffer(16));
      const monoLength = (harness.engine.embedded[0] as PcmAudio).samples.length;

      harness.decoded.value = { channels: [speechAudio(), speechAudio()], sampleRate: 16_000 };
      await tts.cloneVoice(new ArrayBuffer(16));

      expect(harness.engine.embedded[1]?.samples).toHaveLength(monoLength);
    });

    it("trims silence around the reference audio", async () => {
      const padded = new Float32Array(1_400);
      padded.set(speechAudio(), 200);
      harness.decoded.value = { channels: [padded], sampleRate: 16_000 };

      await tts.cloneVoice(new ArrayBuffer(16));

      expect(harness.engine.embedded[0]?.samples.length).toBeLessThan(1_100);
    });

    it("resamples reference audio to the engine rate", async () => {
      harness.decoded.value = { channels: [speechAudio(2_000)], sampleRate: 32_000 };

      await tts.cloneVoice(new ArrayBuffer(16));

      const embedded = harness.engine.embedded[0] as PcmAudio;
      expect(embedded.sampleRate).toBe(16_000);
      expect(embedded.samples.length).toBeCloseTo(1_000, -2);
    });

    it("rejects reference audio that contains no signal", async () => {
      harness.decoded.value = { channels: [new Float32Array(1_000)], sampleRate: 16_000 };

      await expect(tts.cloneVoice(new ArrayBuffer(16))).rejects.toBeInstanceOf(InvalidInputError);
    });

    it("rejects an unsupported source", async () => {
      await expect(tts.cloneVoice(42 as never)).rejects.toBeInstanceOf(InvalidInputError);
    });

    it("records which engine produced the voice", async () => {
      const voice = await tts.cloneVoice(new ArrayBuffer(16));

      expect(voice.engine).toBe("fake");
    });

    it("keeps engine specific tensors returned by the engine", async () => {
      harness.engine.embedResult = {
        vector: Float32Array.from([1, 2]),
        tensors: {
          speaker_embeddings: { type: "float32", dims: [1, 2], data: Float32Array.from([3, 4]) },
        },
      };

      const voice = await tts.cloneVoice(new ArrayBuffer(16));

      expect(Array.from(voice.vector)).toEqual([1, 2]);
      expect(toArray(voice.tensors?.speaker_embeddings?.data)).toEqual([3, 4]);
    });
  });

  describe("speak", () => {
    beforeEach(async () => {
      await tts.cloneVoice(new ArrayBuffer(16));
    });

    it("synthesizes normalized text with the active voice", async () => {
      const audio = await tts.speak("  Hello   world  ");

      expect(harness.engine.requests).toHaveLength(1);
      expect(harness.engine.requests[0]?.text).toBe("Hello world");
      expect(harness.engine.requests[0]?.voice).toEqual(tts.currentVoice);
      expect(audio).toBeInstanceOf(SynthesizedAudio);
      expect(audio.sampleRate).toBe(16_000);
      expect(audio.samples).toHaveLength("Hello world".length);
    });

    it("splits long text into chunks and joins the audio", async () => {
      const audio = await tts.speak("Hello there. How are you?");

      expect(harness.engine.requests.map((request) => request.text)).toEqual([
        "Hello there.",
        "How are you?",
      ]);
      expect(audio.samples).toHaveLength("Hello there.".length + "How are you?".length);
    });

    it("passes the requested speed through", async () => {
      await tts.speak("hello", { speed: 1.5 });

      expect(harness.engine.requests[0]?.speed).toBe(1.5);
    });

    it("defaults the speed to 1", async () => {
      await tts.speak("hello");

      expect(harness.engine.requests[0]?.speed).toBe(1);
    });

    it("returns audio that can be played through the platform player", async () => {
      const audio = await tts.speak("hello");

      await audio.play();

      expect(harness.player.calls).toHaveLength(1);
      expect(harness.player.calls[0]?.sampleRate).toBe(16_000);
    });

    it("rejects text that normalizes to nothing", async () => {
      await expect(tts.speak("   ")).rejects.toBeInstanceOf(InvalidInputError);
    });

    it("rejects a non string input", async () => {
      await expect(tts.speak(undefined as never)).rejects.toBeInstanceOf(InvalidInputError);
    });
  });

  it("refuses to speak before a voice exists", async () => {
    await expect(tts.speak("hello")).rejects.toBeInstanceOf(NoVoiceError);
  });

  describe("stream", () => {
    beforeEach(async () => {
      await tts.cloneVoice(new ArrayBuffer(16));
    });

    it("yields one chunk per sentence", async () => {
      const chunks: SynthesizedAudio[] = [];
      for await (const chunk of tts.stream("Hello there. How are you?")) {
        chunks.push(chunk);
      }

      expect(chunks).toHaveLength(2);
      expect(chunks[0]?.samples).toHaveLength("Hello there.".length);
      expect(chunks[1]?.samples).toHaveLength("How are you?".length);
    });

    it("synthesizes lazily, one chunk at a time", async () => {
      const iterator = tts.stream("Hello there. How are you?")[Symbol.asyncIterator]();

      await iterator.next();

      expect(harness.engine.requests).toHaveLength(1);
    });

    it("requires an active voice", async () => {
      const fresh = await ZeroVox.create({
        engine: new FakeEngine(),
        platform: harness.platform,
        voiceStore: new MemoryVoiceStore(),
      });

      await expect(fresh.stream("hello").next()).rejects.toBeInstanceOf(NoVoiceError);
    });

    it("rejects empty text", async () => {
      await expect(tts.stream("  ").next()).rejects.toBeInstanceOf(InvalidInputError);
    });
  });

  describe("voice management", () => {
    it("saves the active voice and lists it", async () => {
      await tts.cloneVoice(new ArrayBuffer(16));

      await tts.saveVoice("alice");

      expect(await tts.listVoices()).toEqual(["alice"]);
      expect(await store.load("alice")).toBeDefined();
    });

    it("refuses to save when no voice is active", async () => {
      await expect(tts.saveVoice("alice")).rejects.toBeInstanceOf(NoVoiceError);
    });

    it("activates a saved voice", async () => {
      await tts.cloneVoice(new ArrayBuffer(16));
      await tts.saveVoice("alice");
      const saved = tts.currentVoice;

      const other = await ZeroVox.create({
        engine: harness.engine,
        platform: harness.platform,
        voiceStore: store,
      });
      await other.useVoice("alice");

      expect(other.currentVoice).toEqual(saved);
    });

    it("reports a missing voice", async () => {
      await expect(tts.useVoice("nobody")).rejects.toBeInstanceOf(VoiceNotFoundError);
    });

    it("deletes a saved voice", async () => {
      await tts.cloneVoice(new ArrayBuffer(16));
      await tts.saveVoice("alice");

      await expect(tts.deleteVoice("alice")).resolves.toBe(true);
      expect(await tts.listVoices()).toEqual([]);
    });

    it("reports a delete that removed nothing", async () => {
      await expect(tts.deleteVoice("nobody")).resolves.toBe(false);
    });

    it("keeps the active voice loaded after it is deleted from the store", async () => {
      await tts.cloneVoice(new ArrayBuffer(16));
      await tts.saveVoice("alice");

      await tts.deleteVoice("alice");

      expect(tts.currentVoice).toBeDefined();
    });
  });

  describe("dispose", () => {
    it("disposes the engine", async () => {
      await tts.dispose();

      expect(harness.engine.disposed).toBe(true);
    });

    it("is idempotent", async () => {
      await tts.dispose();

      await expect(tts.dispose()).resolves.toBeUndefined();
    });

    it("rejects every operation afterwards", async () => {
      await tts.dispose();

      await expect(tts.speak("hello")).rejects.toBeInstanceOf(DisposedError);
      await expect(tts.cloneVoice(new ArrayBuffer(8))).rejects.toBeInstanceOf(DisposedError);
      await expect(tts.saveVoice("alice")).rejects.toBeInstanceOf(DisposedError);
      await expect(tts.useVoice("alice")).rejects.toBeInstanceOf(DisposedError);
      await expect(tts.deleteVoice("alice")).rejects.toBeInstanceOf(DisposedError);
      await expect(tts.listVoices()).rejects.toBeInstanceOf(DisposedError);
      await expect(tts.stream("hello").next()).rejects.toBeInstanceOf(DisposedError);
    });
  });
});

describe("ZeroVox defaults", () => {
  it("falls back to the built in engine, store and platform", async () => {
    const tts = await ZeroVox.create({ device: "wasm" });

    expect(tts.sampleRate).toBe(24_000);
    expect(await tts.listVoices()).toEqual([]);

    await tts.dispose();
  });
});
