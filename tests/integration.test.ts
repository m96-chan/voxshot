import { describe, expect, it } from "vitest";

import type { SynthesisEngine, SynthesisRequest } from "../src/engine/types.js";
import type { PcmAudio, Platform } from "../src/platform.js";
import { exposeEngine } from "../src/worker/expose.js";
import { WorkerSynthesisEngine } from "../src/worker/worker-engine.js";
import { MemoryVoiceStore } from "../src/voice/memory-store.js";
import { VoxShot } from "../src/voxshot.js";

/** Reference audio: a 180 Hz tone with a little breathy jitter. */
function referenceAudio(sampleRate = 16_000, seconds = 1): PcmAudio {
  const samples = new Float32Array(sampleRate * seconds);
  for (let index = 0; index < samples.length; index += 1) {
    const t = index / sampleRate;
    samples[index] = 0.6 * Math.sin(2 * Math.PI * 180 * t) + 0.05 * Math.sin(2 * Math.PI * 900 * t);
  }
  return { samples, sampleRate };
}

function testPlatform(played: PcmAudio[]): Platform {
  return {
    decoder: {
      decode: async () => ({ channels: [referenceAudio().samples], sampleRate: 16_000 }),
    },
    player: {
      play: async (audio) => {
        played.push(audio);
      },
    },
    gpu: { isAvailable: async () => false },
  };
}

describe("end to end with the built in engine", () => {
  const createInstance = (played: PcmAudio[] = []) =>
    VoxShot.create({ platform: testPlatform(played), voiceStore: new MemoryVoiceStore() });

  it("clones a voice, speaks and plays the result", async () => {
    const played: PcmAudio[] = [];
    const tts = await createInstance(played);

    await tts.cloneVoice(referenceAudio());
    const audio = await tts.speak("Hello there. This is VoxShot.");

    expect(audio.duration).toBeGreaterThan(0.5);
    expect(audio.sampleRate).toBe(24_000);

    await audio.play();

    expect(played).toHaveLength(1);
    expect(played[0]?.samples).toBe(audio.samples);

    await tts.dispose();
  });

  it("produces a playable WAV file", async () => {
    const tts = await createInstance();
    await tts.cloneVoice(referenceAudio());

    const wav = await (await tts.speak("Hello.")).toWav();

    expect(wav.byteLength).toBeGreaterThan(44);
    expect(new DataView(wav).getUint32(24, true)).toBe(24_000);

    await tts.dispose();
  });

  it("speaks Japanese text", async () => {
    const tts = await createInstance();
    await tts.cloneVoice(referenceAudio());

    const audio = await tts.speak("こんにちは、世界。ゼロボックスです。");

    expect(audio.duration).toBeGreaterThan(0.5);

    await tts.dispose();
  });

  it("streams the same audio that speak returns", async () => {
    const tts = await createInstance();
    await tts.cloneVoice(referenceAudio());

    const streamed: number[] = [];
    for await (const chunk of tts.stream("Hello there. This is VoxShot.")) {
      streamed.push(...chunk.samples);
    }
    const spoken = await tts.speak("Hello there. This is VoxShot.");

    expect(streamed).toEqual(Array.from(spoken.samples));

    await tts.dispose();
  });

  it("round trips a voice through the store", async () => {
    const store = new MemoryVoiceStore();
    const platform = testPlatform([]);

    const first = await VoxShot.create({ platform, voiceStore: store });
    await first.cloneVoice(referenceAudio());
    await first.saveVoice("alice");
    const original = await first.speak("Hello.");
    await first.dispose();

    const second = await VoxShot.create({ platform, voiceStore: store });
    await second.useVoice("alice");
    const restored = await second.speak("Hello.");
    await second.dispose();

    expect(Array.from(restored.samples)).toEqual(Array.from(original.samples));
  });

  it("renders different speakers differently", async () => {
    const tts = await createInstance();

    await tts.cloneVoice(referenceAudio());
    const low = await tts.speak("Hello.");

    await tts.cloneVoice({
      samples: referenceAudio(16_000).samples.map((_, index) =>
        0.6 * Math.sin((2 * Math.PI * 320 * index) / 16_000),
      ),
      sampleRate: 16_000,
    });
    const high = await tts.speak("Hello.");

    expect(Array.from(low.samples)).not.toEqual(Array.from(high.samples));

    await tts.dispose();
  });
});

/**
 * #67, end to end: cutting an utterance used to leave a render running in the
 * worker, and the next request re-entered the same session and never came
 * back. This drives the real RPC over a MessageChannel, so the whole chain is
 * exercised — playback, cancellation, the worker queue and the reply path.
 */
describe("cutting an utterance mid-render", () => {
  /** Engine that never finishes a render until the test lets it. */
  class GatedEngine implements SynthesisEngine {
    readonly name = "gated";
    readonly sampleRate = 24_000;
    readonly started: string[] = [];
    readonly signals: (AbortSignal | undefined)[] = [];
    peakConcurrency = 0;
    #inFlight = 0;
    #release: (() => void)[] = [];

    async load(): Promise<void> {}
    async embed(): Promise<Float32Array> {
      return Float32Array.from([0.5]);
    }
    async synthesize(request: SynthesisRequest): Promise<Float32Array> {
      this.#inFlight += 1;
      this.peakConcurrency = Math.max(this.peakConcurrency, this.#inFlight);
      this.started.push(request.text);
      this.signals.push(request.signal);
      try {
        await new Promise<void>((resolve) => {
          this.#release.push(resolve);
          request.signal?.addEventListener("abort", () => resolve(), { once: true });
        });
        request.signal?.throwIfAborted();
        return Float32Array.from([0.1, 0.2, 0.3, 0.4]);
      } finally {
        this.#inFlight -= 1;
      }
    }
    releaseAll(): void {
      for (const resolve of this.#release.splice(0)) resolve();
    }
    async dispose(): Promise<void> {}
  }

  const settle = async (): Promise<void> => {
    for (let tick = 0; tick < 5; tick += 1) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  };

  it("cancels the abandoned render and answers the next utterance", async () => {
    const channel = new MessageChannel();
    const worker = new GatedEngine();
    const stop = exposeEngine(worker, channel.port2);
    const engine = new WorkerSynthesisEngine(channel.port1, { sampleRate: 24_000 });

    const written: Float32Array[] = [];
    const platform: Platform = {
      ...testPlatform([]),
      streamingPlayer: {
        open: async () => ({
          write: async (samples: Float32Array) => {
            written.push(samples);
          },
          flush: async () => 0,
          end: async () => {},
          stop: async () => {},
          setVolume: () => {},
        }),
      },
    } as unknown as Platform;

    const tts = await VoxShot.create({ platform, engine, voiceStore: new MemoryVoiceStore() });
    await tts.cloneVoice(referenceAudio());

    const speech = tts.play("First one. Second one. Third one.");
    await settle();
    expect(worker.started).toHaveLength(1);

    // Cut it while the worker is still rendering.
    await speech.stop();
    await settle();

    expect(worker.signals[0]?.aborted).toBe(true);

    // The engine has to be usable straight afterwards — this is the request
    // that used to go out and never come back.
    const next = tts.play("A brand new line.");
    await settle();
    worker.releaseAll();
    await next.done;

    expect(written.length).toBeGreaterThan(0);
    expect(worker.peakConcurrency).toBe(1);

    stop();
    engine.disconnect();
    channel.port1.close();
    channel.port2.close();
  });
});
