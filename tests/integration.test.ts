import { describe, expect, it } from "vitest";

import type { PcmAudio, Platform } from "../src/platform.js";
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
