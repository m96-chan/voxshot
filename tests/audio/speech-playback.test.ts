import { describe, expect, it, vi } from "vitest";

import { startSpeechPlayback } from "../../src/audio/speech-playback.js";
import type { StreamingPlayback } from "../../src/platform.js";

/** A streaming device the test resolves manually, one write at a time. */
class FakePlayback implements StreamingPlayback {
  writes: Float32Array[] = [];
  volumes: number[] = [];
  played = 0;
  ended = false;
  stopped = false;

  #pendingWrites: (() => void)[] = [];

  write(samples: Float32Array): Promise<void> {
    this.writes.push(samples);
    return new Promise((resolve) => {
      this.#pendingWrites.push(resolve);
    });
  }

  /** Simulate the device draining below its low-water mark. */
  resolveWrite(): void {
    this.#pendingWrites.shift()?.();
  }

  get pendingWriteCount(): number {
    return this.#pendingWrites.length;
  }

  async end(): Promise<void> {
    this.ended = true;
  }

  async flush(): Promise<number> {
    while (this.#pendingWrites.length > 0) {
      this.resolveWrite();
    }
    return this.played;
  }

  async stop(): Promise<void> {
    this.stopped = true;
    while (this.#pendingWrites.length > 0) {
      this.resolveWrite();
    }
  }

  setVolume(volume: number): void {
    this.volumes.push(volume);
  }
}

const audioOf = (length: number, value: number) => {
  const samples = new Float32Array(length);
  samples.fill(value);
  return samples;
};

const flushMicrotasks = () => new Promise((resolve) => setTimeout(resolve, 0));

interface Setup {
  chunks?: string[];
  chunkLength?: number;
  volume?: number;
}

function setup(options: Setup = {}) {
  const chunks = options.chunks ?? ["one", "two", "three"];
  const chunkLength = options.chunkLength ?? 100;
  const playback = new FakePlayback();
  const synthesized: string[] = [];
  const resolvers = new Map<string, (samples: Float32Array) => void>();

  const signals = new Map<string, AbortSignal | undefined>();
  const synthesize = vi.fn(async (chunk: string, signal?: AbortSignal) => {
    synthesized.push(chunk);
    signals.set(chunk, signal);
    return new Promise<Float32Array>((resolve, reject) => {
      resolvers.set(chunk, resolve);
      signal?.addEventListener(
        "abort",
        () => reject(new Error("The request was cancelled.")),
        { once: true },
      );
    });
  });
  const resolveSynthesis = (chunk: string) => {
    const resolve = resolvers.get(chunk);
    if (!resolve) {
      throw new Error(`no pending synthesis for "${chunk}"`);
    }
    resolve(audioOf(chunkLength, chunks.indexOf(chunk) + 1));
    resolvers.delete(chunk);
  };

  const open = vi.fn(async () => playback);
  const speech = startSpeechPlayback({
    chunks,
    synthesize,
    open,
    ...(options.volume !== undefined ? { volume: options.volume } : {}),
  });
  return { chunks, playback, synthesize, synthesized, signals, resolveSynthesis, speech };
}

describe("startSpeechPlayback", () => {
  it("plays every chunk in order and ends the stream", async () => {
    const { playback, resolveSynthesis, speech } = setup();

    await flushMicrotasks();
    resolveSynthesis("one");
    await flushMicrotasks();
    resolveSynthesis("two");
    await flushMicrotasks();
    playback.resolveWrite();
    await flushMicrotasks();
    resolveSynthesis("three");
    await flushMicrotasks();
    playback.resolveWrite();
    await flushMicrotasks();
    playback.resolveWrite();
    await speech.done;

    expect(playback.writes.map((w) => w[0])).toEqual([1, 2, 3]);
    expect(playback.ended).toBe(true);
  });

  it("prefetches exactly one chunk ahead", async () => {
    const { synthesized, resolveSynthesis, speech, playback } = setup();

    await flushMicrotasks();
    expect(synthesized).toEqual(["one"]);

    resolveSynthesis("one");
    await flushMicrotasks();
    // While "one" is playing (write pending), "two" is being synthesized —
    // but "three" must not start yet.
    expect(synthesized).toEqual(["one", "two"]);

    resolveSynthesis("two");
    await flushMicrotasks();
    expect(synthesized).toEqual(["one", "two"]);

    playback.resolveWrite();
    await flushMicrotasks();
    expect(synthesized).toEqual(["one", "two", "three"]);

    resolveSynthesis("three");
    await flushMicrotasks();
    playback.resolveWrite();
    await flushMicrotasks();
    playback.resolveWrite();
    await speech.done;
  });

  it("applies the initial volume before playing", async () => {
    const { playback, resolveSynthesis, speech } = setup({ chunks: ["only"], volume: 0.25 });

    await flushMicrotasks();
    expect(playback.volumes).toEqual([0.25]);

    resolveSynthesis("only");
    await flushMicrotasks();
    playback.resolveWrite();
    await speech.done;
  });

  it("forwards later volume changes to the device", async () => {
    const { playback, resolveSynthesis, speech } = setup({ chunks: ["only"] });

    await flushMicrotasks();
    speech.setVolume(0.5);
    expect(playback.volumes).toEqual([0.5]);

    resolveSynthesis("only");
    await flushMicrotasks();
    playback.resolveWrite();
    await speech.done;
  });

  it("stop() halts the device and resolves done", async () => {
    const { playback, resolveSynthesis, speech, synthesized } = setup();

    await flushMicrotasks();
    resolveSynthesis("one");
    await flushMicrotasks();

    await speech.stop();
    await speech.done;

    expect(playback.stopped).toBe(true);
    expect(synthesized).toEqual(["one", "two"]);
    // A late synthesis result must not be written after stop.
    resolveSynthesis("two");
    await flushMicrotasks();
    expect(playback.writes).toHaveLength(1);
  });

  it("skip() flushes the device and continues with the next chunk", async () => {
    const { playback, resolveSynthesis, speech } = setup();

    await flushMicrotasks();
    resolveSynthesis("one");
    await flushMicrotasks();
    expect(playback.writes).toHaveLength(1);

    // Half of chunk "one" has played when the user skips it.
    playback.played = 50;
    await speech.skip();
    await flushMicrotasks();

    resolveSynthesis("two");
    await flushMicrotasks();
    resolveSynthesis("three");
    await flushMicrotasks();
    playback.resolveWrite();
    await flushMicrotasks();
    playback.resolveWrite();
    await speech.done;

    expect(playback.writes.map((w) => w[0])).toEqual([1, 2, 3]);
    expect(playback.ended).toBe(true);
  });

  it("skip() re-enqueues a chunk that was buffered but not yet audible", async () => {
    const { playback, resolveSynthesis, speech } = setup();

    await flushMicrotasks();
    resolveSynthesis("one");
    await flushMicrotasks();
    playback.resolveWrite();
    await flushMicrotasks();
    resolveSynthesis("two");
    await flushMicrotasks();
    // "one" and "two" are both in the device; "one" is still audible.
    expect(playback.writes.map((w) => w[0])).toEqual([1, 2]);

    playback.played = 80; // inside chunk "one"
    await speech.skip();
    await flushMicrotasks();

    // "two" was flushed with the tail of "one", so it must be written again.
    expect(playback.writes.map((w) => w[0])).toEqual([1, 2, 2]);

    resolveSynthesis("three");
    await flushMicrotasks();
    playback.resolveWrite();
    await flushMicrotasks();
    playback.resolveWrite();
    await speech.done;

    expect(playback.writes.map((w) => w[0])).toEqual([1, 2, 2, 3]);
  });

  it("skip() after everything ended is a no-op", async () => {
    const { playback, resolveSynthesis, speech } = setup({ chunks: ["only"] });

    await flushMicrotasks();
    resolveSynthesis("only");
    await flushMicrotasks();
    playback.resolveWrite();
    await speech.done;

    await speech.skip();
    expect(playback.writes).toHaveLength(1);
  });

  it("rejects done and stops the device when synthesis fails", async () => {
    const playback = new FakePlayback();
    const speech = startSpeechPlayback({
      chunks: ["boom"],
      synthesize: async () => {
        throw new Error("synthesis exploded");
      },
      open: async () => playback,
    });

    await expect(speech.done).rejects.toThrow("synthesis exploded");
    expect(playback.stopped).toBe(true);
  });

  it("rejects done when the device cannot be opened", async () => {
    const speech = startSpeechPlayback({
      chunks: ["one"],
      synthesize: async () => audioOf(10, 1),
      open: async () => {
        throw new Error("no audio output");
      },
    });

    await expect(speech.done).rejects.toThrow("no audio output");
  });
});

/**
 * #67: stopping mid-utterance used to leave the lookahead render running.
 * Serialising the worker turns that from a deadlock into a queue of work
 * nobody wants, so the render has to actually be called off.
 */
describe("cancellation on stop", () => {
  it("hands every render a signal", async () => {
    const { signals, speech } = setup();
    await flushMicrotasks();

    expect(signals.get("one")).toBeInstanceOf(AbortSignal);
    expect(signals.get("one")?.aborted).toBe(false);

    await speech.stop();
  });

  it("aborts the in-flight lookahead when stopped", async () => {
    const { signals, resolveSynthesis, speech, synthesized } = setup();
    await flushMicrotasks();
    resolveSynthesis("one");
    await flushMicrotasks();

    // "two" is now being rendered ahead of playback — exactly the render that
    // used to be abandoned and keep the engine busy.
    expect(synthesized).toEqual(["one", "two"]);
    expect(signals.get("two")?.aborted).toBe(false);

    await speech.stop();

    expect(signals.get("two")?.aborted).toBe(true);
  });

  it("still resolves cleanly when the cancelled render rejects", async () => {
    const { resolveSynthesis, speech } = setup();
    await flushMicrotasks();
    resolveSynthesis("one");
    await flushMicrotasks();

    await speech.stop();

    // A rejection we caused by stopping is not an error to report back.
    await expect(speech.done).resolves.toBeUndefined();
  });

  it("does not abort anything when playback finishes normally", async () => {
    const { signals, resolveSynthesis, playback, speech, chunks } = setup();
    for (const chunk of chunks) {
      await flushMicrotasks();
      resolveSynthesis(chunk);
      await flushMicrotasks();
      playback.resolveWrite();
    }
    await flushMicrotasks();
    await speech.done;

    for (const chunk of chunks) {
      expect(signals.get(chunk)?.aborted).toBe(false);
    }
  });
});
