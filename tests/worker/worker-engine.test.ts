import { afterEach, describe, expect, it, vi } from "vitest";

import type { SynthesisEngine, SynthesisRequest } from "../../src/engine/types.js";
import { InvalidInputError, NoVoiceError, ZeroVoxError } from "../../src/errors.js";
import type { PcmAudio } from "../../src/platform.js";
import type { VoiceEmbedding } from "../../src/voice/types.js";
import { exposeEngine } from "../../src/worker/expose.js";
import { WorkerSynthesisEngine } from "../../src/worker/worker-engine.js";
import { toArray } from "../helpers/tensor.js";

class RecordingEngine implements SynthesisEngine {
  readonly name = "recording";
  readonly sampleRate = 24_000;

  loadedOn: string | undefined;
  embedded: PcmAudio[] = [];
  requests: SynthesisRequest[] = [];
  disposed = false;
  failure: unknown;

  async load(device: string): Promise<void> {
    if (this.failure) throw this.failure;
    this.loadedOn = device;
  }

  async embed(audio: PcmAudio): Promise<Float32Array> {
    if (this.failure) throw this.failure;
    this.embedded.push(audio);
    return Float32Array.from([audio.samples.length, audio.sampleRate]);
  }

  async synthesize(request: SynthesisRequest): Promise<Float32Array> {
    if (this.failure) throw this.failure;
    this.requests.push(request);
    return Float32Array.from([0.1, 0.2, 0.3]);
  }

  async dispose(): Promise<void> {
    this.disposed = true;
  }
}

interface Wired {
  engine: WorkerSynthesisEngine;
  worker: RecordingEngine;
  close: () => void;
}

const openConnections: Wired[] = [];

/** Wire a main-thread engine to a worker-side engine over a MessageChannel. */
function wire(options: Record<string, unknown> = {}): Wired {
  const channel = new MessageChannel();
  const worker = new RecordingEngine();
  const stop = exposeEngine(worker, channel.port2);
  const engine = new WorkerSynthesisEngine(channel.port1, options);

  const wired: Wired = {
    engine,
    worker,
    close: () => {
      stop();
      engine.disconnect();
      channel.port1.close();
      channel.port2.close();
    },
  };
  openConnections.push(wired);
  return wired;
}

afterEach(() => {
  while (openConnections.length > 0) {
    openConnections.pop()?.close();
  }
});

const voice: VoiceEmbedding = {
  vector: Float32Array.from([0.5]),
  sampleRate: 24_000,
  createdAt: 0,
  engine: "recording",
  tensors: { speaker_embeddings: { type: "float32", dims: [1], data: Float32Array.from([0.5]) } },
};

describe("WorkerSynthesisEngine", () => {
  it("forwards load to the worker side engine", async () => {
    const { engine, worker } = wire();

    await engine.load("webgpu");

    expect(worker.loadedOn).toBe("webgpu");
  });

  it("adopts the worker engine's name and sample rate after loading", async () => {
    const { engine } = wire();

    await engine.load("wasm");

    expect(engine.name).toBe("recording");
    expect(engine.sampleRate).toBe(24_000);
  });

  it("reports a configured sample rate before loading", () => {
    const { engine } = wire({ sampleRate: 16_000 });

    expect(engine.sampleRate).toBe(16_000);
  });

  it("forwards reference audio and returns the embedding", async () => {
    const { engine, worker } = wire();
    await engine.load("wasm");

    const vector = await engine.embed({ samples: new Float32Array(320), sampleRate: 16_000 });

    expect(worker.embedded[0]?.samples).toHaveLength(320);
    expect(worker.embedded[0]?.sampleRate).toBe(16_000);
    expect(Array.from(vector)).toEqual([320, 16_000]);
  });

  it("does not detach the caller's sample buffer", async () => {
    const { engine } = wire();
    await engine.load("wasm");
    const samples = new Float32Array([0.1, 0.2, 0.3]);

    await engine.embed({ samples, sampleRate: 16_000 });

    expect(samples).toHaveLength(3);
    expect(samples[0]).toBeCloseTo(0.1, 6);
  });

  it("forwards synthesis requests and returns samples", async () => {
    const { engine, worker } = wire();
    await engine.load("wasm");

    const samples = await engine.synthesize({ text: "こんにちは", voice, speed: 1.5 });

    expect(worker.requests[0]?.text).toBe("こんにちは");
    expect(worker.requests[0]?.speed).toBe(1.5);
    expect(Array.from(samples)).toEqual([
      expect.closeTo(0.1, 6),
      expect.closeTo(0.2, 6),
      expect.closeTo(0.3, 6),
    ]);
  });

  it("round trips voice tensors through the boundary", async () => {
    const { engine, worker } = wire();
    await engine.load("wasm");

    await engine.synthesize({ text: "hi", voice, speed: 1 });

    const received = worker.requests[0]?.voice;
    expect(received?.engine).toBe("recording");
    expect(toArray(received?.tensors?.speaker_embeddings?.data)).toEqual([
      expect.closeTo(0.5, 6),
    ]);
  });

  it("handles concurrent calls independently", async () => {
    const { engine, worker } = wire();
    await engine.load("wasm");

    const [first, second] = await Promise.all([
      engine.embed({ samples: new Float32Array(10), sampleRate: 16_000 }),
      engine.embed({ samples: new Float32Array(20), sampleRate: 8_000 }),
    ]);

    expect(Array.from(first)).toEqual([10, 16_000]);
    expect(Array.from(second)).toEqual([20, 8_000]);
    expect(worker.embedded).toHaveLength(2);
  });

  it("propagates worker side errors with their code", async () => {
    const { engine, worker } = wire();
    await engine.load("wasm");
    worker.failure = new InvalidInputError("bad reference audio");

    const failure = engine.embed({ samples: new Float32Array(4), sampleRate: 16_000 });

    await expect(failure).rejects.toBeInstanceOf(ZeroVoxError);
    await expect(failure).rejects.toMatchObject({
      code: "INVALID_INPUT",
      message: "bad reference audio",
    });
  });

  it("propagates a plain error as an unknown ZeroVoxError", async () => {
    const { engine, worker } = wire();
    await engine.load("wasm");
    worker.failure = new Error("out of memory");

    await expect(
      engine.synthesize({ text: "hi", voice, speed: 1 }),
    ).rejects.toMatchObject({ code: "UNKNOWN", message: "out of memory" });
  });

  it("forwards dispose to the worker", async () => {
    const { engine, worker } = wire();
    await engine.load("wasm");

    await engine.dispose();

    expect(worker.disposed).toBe(true);
  });

  it("forwards progress events raised inside the worker", async () => {
    const onProgress = vi.fn();
    const { engine } = wire({ onProgress });
    await engine.load("wasm");

    expect(onProgress).toHaveBeenCalledWith({ status: "ready", file: "recording" });
  });

  it("serializes a thrown non-error value", async () => {
    const { engine, worker } = wire();
    await engine.load("wasm");
    worker.failure = "just a string";

    await expect(
      engine.synthesize({ text: "hi", voice, speed: 1 }),
    ).rejects.toMatchObject({ code: "UNKNOWN", message: "just a string" });
  });

  it("ignores replies it has no pending call for", async () => {
    const channel = new MessageChannel();
    const engine = new WorkerSynthesisEngine(channel.port1);

    channel.port2.postMessage({ zerovox: 1, id: 999, ok: true, result: null });
    channel.port2.postMessage("not a protocol message");
    await new Promise((resolve) => setTimeout(resolve, 10));

    expect(engine.name).toBe("worker");
    engine.disconnect();
    channel.port1.close();
    channel.port2.close();
  });

  it("clears the reply timer on success and on disconnect", async () => {
    const { engine } = wire({ timeoutMs: 5_000 });

    await engine.load("wasm");
    const pending = engine.embed({ samples: new Float32Array(4), sampleRate: 16_000 });
    await pending;

    const stalled = engine.synthesize({ text: "hi", voice, speed: 1 });
    engine.disconnect();
    await expect(stalled).rejects.toBeInstanceOf(ZeroVoxError);
  });

  it("rejects pending calls when the connection is dropped", async () => {
    const { engine } = wire();
    await engine.load("wasm");
    const pending = engine.embed({ samples: new Float32Array(4), sampleRate: 16_000 });

    engine.disconnect();

    await expect(pending).rejects.toBeInstanceOf(ZeroVoxError);
  });

  it("refuses to send after disconnecting", async () => {
    const { engine } = wire();
    engine.disconnect();

    await expect(engine.load("wasm")).rejects.toBeInstanceOf(ZeroVoxError);
  });
});

describe("exposeEngine", () => {
  it("ignores messages that are not requests", async () => {
    const channel = new MessageChannel();
    const worker = new RecordingEngine();
    const stop = exposeEngine(worker, channel.port2);
    const responses: unknown[] = [];
    channel.port1.addEventListener("message", (event) => responses.push(event.data));
    channel.port1.start();

    channel.port1.postMessage({ hello: "world" });
    channel.port1.postMessage(null);
    await new Promise((resolve) => setTimeout(resolve, 10));

    expect(responses).toEqual([]);
    stop();
    channel.port1.close();
    channel.port2.close();
  });

  it("reports an unknown method instead of hanging", async () => {
    const channel = new MessageChannel();
    const stop = exposeEngine(new RecordingEngine(), channel.port2);
    const responses: Record<string, unknown>[] = [];
    channel.port1.addEventListener("message", (event) =>
      responses.push(event.data as Record<string, unknown>),
    );
    channel.port1.start();

    channel.port1.postMessage({ zerovox: 1, id: 7, method: "teleport" });
    await new Promise((resolve) => setTimeout(resolve, 10));

    expect(responses[0]).toMatchObject({ id: 7, ok: false });
    stop();
    channel.port1.close();
    channel.port2.close();
  });

  it("stops answering once disposed", async () => {
    const channel = new MessageChannel();
    const worker = new RecordingEngine();
    const stop = exposeEngine(worker, channel.port2);
    stop();

    const engine = new WorkerSynthesisEngine(channel.port1, { timeoutMs: 50 });
    await expect(engine.load("wasm")).rejects.toBeInstanceOf(ZeroVoxError);

    engine.disconnect();
    channel.port1.close();
    channel.port2.close();
  });
});

describe("worker errors that are not ZeroVoxError instances", () => {
  it("keeps the NoVoiceError code intact", async () => {
    const { engine, worker } = wire();
    await engine.load("wasm");
    worker.failure = new NoVoiceError();

    await expect(
      engine.synthesize({ text: "hi", voice, speed: 1 }),
    ).rejects.toMatchObject({ code: "NO_VOICE" });
  });
});
