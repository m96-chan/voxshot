import { afterEach, describe, expect, it, vi } from "vitest";

import type { SynthesisEngine, SynthesisRequest } from "../../src/engine/types.js";
import {
  InvalidInputError,
  LoadStalledError,
  NoVoiceError,
  VoxShotError,
} from "../../src/errors.js";
import type { PcmAudio } from "../../src/platform.js";
import type { VoiceEmbedding } from "../../src/voice/types.js";
import { exposeEngine } from "../../src/worker/expose.js";
import type { RpcEndpoint } from "../../src/worker/protocol.js";
import { PROTOCOL_VERSION } from "../../src/worker/protocol.js";
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

  /** Set to hold every synthesize open until the test releases it. */
  gate: Promise<void> | undefined;
  /** When true the fake ignores its abort signal, like an engine that cannot
   *  interrupt work already started. */
  stubborn = false;
  /** Highest number of synthesize calls that were ever in flight at once. */
  peakConcurrency = 0;
  #inFlight = 0;
  /** Signals seen by synthesize, in call order. */
  signals: (AbortSignal | undefined)[] = [];

  async synthesize(request: SynthesisRequest): Promise<Float32Array> {
    if (this.failure) throw this.failure;
    this.#inFlight += 1;
    this.peakConcurrency = Math.max(this.peakConcurrency, this.#inFlight);
    this.requests.push(request);
    this.signals.push(request.signal);
    try {
      if (this.gate) {
        // A well-behaved engine stops when told to; one that ignores the
        // signal cannot have its slot freed by anyone, which is a property of
        // the engine rather than of the RPC.
        if (this.stubborn) {
          await this.gate;
        } else {
          await Promise.race([
            this.gate,
            new Promise<void>((resolve) => {
              request.signal?.addEventListener("abort", () => resolve(), { once: true });
            }),
          ]);
          request.signal?.throwIfAborted();
        }
      }
      return Float32Array.from([0.1, 0.2, 0.3]);
    } finally {
      this.#inFlight -= 1;
    }
  }

  /** How many synthesize calls were running when dispose was invoked. */
  peakConcurrencyAtDispose = -1;

  async dispose(): Promise<void> {
    this.peakConcurrencyAtDispose = this.#inFlight;
    this.disposed = true;
  }
}

interface Wired {
  engine: WorkerSynthesisEngine;
  worker: RecordingEngine;
  /** The main-thread end of the channel, for posting raw protocol messages. */
  port: MessagePort;
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
    port: channel.port1,
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

    await expect(failure).rejects.toBeInstanceOf(VoxShotError);
    await expect(failure).rejects.toMatchObject({
      code: "INVALID_INPUT",
      message: "bad reference audio",
    });
  });

  /**
   * The path consumers actually use (#43): the stall guard lives in the
   * engine inside the worker, so its failure has to survive serialization
   * for a caller to branch on it.
   */
  it("carries a stalled load across the worker boundary as LOAD_STALLED", async () => {
    const { engine, worker } = wire();
    worker.failure = new LoadStalledError(300_000);

    const failure = engine.load("webgpu");

    await expect(failure).rejects.toBeInstanceOf(VoxShotError);
    await expect(failure).rejects.toMatchObject({
      code: "LOAD_STALLED",
      name: "LoadStalledError",
    });
  });

  it("propagates a plain error as an unknown VoxShotError", async () => {
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

    channel.port2.postMessage({ voxshot: 1, id: 999, ok: true, result: null });
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
    await expect(stalled).rejects.toBeInstanceOf(VoxShotError);
  });

  it("rejects pending calls when the connection is dropped", async () => {
    const { engine } = wire();
    await engine.load("wasm");
    const pending = engine.embed({ samples: new Float32Array(4), sampleRate: 16_000 });

    engine.disconnect();

    await expect(pending).rejects.toBeInstanceOf(VoxShotError);
  });

  it("refuses to send after disconnecting", async () => {
    const { engine } = wire();
    engine.disconnect();

    await expect(engine.load("wasm")).rejects.toBeInstanceOf(VoxShotError);
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

    channel.port1.postMessage({ voxshot: 1, id: 7, method: "teleport" });
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
    await expect(engine.load("wasm")).rejects.toBeInstanceOf(VoxShotError);

    engine.disconnect();
    channel.port1.close();
    channel.port2.close();
  });
});

describe("worker errors that are not VoxShotError instances", () => {
  it("keeps the NoVoiceError code intact", async () => {
    const { engine, worker } = wire();
    await engine.load("wasm");
    worker.failure = new NoVoiceError();

    await expect(
      engine.synthesize({ text: "hi", voice, speed: 1 }),
    ).rejects.toMatchObject({ code: "NO_VOICE" });
  });
});

/**
 * #67: a cut utterance left a synthesize running in the worker, and the next
 * request re-entered the same ONNX session concurrently, wedging the engine.
 * The RPC layer must never hand the engine two overlapping calls, and must
 * offer a way to abandon work that is already under way.
 */
describe("serialized execution", () => {
  const release = () => {
    let open!: () => void;
    const gate = new Promise<void>((resolve) => {
      open = resolve;
    });
    return { gate, open };
  };

  /**
   * Port messages are delivered as macrotasks, so a microtask tick is not
   * enough to let queued requests reach the worker — awaiting one would open
   * the gate before either call had started, and concurrency would look fine
   * whether or not it was.
   */
  const deliver = async (): Promise<void> => {
    for (let tick = 0; tick < 3; tick += 1) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  };

  it("never runs two engine calls at once", async () => {
    const { engine, worker } = wire();
    await engine.load("wasm");
    const { gate, open } = release();
    worker.gate = gate;

    const both = Promise.all([
      engine.synthesize({ text: "one", voice, speed: 1 }),
      engine.synthesize({ text: "two", voice, speed: 1 }),
    ]);
    await deliver();

    // Both requests have reached the worker by now. Only one may be running.
    expect(worker.peakConcurrency).toBe(1);
    expect(worker.requests.map((r) => r.text)).toEqual(["one"]);

    open();
    await both;
    expect(worker.peakConcurrency).toBe(1);
    expect(worker.requests.map((r) => r.text)).toEqual(["one", "two"]);
  });

  it("keeps the queue in the order requests arrived", async () => {
    const { engine, worker } = wire();
    await engine.load("wasm");
    const { gate, open } = release();
    worker.gate = gate;

    const all = Promise.all(
      ["a", "b", "c"].map((text) => engine.synthesize({ text, voice, speed: 1 })),
    );
    await deliver();
    open();
    await all;

    expect(worker.requests.map((r) => r.text)).toEqual(["a", "b", "c"]);
  });

  it("hands the running call a signal that aborts when the caller does", async () => {
    const { engine, worker } = wire();
    await engine.load("wasm");
    const { gate, open } = release();
    worker.gate = gate;
    const controller = new AbortController();

    const call = engine.synthesize({
      text: "cut me",
      voice,
      speed: 1,
      signal: controller.signal,
    });
    // Attach the handler before aborting: abort rejects synchronously, and a
    // rejection left unobserved across a turn of the microtask queue is
    // reported as unhandled even though the test does await it later.
    const rejected = expect(call).rejects.toBeInstanceOf(VoxShotError);
    await deliver();
    controller.abort();
    await vi.waitFor(() => expect(worker.signals[0]?.aborted).toBe(true));

    open();
    await rejected;
  });

  it("drops a queued request without ever starting it", async () => {
    const { engine, worker } = wire();
    await engine.load("wasm");
    const { gate, open } = release();
    worker.gate = gate;
    const controller = new AbortController();

    const first = engine.synthesize({ text: "running", voice, speed: 1 });
    const second = engine.synthesize({
      text: "queued",
      voice,
      speed: 1,
      signal: controller.signal,
    });
    const rejected = expect(second).rejects.toBeInstanceOf(VoxShotError);
    await deliver();
    controller.abort();
    await rejected;
    // The caller stops waiting immediately, but the worker only learns of the
    // cancellation when the control message lands.
    await deliver();

    open();
    await first;

    // The cancelled one must never reach the engine at all.
    expect(worker.requests.map((r) => r.text)).toEqual(["running"]);
  });

  it("reaches dispose past a render that never finishes", async () => {
    const { engine, worker } = wire();
    await engine.load("wasm");
    // Never opened: this is the wedged render from the report, not a slow one.
    worker.gate = new Promise<void>(() => {});

    const wedged = engine.synthesize({ text: "never ends", voice, speed: 1 });
    const abandoned = wedged.catch(() => undefined);
    await deliver();

    // Teardown has to be reachable even when the thing blocking the queue
    // will never complete on its own.
    await expect(engine.dispose()).resolves.toBeUndefined();
    expect(worker.disposed).toBe(true);

    engine.disconnect();
    await abandoned;
  });

  it("ignores a cancel for a request it no longer knows about", async () => {
    const { engine, worker, port } = wire();
    await engine.load("wasm");

    // A cancel racing its own reply arrives after the job is gone. The worker
    // answers every control message, so a missing reply is the signal that
    // handling it threw instead of treating the unknown id as a no-op.
    const replies: number[] = [];
    port.addEventListener("message", (event: MessageEvent) => {
      const data = event.data as { id?: number };
      if (typeof data?.id === "number") replies.push(data.id);
    });

    port.postMessage({
      voxshot: PROTOCOL_VERSION,
      id: 9999,
      method: "cancel",
      target: 4242,
    });
    await deliver();

    expect(replies).toContain(9999);

    const samples = await engine.synthesize({ text: "still fine", voice, speed: 1 });
    expect(Array.from(samples)).toHaveLength(3);
    expect(worker.requests.map((r) => r.text)).toEqual(["still fine"]);
  });

  it("lets dispose through while a synthesize is still running", async () => {
    const { engine, worker } = wire();
    await engine.load("wasm");
    const { gate, open } = release();
    worker.gate = gate;

    const rendering = engine.synthesize({ text: "long", voice, speed: 1 });
    await deliver();

    // Serialising must not make teardown unreachable: a queued dispose behind
    // a wedged synthesize is exactly how #67 left consumers with no way out.
    await expect(engine.dispose()).resolves.toBeUndefined();
    expect(worker.disposed).toBe(true);

    open();
    await rendering.catch(() => undefined);
  });
});

/**
 * #80: serialization closed the wedge in #67 but opened three more routes to
 * the same outcome — a request the caller waits on forever. Each of these was
 * reproduced against the real RPC before being written.
 */
describe("lifecycle after teardown and failure", () => {
  const deliver = async (): Promise<void> => {
    for (let tick = 0; tick < 4; tick += 1) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  };

  it("answers requests that arrive after dispose", async () => {
    const { engine, worker } = wire();
    await engine.load("wasm");
    worker.gate = new Promise<void>(() => {}); // a render that never settles

    const wedged = engine.synthesize({ text: "wedged", voice, speed: 1 });
    const abandoned = wedged.catch(() => undefined);
    await deliver();
    await engine.dispose();

    // Before the fix this stayed pending forever: dispose left the drain loop
    // parked, so nothing ever entered it again.
    await expect(
      engine.embed({ samples: new Float32Array(4), sampleRate: 16_000 }),
    ).rejects.toBeInstanceOf(VoxShotError);

    engine.disconnect();
    await abandoned;
  });

  it("answers after dispose even when the running call cannot be interrupted", async () => {
    const { engine, worker } = wire();
    await engine.load("wasm");
    worker.stubborn = true;
    worker.gate = new Promise<void>(() => {});

    const wedged = engine.synthesize({ text: "wedged", voice, speed: 1 });
    const abandoned = wedged.catch(() => undefined);
    await deliver();
    await engine.dispose();

    // The drain loop is still parked on a render that will never end. Nothing
    // may be allowed to queue behind it silently.
    await expect(
      engine.embed({ samples: new Float32Array(4), sampleRate: 16_000 }),
    ).rejects.toBeInstanceOf(VoxShotError);

    engine.disconnect();
    await abandoned;
  });

  it("does not dispose the engine while a call is still running", async () => {
    const { engine, worker } = wire();
    await engine.load("wasm");
    worker.gate = new Promise<void>(() => {});

    const wedged = engine.synthesize({ text: "wedged", voice, speed: 1 });
    const abandoned = wedged.catch(() => undefined);
    await deliver();
    await engine.dispose();

    // The queue exists to keep engine calls from overlapping; teardown must
    // not be the one thing that breaks that.
    expect(worker.peakConcurrencyAtDispose).toBe(0);

    engine.disconnect();
    await abandoned;
  });

  it("frees the slot when a request times out", async () => {
    const { engine, worker } = wire({ timeoutMs: 30 });
    await engine.load("wasm");
    worker.gate = new Promise<void>(() => {});

    await expect(
      engine.synthesize({ text: "slow", voice, speed: 1 }),
    ).rejects.toBeInstanceOf(VoxShotError);
    worker.gate = undefined;
    await deliver();

    // Before the fix the timed-out render kept the only slot and every later
    // request queued behind it for the life of the worker.
    await expect(engine.synthesize({ text: "next", voice, speed: 1 })).resolves.toBeDefined();
    expect(worker.requests.map((r) => r.text)).toContain("next");
  });

  it("stops serving without leaving queued work to run", async () => {
    const { engine, worker, close } = wire();
    await engine.load("wasm");
    let release!: () => void;
    worker.gate = new Promise<void>((resolve) => {
      release = resolve;
    });

    const first = engine.synthesize({ text: "running", voice, speed: 1 }).catch(() => undefined);
    const second = engine.synthesize({ text: "queued", voice, speed: 1 }).catch(() => undefined);
    await deliver();

    close();
    release();
    await deliver();
    await Promise.all([first, second]);

    // "queued" must never reach the engine after the server was told to stop.
    expect(worker.requests.map((r) => r.text)).not.toContain("queued");
  });
});

/**
 * Replying is the one thing that cannot be assumed to work: `postMessage`
 * throws on a closed port, and on a payload that cannot be cloned. A failure
 * there must not take the rest of the queue with it.
 */
describe("when replying itself fails", () => {
  const deliver = async (): Promise<void> => {
    for (let tick = 0; tick < 4; tick += 1) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  };

  /** A port on which every reply to one particular request fails to send. */
  function brittlePort(port: MessagePort, doomedId: number): RpcEndpoint {
    return {
      postMessage(message: unknown, transfer?: Transferable[]) {
        if ((message as { id?: number })?.id === doomedId) {
          throw new DOMException("could not be cloned", "DataCloneError");
        }
        port.postMessage(message, transfer ?? []);
      },
      addEventListener: (type: string, listener: EventListener) =>
        port.addEventListener(type, listener),
      removeEventListener: (type: string, listener: EventListener) =>
        port.removeEventListener(type, listener),
      start: () => port.start(),
    } as unknown as RpcEndpoint;
  }

  it("keeps serving the queue when a job cannot be answered at all", async () => {
    const channel = new MessageChannel();
    const worker = new RecordingEngine();
    // load is id 1, so the first synthesize is id 2. Every attempt to answer
    // that one fails: its reply, the failure notice sent in its place, and the
    // drain loop's fallback.
    const stop = exposeEngine(worker, brittlePort(channel.port2, 2));
    const engine = new WorkerSynthesisEngine(channel.port1, { timeoutMs: 800 });
    await engine.load("wasm");

    // The first render is held open so the second is genuinely queued behind
    // it — otherwise the first finishes before the second even arrives, and
    // the second starts a fresh drain that hides the problem.
    let release!: () => void;
    worker.stubborn = true;
    worker.gate = new Promise<void>((resolve) => {
      release = resolve;
    });

    const first = engine.synthesize({ text: "doomed", voice, speed: 1 }).catch(() => undefined);
    const second = engine.synthesize({ text: "behind it", voice, speed: 1 });
    await deliver();
    worker.gate = undefined;
    release();

    await expect(second).resolves.toBeDefined();
    await first;

    stop();
    channel.port1.close();
    channel.port2.close();
  });
});
