import { afterEach, describe, expect, it, vi } from "vitest";

import { BrowserStreamingAudioPlayer } from "../src/browser-platform.js";
import { VoxShotError } from "../src/errors.js";

class FakePort {
  onmessage: ((event: { data: unknown }) => void) | null = null;
  posted: { message: Record<string, unknown>; transfer: unknown[] | undefined }[] = [];

  postMessage(message: Record<string, unknown>, transfer?: unknown[]): void {
    this.posted.push({ message, transfer });
  }

  /** Simulate a message from the worklet processor. */
  emit(data: unknown): void {
    this.onmessage?.({ data });
  }
}

class FakeWorkletNode {
  static instances: FakeWorkletNode[] = [];
  readonly port = new FakePort();
  connectedTo: unknown = null;

  constructor(
    readonly context: unknown,
    readonly name: string,
  ) {
    FakeWorkletNode.instances.push(this);
  }

  connect(destination: unknown): void {
    this.connectedTo = destination;
  }
}

class FakeGain {
  gain = { value: 1 };
  connectedTo: unknown = null;

  connect(destination: unknown): void {
    this.connectedTo = destination;
  }
}

class FakeStreamingContext {
  static instances: FakeStreamingContext[] = [];

  destination = { id: "destination" };
  gains: FakeGain[] = [];
  closed = false;
  addedModules: string[] = [];
  audioWorklet: { addModule: (url: string) => Promise<void> } | undefined;

  constructor(readonly options?: { sampleRate?: number }) {
    FakeStreamingContext.instances.push(this);
    this.audioWorklet = {
      addModule: async (url: string) => {
        this.addedModules.push(url);
      },
    };
  }

  createGain(): FakeGain {
    const gain = new FakeGain();
    this.gains.push(gain);
    return gain;
  }

  async close(): Promise<void> {
    this.closed = true;
  }
}

function install(): void {
  vi.stubGlobal("AudioContext", FakeStreamingContext);
  vi.stubGlobal("AudioWorkletNode", FakeWorkletNode);
  vi.stubGlobal("URL", {
    createObjectURL: vi.fn(() => "blob:voxshot-worklet"),
  });
}

async function openPlayback() {
  const player = new BrowserStreamingAudioPlayer();
  const playback = await player.open(24_000);
  const context = FakeStreamingContext.instances[
    FakeStreamingContext.instances.length - 1
  ] as FakeStreamingContext;
  const node = FakeWorkletNode.instances[FakeWorkletNode.instances.length - 1] as FakeWorkletNode;
  return { player, playback, context, node };
}

afterEach(() => {
  vi.unstubAllGlobals();
  FakeStreamingContext.instances = [];
  FakeWorkletNode.instances = [];
});

describe("BrowserStreamingAudioPlayer", () => {
  it("opens a context at the requested sample rate and wires node → gain → destination", async () => {
    install();
    const { context, node } = await openPlayback();

    expect(context.options?.sampleRate).toBe(24_000);
    expect(context.addedModules).toHaveLength(1);
    expect(node.connectedTo).toBe(context.gains[0]);
    expect(context.gains[0]?.connectedTo).toBe(context.destination);
  });

  it("throws a VoxShotError without an AudioContext", async () => {
    vi.stubGlobal("AudioContext", undefined);

    await expect(new BrowserStreamingAudioPlayer().open(24_000)).rejects.toBeInstanceOf(
      VoxShotError,
    );
  });

  it("throws a VoxShotError when AudioWorklet is unavailable", async () => {
    install();
    vi.stubGlobal("AudioWorkletNode", undefined);

    await expect(new BrowserStreamingAudioPlayer().open(24_000)).rejects.toBeInstanceOf(
      VoxShotError,
    );
  });

  it("posts written samples as a transferable copy", async () => {
    install();
    const { playback, node } = await openPlayback();
    const samples = Float32Array.from([0.1, 0.2, 0.3]);

    void playback.write(samples);

    const posted = node.port.posted[0] as { message: Record<string, unknown>; transfer: unknown[] };
    expect(posted.message.type).toBe("write");
    const sent = posted.message.samples as Float32Array;
    expect(Array.from(sent)).toEqual([0.1, 0.2, 0.3].map((v) => Math.fround(v)));
    expect(sent).not.toBe(samples);
    expect(posted.transfer).toEqual([sent.buffer]);
    // The caller's buffer must stay usable for later re-writes after a skip.
    expect(samples.length).toBe(3);
  });

  it("resolves writes in order as the worklet reports readiness", async () => {
    install();
    const { playback, node } = await openPlayback();

    const order: string[] = [];
    void playback.write(new Float32Array(4)).then(() => order.push("first"));
    void playback.write(new Float32Array(4)).then(() => order.push("second"));

    node.port.emit({ type: "ready" });
    await Promise.resolve();
    expect(order).toEqual(["first"]);

    node.port.emit({ type: "ready" });
    await Promise.resolve();
    expect(order).toEqual(["first", "second"]);
  });

  it("end() posts end, resolves on drained and closes the context", async () => {
    install();
    const { playback, node, context } = await openPlayback();

    let ended = false;
    const done = playback.end().then(() => {
      ended = true;
    });
    expect(node.port.posted.some((p) => p.message.type === "end")).toBe(true);
    expect(ended).toBe(false);

    node.port.emit({ type: "drained" });
    await done;
    expect(context.closed).toBe(true);
  });

  it("flush() resolves with the played sample count from the worklet", async () => {
    install();
    const { playback, node } = await openPlayback();

    const flushed = playback.flush();
    const posted = node.port.posted.find((p) => p.message.type === "flush");
    expect(posted).toBeDefined();

    node.port.emit({ type: "flushed", id: posted?.message.id, played: 4_321 });
    await expect(flushed).resolves.toBe(4_321);
  });

  it("setVolume drives the gain node and clamps negatives to silence", async () => {
    install();
    const { playback, context } = await openPlayback();

    playback.setVolume(0.5);
    expect(context.gains[0]?.gain.value).toBe(0.5);

    playback.setVolume(-1);
    expect(context.gains[0]?.gain.value).toBe(0);
  });

  it("stop() closes the context and unblocks all pending calls", async () => {
    install();
    const { playback, context } = await openPlayback();

    const write = playback.write(new Float32Array(4));
    const ended = playback.end();
    const flushed = playback.flush();

    await playback.stop();

    expect(context.closed).toBe(true);
    await expect(write).resolves.toBeUndefined();
    await expect(ended).resolves.toBeUndefined();
    await expect(flushed).resolves.toBe(0);
  });

  it("write after stop resolves immediately without posting", async () => {
    install();
    const { playback, node } = await openPlayback();

    await playback.stop();
    const before = node.port.posted.length;
    await playback.write(new Float32Array(4));

    expect(node.port.posted.length).toBe(before);
  });
});
