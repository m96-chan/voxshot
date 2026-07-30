import { afterEach, describe, expect, it, vi } from "vitest";

import {
  BrowserAudioDecoder,
  BrowserAudioPlayer,
  BrowserGpuProbe,
  createBrowserPlatform,
} from "../src/browser-platform.js";
import { AudioDecodeError, VoxShotError } from "../src/errors.js";

/** Minimal stand-in for the parts of the Web Audio API the library uses. */
class FakeAudioBuffer {
  readonly channels: Float32Array[];

  constructor(
    readonly numberOfChannels: number,
    readonly length: number,
    readonly sampleRate: number,
  ) {
    this.channels = Array.from({ length: numberOfChannels }, () => new Float32Array(length));
  }

  getChannelData(channel: number): Float32Array {
    return this.channels[channel] as Float32Array;
  }

  copyToChannel(source: Float32Array, channel: number): void {
    (this.channels[channel] as Float32Array).set(source);
  }
}

class FakeBufferSource {
  buffer: FakeAudioBuffer | null = null;
  onended: (() => void) | null = null;
  connectedTo: unknown = null;
  started = false;

  connect(destination: unknown): void {
    this.connectedTo = destination;
  }

  start(): void {
    this.started = true;
    queueMicrotask(() => this.onended?.());
  }
}

class FakeAudioContext {
  static instances: FakeAudioContext[] = [];

  state: "running" | "suspended" | "closed" = "running";
  destination = { id: "destination" };
  closed = false;
  sources: FakeBufferSource[] = [];
  decodeResult: FakeAudioBuffer | Error = new FakeAudioBuffer(2, 4, 44_100);

  constructor() {
    FakeAudioContext.instances.push(this);
  }

  async decodeAudioData(_data: ArrayBuffer): Promise<FakeAudioBuffer> {
    if (this.decodeResult instanceof Error) {
      throw this.decodeResult;
    }
    return this.decodeResult;
  }

  createBuffer(channels: number, length: number, sampleRate: number): FakeAudioBuffer {
    return new FakeAudioBuffer(channels, length, sampleRate);
  }

  createBufferSource(): FakeBufferSource {
    const source = new FakeBufferSource();
    this.sources.push(source);
    return source;
  }

  async resume(): Promise<void> {
    this.state = "running";
  }

  async close(): Promise<void> {
    this.closed = true;
    this.state = "closed";
  }
}

function installAudioContext(): typeof FakeAudioContext {
  FakeAudioContext.instances = [];
  vi.stubGlobal("AudioContext", FakeAudioContext);
  return FakeAudioContext;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("BrowserAudioDecoder", () => {
  it("decodes into per channel Float32Arrays", async () => {
    installAudioContext();
    const decoder = new BrowserAudioDecoder();

    const decoded = await decoder.decode(new ArrayBuffer(8));

    expect(decoded.sampleRate).toBe(44_100);
    expect(decoded.channels).toHaveLength(2);
    expect(decoded.channels[0]).toHaveLength(4);
  });

  it("reuses one AudioContext across calls", async () => {
    const context = installAudioContext();
    const decoder = new BrowserAudioDecoder();

    await decoder.decode(new ArrayBuffer(8));
    await decoder.decode(new ArrayBuffer(8));

    expect(context.instances).toHaveLength(1);
  });

  it("wraps decoding failures in AudioDecodeError", async () => {
    const context = installAudioContext();
    const decoder = new BrowserAudioDecoder();
    await decoder.decode(new ArrayBuffer(8));
    (context.instances[0] as FakeAudioContext).decodeResult = new Error("corrupt");

    await expect(decoder.decode(new ArrayBuffer(8))).rejects.toBeInstanceOf(AudioDecodeError);
  });

  it("throws when the Web Audio API is missing", async () => {
    vi.stubGlobal("AudioContext", undefined);

    await expect(new BrowserAudioDecoder().decode(new ArrayBuffer(8))).rejects.toThrow(VoxShotError);
  });
});

describe("BrowserAudioPlayer", () => {
  it("plays the samples through a buffer source and resolves when done", async () => {
    const context = installAudioContext();
    const player = new BrowserAudioPlayer();

    await player.play({ samples: new Float32Array([0, 0.5, -0.5]), sampleRate: 16_000 });

    const instance = context.instances[0] as FakeAudioContext;
    const source = instance.sources[0] as FakeBufferSource;
    expect(source.started).toBe(true);
    expect(source.connectedTo).toBe(instance.destination);
    expect(Array.from((source.buffer as FakeAudioBuffer).getChannelData(0))).toEqual([0, 0.5, -0.5]);
  });

  it("resumes a suspended context before playing", async () => {
    const context = installAudioContext();
    const player = new BrowserAudioPlayer();
    const first = player.play({ samples: new Float32Array([1]), sampleRate: 16_000 });
    (context.instances[0] as FakeAudioContext).state = "suspended";
    await first;

    await player.play({ samples: new Float32Array([1]), sampleRate: 16_000 });

    expect((context.instances[0] as FakeAudioContext).state).toBe("running");
  });

  it("ignores an empty signal", async () => {
    const context = installAudioContext();

    await new BrowserAudioPlayer().play({ samples: new Float32Array(0), sampleRate: 16_000 });

    expect(context.instances).toHaveLength(0);
  });

  it("closes the context on dispose", async () => {
    const context = installAudioContext();
    const player = new BrowserAudioPlayer();
    await player.play({ samples: new Float32Array([1]), sampleRate: 16_000 });

    await player.dispose();

    expect((context.instances[0] as FakeAudioContext).closed).toBe(true);
  });

  it("tolerates dispose before any playback", async () => {
    installAudioContext();

    await expect(new BrowserAudioPlayer().dispose()).resolves.toBeUndefined();
  });

  it("throws when the Web Audio API is missing", async () => {
    vi.stubGlobal("AudioContext", undefined);

    await expect(
      new BrowserAudioPlayer().play({ samples: new Float32Array([1]), sampleRate: 16_000 }),
    ).rejects.toThrow(VoxShotError);
  });
});

describe("BrowserGpuProbe", () => {
  it("reports true when an adapter is granted", async () => {
    vi.stubGlobal("navigator", { gpu: { requestAdapter: async () => ({ name: "fake" }) } });

    await expect(new BrowserGpuProbe().isAvailable()).resolves.toBe(true);
  });

  it("reports false when no adapter is granted", async () => {
    vi.stubGlobal("navigator", { gpu: { requestAdapter: async () => null } });

    await expect(new BrowserGpuProbe().isAvailable()).resolves.toBe(false);
  });

  it("reports false when navigator.gpu is missing", async () => {
    vi.stubGlobal("navigator", {});

    await expect(new BrowserGpuProbe().isAvailable()).resolves.toBe(false);
  });

  it("reports false when requesting an adapter throws", async () => {
    vi.stubGlobal("navigator", {
      gpu: {
        requestAdapter: async () => {
          throw new Error("denied");
        },
      },
    });

    await expect(new BrowserGpuProbe().isAvailable()).resolves.toBe(false);
  });
});

describe("createBrowserPlatform", () => {
  it("bundles the browser implementations", () => {
    const platform = createBrowserPlatform();

    expect(platform.decoder).toBeInstanceOf(BrowserAudioDecoder);
    expect(platform.player).toBeInstanceOf(BrowserAudioPlayer);
    expect(platform.gpu).toBeInstanceOf(BrowserGpuProbe);
  });
});
