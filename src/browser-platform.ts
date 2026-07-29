import { AudioDecodeError, ZeroVoxError } from "./errors.js";
import type {
  AudioDecoder,
  AudioPlayer,
  DecodedAudio,
  GpuProbe,
  PcmAudio,
  Platform,
  StreamingAudioPlayer,
  StreamingPlayback,
} from "./platform.js";

/**
 * Structural subset of `navigator.gpu`. WebGPU types ship in a separate
 * package, so the shape the probe needs is declared locally instead.
 */
interface GpuNavigator {
  gpu?: {
    requestAdapter(): Promise<unknown>;
  };
}

function requireAudioContext(): typeof AudioContext {
  const ctor = (globalThis as { AudioContext?: typeof AudioContext }).AudioContext;
  if (!ctor) {
    throw new ZeroVoxError(
      "The Web Audio API is not available in this environment. Provide a custom platform instead.",
    );
  }
  return ctor;
}

/** Decodes encoded audio through the Web Audio API. */
export class BrowserAudioDecoder implements AudioDecoder {
  #context: AudioContext | undefined;

  async decode(data: ArrayBuffer): Promise<DecodedAudio> {
    const context = (this.#context ??= new (requireAudioContext())());

    let buffer: AudioBuffer;
    try {
      buffer = await context.decodeAudioData(data);
    } catch (cause) {
      throw new AudioDecodeError(cause instanceof Error ? cause.message : String(cause), { cause });
    }

    const channels: Float32Array[] = [];
    for (let index = 0; index < buffer.numberOfChannels; index += 1) {
      channels.push(buffer.getChannelData(index));
    }
    return { channels, sampleRate: buffer.sampleRate };
  }
}

/** Plays mono PCM through the Web Audio API. */
export class BrowserAudioPlayer implements AudioPlayer {
  #context: AudioContext | undefined;

  async play(audio: PcmAudio): Promise<void> {
    if (audio.samples.length === 0) {
      return;
    }

    const context = (this.#context ??= new (requireAudioContext())());
    if (context.state === "suspended") {
      await context.resume();
    }

    const buffer = context.createBuffer(1, audio.samples.length, audio.sampleRate);
    // `copyToChannel` is typed against ArrayBuffer-backed views only; the call
    // is safe for any backing buffer because it copies out of `samples`.
    buffer.copyToChannel(audio.samples as Float32Array<ArrayBuffer>, 0);

    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);

    await new Promise<void>((resolve) => {
      source.onended = () => resolve();
      source.start();
    });
  }

  /** Release the underlying `AudioContext`. */
  async dispose(): Promise<void> {
    const context = this.#context;
    this.#context = undefined;
    await context?.close();
  }
}

/** Reports WebGPU availability by asking for an adapter. */
export class BrowserGpuProbe implements GpuProbe {
  async isAvailable(): Promise<boolean> {
    const gpu = (globalThis.navigator as GpuNavigator | undefined)?.gpu;
    if (!gpu) {
      return false;
    }
    try {
      return (await gpu.requestAdapter()) !== null;
    } catch {
      return false;
    }
  }
}

/** Name under which the streaming processor registers itself. */
const STREAM_PROCESSOR_NAME = "zerovox-stream-player";

/**
 * The AudioWorklet processor, shipped as source text and loaded through a
 * blob URL so the library needs no separate worklet asset.
 *
 * It plays a ring of PCM chunks back to back. Backpressure: each `write`
 * owes one `ready` reply, sent once buffered audio falls under the low-water
 * mark, so the producer stays exactly one chunk ahead. `flush` drops the ring
 * and reports how many samples were actually played, which is how skipping
 * finds out what was audible.
 */
const STREAM_PROCESSOR_CODE = `
const LOW_WATER_SECONDS = 0.2;

class ZeroVoxStreamPlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunks = [];
    this.offset = 0;
    this.buffered = 0;
    this.played = 0;
    this.ended = false;
    this.pendingReady = 0;
    this.port.onmessage = (event) => this.handle(event.data);
  }

  handle(message) {
    if (message.type === "write") {
      this.chunks.push(message.samples);
      this.buffered += message.samples.length;
      this.pendingReady += 1;
      this.notifyReady();
    } else if (message.type === "end") {
      this.ended = true;
      this.notifyDrained();
    } else if (message.type === "flush") {
      this.chunks = [];
      this.offset = 0;
      this.buffered = 0;
      this.notifyReady();
      this.port.postMessage({ type: "flushed", id: message.id, played: this.played });
      this.notifyDrained();
    }
  }

  notifyReady() {
    while (this.pendingReady > 0 && this.buffered <= LOW_WATER_SECONDS * sampleRate) {
      this.pendingReady -= 1;
      this.port.postMessage({ type: "ready" });
    }
  }

  notifyDrained() {
    if (this.ended && this.buffered === 0) {
      this.ended = false;
      this.port.postMessage({ type: "drained" });
    }
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    let index = 0;
    while (index < output.length && this.chunks.length > 0) {
      const head = this.chunks[0];
      const count = Math.min(output.length - index, head.length - this.offset);
      output.set(head.subarray(this.offset, this.offset + count), index);
      index += count;
      this.offset += count;
      this.buffered -= count;
      this.played += count;
      if (this.offset === head.length) {
        this.chunks.shift();
        this.offset = 0;
      }
    }
    output.fill(0, index);
    this.notifyReady();
    this.notifyDrained();
    return true;
  }
}

registerProcessor("${STREAM_PROCESSOR_NAME}", ZeroVoxStreamPlayer);
`;

let streamProcessorUrl: string | undefined;

function getStreamProcessorUrl(): string {
  return (streamProcessorUrl ??= URL.createObjectURL(
    new Blob([STREAM_PROCESSOR_CODE], { type: "application/javascript" }),
  ));
}

/** Structural subset of `AudioWorkletNode`, declared locally like the rest. */
interface WorkletNodeLike {
  readonly port: {
    onmessage: ((event: { data: unknown }) => void) | null;
    postMessage(message: unknown, transfer?: unknown[]): void;
    start?(): void;
  };
  connect(destination: unknown): void;
}

interface StreamingContextLike {
  readonly destination: unknown;
  readonly audioWorklet?: { addModule(url: string): Promise<void> };
  createGain(): GainNodeLike;
  close(): Promise<void>;
}

interface GainNodeLike {
  readonly gain: { value: number };
  connect(destination: unknown): void;
}

interface WorkletMessage {
  type?: unknown;
  id?: unknown;
  played?: unknown;
}

/** Gapless PCM streaming through an AudioWorklet ring buffer. */
export class BrowserStreamingAudioPlayer implements StreamingAudioPlayer {
  async open(sampleRate: number): Promise<StreamingPlayback> {
    const ContextCtor = requireAudioContext();
    const NodeCtor = (globalThis as { AudioWorkletNode?: new (context: unknown, name: string) => WorkletNodeLike })
      .AudioWorkletNode;

    const context = new ContextCtor({ sampleRate }) as unknown as StreamingContextLike;
    if (!NodeCtor || !context.audioWorklet) {
      await context.close();
      throw new ZeroVoxError(
        "AudioWorklet is not available in this environment. Provide a custom streaming player instead.",
      );
    }

    await context.audioWorklet.addModule(getStreamProcessorUrl());
    const node = new NodeCtor(context, STREAM_PROCESSOR_NAME);
    const gain = context.createGain();
    node.connect(gain);
    gain.connect(context.destination);
    return new BrowserStreamingPlayback(context, node, gain);
  }
}

class BrowserStreamingPlayback implements StreamingPlayback {
  readonly #context: StreamingContextLike;
  readonly #node: WorkletNodeLike;
  readonly #gain: GainNodeLike;

  readonly #pendingWrites: (() => void)[] = [];
  readonly #pendingFlushes = new Map<number, (played: number) => void>();
  readonly #drainWaiters: (() => void)[] = [];
  #nextFlushId = 1;
  #closed = false;

  constructor(context: StreamingContextLike, node: WorkletNodeLike, gain: GainNodeLike) {
    this.#context = context;
    this.#node = node;
    this.#gain = gain;
    node.port.onmessage = (event) => this.#receive(event.data as WorkletMessage);
    node.port.start?.();
  }

  write(samples: Float32Array): Promise<void> {
    if (this.#closed) {
      return Promise.resolve();
    }
    // Copy before transferring: the caller keeps its samples, so a skip can
    // re-enqueue a chunk that the worklet flushed away.
    const copy = Float32Array.from(samples);
    return new Promise((resolve) => {
      this.#pendingWrites.push(resolve);
      this.#node.port.postMessage({ type: "write", samples: copy }, [copy.buffer]);
    });
  }

  end(): Promise<void> {
    if (this.#closed) {
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      this.#drainWaiters.push(() => {
        void this.#close().then(resolve);
      });
      this.#node.port.postMessage({ type: "end" });
    });
  }

  flush(): Promise<number> {
    if (this.#closed) {
      return Promise.resolve(0);
    }
    const id = this.#nextFlushId++;
    return new Promise((resolve) => {
      this.#pendingFlushes.set(id, resolve);
      this.#node.port.postMessage({ type: "flush", id });
    });
  }

  async stop(): Promise<void> {
    if (this.#closed) {
      return;
    }
    await this.#close();
  }

  setVolume(volume: number): void {
    this.#gain.gain.value = Math.max(0, volume);
  }

  async #close(): Promise<void> {
    if (this.#closed) {
      return;
    }
    this.#closed = true;

    for (const resolve of this.#pendingWrites.splice(0)) {
      resolve();
    }
    for (const resolve of this.#pendingFlushes.values()) {
      resolve(0);
    }
    this.#pendingFlushes.clear();
    for (const resolve of this.#drainWaiters.splice(0)) {
      resolve();
    }
    await this.#context.close();
  }

  #receive(message: WorkletMessage): void {
    if (message.type === "ready") {
      this.#pendingWrites.shift()?.();
    } else if (message.type === "flushed") {
      const resolve = this.#pendingFlushes.get(message.id as number);
      this.#pendingFlushes.delete(message.id as number);
      resolve?.(message.played as number);
    } else if (message.type === "drained") {
      for (const resolve of this.#drainWaiters.splice(0)) {
        resolve();
      }
    }
  }
}

/** The default platform: everything backed by real browser APIs. */
export function createBrowserPlatform(): Platform {
  return {
    decoder: new BrowserAudioDecoder(),
    player: new BrowserAudioPlayer(),
    gpu: new BrowserGpuProbe(),
    streamingPlayer: new BrowserStreamingAudioPlayer(),
  };
}
