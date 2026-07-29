import { AudioDecodeError, ZeroVoxError } from "./errors.js";
import type { AudioDecoder, AudioPlayer, DecodedAudio, GpuProbe, PcmAudio, Platform } from "./platform.js";

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

/** The default platform: everything backed by real browser APIs. */
export function createBrowserPlatform(): Platform {
  return {
    decoder: new BrowserAudioDecoder(),
    player: new BrowserAudioPlayer(),
    gpu: new BrowserGpuProbe(),
  };
}
