import { InvalidInputError, VoxShotError } from "../errors.js";
import type { AudioPlayer } from "../platform.js";
import { encodeWav } from "./wav.js";

/**
 * The result of a synthesis call: raw samples plus the conveniences most
 * applications need (playback, WAV export, blob creation).
 */
export class SynthesizedAudio {
  readonly samples: Float32Array;
  readonly sampleRate: number;

  readonly #player: AudioPlayer | undefined;

  constructor(samples: Float32Array, sampleRate: number, player?: AudioPlayer) {
    if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
      throw new InvalidInputError("sampleRate must be a positive finite number.");
    }
    this.samples = samples;
    this.sampleRate = sampleRate;
    this.#player = player;
  }

  /** Length of the signal in seconds. */
  get duration(): number {
    return this.samples.length / this.sampleRate;
  }

  /** Encode the signal as a 16 bit PCM WAV file. */
  toWav(): ArrayBuffer {
    return encodeWav(this.samples, this.sampleRate);
  }

  /** Encode the signal as an `audio/wav` blob, ready for an `<audio>` element. */
  toBlob(): Blob {
    return new Blob([this.toWav()], { type: "audio/wav" });
  }

  /** Play the signal, resolving once playback has finished. */
  async play(): Promise<void> {
    if (!this.#player) {
      throw new VoxShotError(
        "This audio has no player attached. Create it through VoxShot, or pass a player explicitly.",
      );
    }
    await this.#player.play({ samples: this.samples, sampleRate: this.sampleRate });
  }

  /** Join consecutive chunks — typically the output of `VoxShot.stream()`. */
  static concat(chunks: readonly SynthesizedAudio[]): SynthesizedAudio {
    const first = chunks[0];
    if (!first) {
      throw new InvalidInputError("At least one audio chunk is required.");
    }

    let length = 0;
    for (const chunk of chunks) {
      if (chunk.sampleRate !== first.sampleRate) {
        throw new InvalidInputError("Every chunk must share the same sample rate.");
      }
      length += chunk.samples.length;
    }

    const samples = new Float32Array(length);
    let offset = 0;
    for (const chunk of chunks) {
      samples.set(chunk.samples, offset);
      offset += chunk.samples.length;
    }
    return new SynthesizedAudio(samples, first.sampleRate, first.#player);
  }
}
