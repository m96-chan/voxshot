import { normalizePeak } from "../audio/pcm.js";
import type { ResolvedDevice } from "../device.js";
import { InvalidInputError } from "../errors.js";
import type { PcmAudio } from "../platform.js";
import type { SynthesisEngine, SynthesisRequest } from "./types.js";

/** Dimensionality of the descriptor {@link PlaceholderEngine.embed} produces. */
export const VOICE_EMBEDDING_SIZE = 4;

const DEFAULT_SAMPLE_RATE = 24_000;
const SECONDS_PER_CHARACTER = 0.08;
const MIN_SPEED = 0.25;
const MAX_SPEED = 4;
const MIN_PITCH_HZ = 80;
const MAX_PITCH_HZ = 400;
const OUTPUT_PEAK = 0.95;

export interface PlaceholderEngineOptions {
  /**
   * Sample rate of the rendered audio.
   *
   * @defaultValue 24000
   */
  sampleRate?: number;
}

/**
 * A dependency-free stand-in for a neural TTS model.
 *
 * It extracts a coarse timbre descriptor (pitch, brightness, loudness,
 * roughness) from reference audio and renders text as a harmonic tone driven
 * by that descriptor. The output is intelligible as *speech-shaped* audio, not
 * as speech: it exists so the whole pipeline — cloning, storage, chunking,
 * streaming, playback — can be built and tested before an ONNX Runtime Web
 * backend lands.
 *
 * Replace it by passing your own {@link SynthesisEngine} to `VoxShot.create`.
 */
export class PlaceholderEngine implements SynthesisEngine {
  readonly name = "placeholder";
  readonly sampleRate: number;

  #device: ResolvedDevice | undefined;

  constructor(options: PlaceholderEngineOptions = {}) {
    const sampleRate = options.sampleRate ?? DEFAULT_SAMPLE_RATE;
    if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
      throw new InvalidInputError("sampleRate must be a positive finite number.");
    }
    this.sampleRate = sampleRate;
  }

  /** The device this engine was loaded on, if any. */
  get device(): ResolvedDevice | undefined {
    return this.#device;
  }

  async load(device: ResolvedDevice): Promise<void> {
    this.#device = device;
  }

  async embed(audio: PcmAudio): Promise<Float32Array> {
    const { samples, sampleRate } = audio;
    if (samples.length === 0) {
      throw new InvalidInputError("Reference audio must not be empty.");
    }

    let sumSquares = 0;
    let sumMagnitude = 0;
    let sumDelta = 0;
    let crossings = 0;
    let peak = 0;
    let previous = samples[0] as number;

    for (let index = 0; index < samples.length; index += 1) {
      const sample = samples[index] as number;
      const magnitude = Math.abs(sample);

      sumSquares += sample * sample;
      sumMagnitude += magnitude;
      peak = Math.max(peak, magnitude);
      if (index > 0) {
        sumDelta += Math.abs(sample - previous);
        if ((sample >= 0 && previous < 0) || (sample < 0 && previous >= 0)) {
          crossings += 1;
        }
      }
      previous = sample;
    }

    if (peak === 0) {
      throw new InvalidInputError("Reference audio is silent; cannot extract a voice.");
    }

    const rms = Math.sqrt(sumSquares / samples.length);
    const meanMagnitude = sumMagnitude / samples.length;
    // Zero crossings of a sine happen twice per period, so f0 ≈ zcr * sr / 2.
    const pitchHz = (crossings / samples.length) * sampleRate * 0.5;
    const brightness = sumDelta / (samples.length - 1 || 1) / meanMagnitude;
    const roughness = peak / rms;

    return Float32Array.from([pitchHz, brightness, rms, roughness]);
  }

  async synthesize(request: SynthesisRequest): Promise<Float32Array> {
    const { text, voice, speed } = request;

    if (!Number.isFinite(speed) || speed < MIN_SPEED || speed > MAX_SPEED) {
      throw new InvalidInputError(`speed must be between ${MIN_SPEED} and ${MAX_SPEED}.`);
    }
    if (voice.vector.length !== VOICE_EMBEDDING_SIZE) {
      throw new InvalidInputError(
        `This engine expects a ${VOICE_EMBEDDING_SIZE} dimensional voice embedding.`,
      );
    }

    const characters = [...text];
    if (characters.length === 0) {
      return new Float32Array(0);
    }

    const basePitch = clamp(voice.vector[0] as number, MIN_PITCH_HZ, MAX_PITCH_HZ);
    const brightness = clamp(voice.vector[1] as number, 0, 1);
    const samplesPerCharacter = Math.max(1, Math.round((SECONDS_PER_CHARACTER * this.sampleRate) / speed));
    const output = new Float32Array(characters.length * samplesPerCharacter);

    let phase = 0;
    let offset = 0;

    for (const character of characters) {
      if (/\s/.test(character)) {
        offset += samplesPerCharacter;
        continue;
      }

      const frequency = basePitch * pitchFactor(character);
      const increment = (2 * Math.PI * frequency) / this.sampleRate;

      for (let index = 0; index < samplesPerCharacter; index += 1) {
        phase += increment;
        const envelope = raisedCosine(index / samplesPerCharacter);
        const harmonics =
          Math.sin(phase) +
          brightness * 0.4 * Math.sin(2 * phase) +
          brightness * 0.2 * Math.sin(3 * phase);
        output[offset + index] = envelope * harmonics;
      }
      offset += samplesPerCharacter;
    }

    return normalizePeak(output, OUTPUT_PEAK);
  }

  async dispose(): Promise<void> {
    this.#device = undefined;
  }
}

/** Deterministic per-character pitch jitter, so text is not rendered monotone. */
function pitchFactor(character: string): number {
  const code = character.codePointAt(0) ?? 0;
  const step = ((code * 2_654_435_761) >>> 0) % 9; // 0..8
  return 1 + (step - 4) * 0.02; // 0.92 .. 1.08
}

/** Click-free attack and release around a flat sustain. */
function raisedCosine(position: number): number {
  const edge = 0.15;
  if (position < edge) {
    return 0.5 - 0.5 * Math.cos((Math.PI * position) / edge);
  }
  if (position > 1 - edge) {
    return 0.5 - 0.5 * Math.cos((Math.PI * (1 - position)) / edge);
  }
  return 1;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
