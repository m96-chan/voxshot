/** A mono PCM signal together with the rate it was sampled at. */
export interface PcmAudio {
  readonly samples: Float32Array;
  readonly sampleRate: number;
}

/** Multi channel PCM as returned by a decoder. */
export interface DecodedAudio {
  readonly channels: readonly Float32Array[];
  readonly sampleRate: number;
}

/** Turns encoded audio (wav, mp3, ogg, ...) into raw PCM. */
export interface AudioDecoder {
  decode(data: ArrayBuffer): Promise<DecodedAudio>;
}

/** Plays a mono PCM signal, resolving when playback finishes. */
export interface AudioPlayer {
  play(audio: PcmAudio): Promise<void>;
}

/** Reports whether WebGPU can be used for inference. */
export interface GpuProbe {
  isAvailable(): Promise<boolean>;
}

/**
 * Every browser capability ZeroVox depends on, bundled behind interfaces so
 * that tests (and non browser hosts) can supply their own implementations.
 */
export interface Platform {
  readonly decoder: AudioDecoder;
  readonly player: AudioPlayer;
  readonly gpu: GpuProbe;
}
