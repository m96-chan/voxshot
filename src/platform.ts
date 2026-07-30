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
 * A live gapless output stream: PCM written here plays back to back with no
 * per-chunk scheduling gaps.
 */
export interface StreamingPlayback {
  /**
   * Enqueue samples for playback. The promise resolves when the stream is
   * ready for the next chunk — buffered audio has drained below the
   * implementation's low-water mark — so awaiting each write paces the
   * producer without ever letting the stream run dry.
   */
  write(samples: Float32Array): Promise<void>;
  /** No more audio will be written. Resolves once everything has played. */
  end(): Promise<void>;
  /**
   * Discard all buffered audio immediately. Pending writes resolve, and the
   * promise carries the total number of samples actually played since the
   * stream opened, so the caller can tell what was audible when it flushed.
   */
  flush(): Promise<number>;
  /** Tear the stream down immediately, discarding buffered audio. */
  stop(): Promise<void>;
  /** Playback gain. Values are clamped to be non negative. */
  setVolume(volume: number): void;
}

/** Opens gapless playback streams. */
export interface StreamingAudioPlayer {
  open(sampleRate: number): Promise<StreamingPlayback>;
}

/**
 * Every browser capability VoxShot depends on, bundled behind interfaces so
 * that tests (and non browser hosts) can supply their own implementations.
 */
export interface Platform {
  readonly decoder: AudioDecoder;
  readonly player: AudioPlayer;
  readonly gpu: GpuProbe;
  /** Gapless streaming output; `VoxShot.play()` requires it. */
  readonly streamingPlayer?: StreamingAudioPlayer;
}
