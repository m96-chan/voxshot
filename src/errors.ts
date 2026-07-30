/**
 * Machine readable identifiers attached to every {@link VoxShotError}.
 */
export type VoxShotErrorCode =
  | "UNKNOWN"
  | "DEVICE_UNAVAILABLE"
  | "NO_VOICE"
  | "VOICE_NOT_FOUND"
  | "INVALID_INPUT"
  | "AUDIO_DECODE_FAILED"
  | "DISPOSED";

/**
 * Base class for every error thrown by VoxShot.
 *
 * Consumers can branch on {@link VoxShotError.code} instead of matching
 * message strings, which keeps error handling stable across releases.
 */
export class VoxShotError extends Error {
  readonly code: VoxShotErrorCode;

  constructor(message: string, code: VoxShotErrorCode = "UNKNOWN", options?: ErrorOptions) {
    super(message, options);
    this.name = "VoxShotError";
    this.code = code;
  }
}

/** The requested inference device is not available in this environment. */
export class DeviceUnavailableError extends VoxShotError {
  readonly device: string;

  constructor(device: string, options?: ErrorOptions) {
    super(`Inference device "${device}" is not available in this environment.`, "DEVICE_UNAVAILABLE", options);
    this.name = "DeviceUnavailableError";
    this.device = device;
  }
}

/** Synthesis was requested before a voice was cloned or selected. */
export class NoVoiceError extends VoxShotError {
  constructor(options?: ErrorOptions) {
    super("No voice is active. Call cloneVoice() or useVoice() first.", "NO_VOICE", options);
    this.name = "NoVoiceError";
  }
}

/** A saved voice was requested but does not exist in the voice store. */
export class VoiceNotFoundError extends VoxShotError {
  readonly voiceName: string;

  constructor(voiceName: string, options?: ErrorOptions) {
    super(`Voice "${voiceName}" was not found.`, "VOICE_NOT_FOUND", options);
    this.name = "VoiceNotFoundError";
    this.voiceName = voiceName;
  }
}

/** An argument did not satisfy the documented contract. */
export class InvalidInputError extends VoxShotError {
  constructor(message: string, options?: ErrorOptions) {
    super(message, "INVALID_INPUT", options);
    this.name = "InvalidInputError";
  }
}

/** Reference audio could not be decoded into PCM samples. */
export class AudioDecodeError extends VoxShotError {
  constructor(message: string, options?: ErrorOptions) {
    super(`Failed to decode audio: ${message}`, "AUDIO_DECODE_FAILED", options);
    this.name = "AudioDecodeError";
  }
}

/** The instance was used after {@link VoxShot.dispose} was called. */
export class DisposedError extends VoxShotError {
  constructor(options?: ErrorOptions) {
    super("This VoxShot instance has been disposed.", "DISPOSED", options);
    this.name = "DisposedError";
  }
}

/** Type guard narrowing an unknown thrown value to a {@link VoxShotError}. */
export function isVoxShotError(value: unknown): value is VoxShotError {
  return value instanceof VoxShotError;
}
