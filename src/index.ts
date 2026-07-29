export { ZeroVox } from "./zerovox.js";
export type { SpeakOptions, VoiceSource, ZeroVoxOptions } from "./zerovox.js";

export { SynthesizedAudio } from "./audio/synthesized-audio.js";
export { normalizePeak, resample, toMono, trimSilence } from "./audio/pcm.js";
export { encodeWav } from "./audio/wav.js";

export { resolveDevice } from "./device.js";
export type { DevicePreference, ResolvedDevice } from "./device.js";

export { PlaceholderEngine, VOICE_EMBEDDING_SIZE } from "./engine/placeholder-engine.js";
export type { PlaceholderEngineOptions } from "./engine/placeholder-engine.js";
export type { SynthesisEngine, SynthesisRequest } from "./engine/types.js";

export {
  AudioDecodeError,
  DeviceUnavailableError,
  DisposedError,
  InvalidInputError,
  NoVoiceError,
  VoiceNotFoundError,
  ZeroVoxError,
  isZeroVoxError,
} from "./errors.js";
export type { ZeroVoxErrorCode } from "./errors.js";

export {
  BrowserAudioDecoder,
  BrowserAudioPlayer,
  BrowserGpuProbe,
  createBrowserPlatform,
} from "./browser-platform.js";
export type {
  AudioDecoder,
  AudioPlayer,
  DecodedAudio,
  GpuProbe,
  PcmAudio,
  Platform,
} from "./platform.js";

export { normalizeText } from "./text/normalize.js";
export { splitSentences } from "./text/segment.js";
export type { SplitSentencesOptions } from "./text/segment.js";

export { MemoryVoiceStore } from "./voice/memory-store.js";
export { IndexedDbVoiceStore } from "./voice/indexeddb-store.js";
export type { IndexedDbVoiceStoreOptions } from "./voice/indexeddb-store.js";
export type { VoiceEmbedding, VoiceStore } from "./voice/types.js";
