export { VoxShot } from "./voxshot.js";
export type { PlayOptions, SpeakOptions, VoiceSource, VoxShotOptions } from "./voxshot.js";

export { SynthesizedAudio } from "./audio/synthesized-audio.js";
export { normalizePeak, resample, toMono, trimSilence } from "./audio/pcm.js";
export { encodeWav } from "./audio/wav.js";
export { startSpeechPlayback } from "./audio/speech-playback.js";
export type { SpeechPlayback, SpeechPlaybackInput } from "./audio/speech-playback.js";
export { SynthesisCache } from "./audio/synthesis-cache.js";
export type { SynthesisCacheOptions } from "./audio/synthesis-cache.js";

export { resolveDevice } from "./device.js";
export type { DevicePreference, ResolvedDevice } from "./device.js";

export { PlaceholderEngine, VOICE_EMBEDDING_SIZE } from "./engine/placeholder-engine.js";
export type { PlaceholderEngineOptions } from "./engine/placeholder-engine.js";
export type { EmbedResult, SynthesisEngine, SynthesisRequest } from "./engine/types.js";

export {
  CHATTERBOX_SAMPLE_RATE,
  ChatterboxEngine,
} from "./engine/chatterbox/chatterbox-engine.js";
export type {
  ChatterboxEngineOptions,
  ChatterboxLifecycleEvent,
  ChatterboxLoadEvent,
} from "./engine/chatterbox/chatterbox-engine.js";
export { buildLoadPlans } from "./engine/chatterbox/dtype-plan.js";
export type {
  ChatterboxSession,
  DtypeConfig,
  LoadPlan,
} from "./engine/chatterbox/dtype-plan.js";
export type {
  LoadProgress,
  TransformersModule,
  TransformersModuleLoader,
} from "./engine/chatterbox/transformers-module.js";

export { WorkerSynthesisEngine } from "./worker/worker-engine.js";
export type { WorkerSynthesisEngineOptions } from "./worker/worker-engine.js";
export { exposeEngine } from "./worker/expose.js";
export { PROTOCOL_VERSION } from "./worker/protocol.js";
export type { RpcEndpoint } from "./worker/protocol.js";

export {
  AudioDecodeError,
  DeviceUnavailableError,
  DisposedError,
  InvalidInputError,
  LoadStalledError,
  NoVoiceError,
  VoiceNotFoundError,
  VoxShotError,
  isVoxShotError,
} from "./errors.js";
export type { VoxShotErrorCode } from "./errors.js";

export {
  BrowserAudioDecoder,
  BrowserAudioPlayer,
  BrowserGpuProbe,
  BrowserStreamingAudioPlayer,
  createBrowserPlatform,
} from "./browser-platform.js";
export type {
  AudioDecoder,
  AudioPlayer,
  DecodedAudio,
  GpuProbe,
  PcmAudio,
  Platform,
  StreamingAudioPlayer,
  StreamingPlayback,
} from "./platform.js";

export { normalizeText } from "./text/normalize.js";
export { toJapaneseReading } from "./text/japanese.js";
export { splitSentences } from "./text/segment.js";
export type { SplitSentencesOptions } from "./text/segment.js";

export { MemoryVoiceStore } from "./voice/memory-store.js";
export { IndexedDbVoiceStore } from "./voice/indexeddb-store.js";
export type { IndexedDbVoiceStoreOptions } from "./voice/indexeddb-store.js";
export type {
  VoiceEmbedding,
  VoiceStore,
  VoiceTensor,
  VoiceTensorType,
} from "./voice/types.js";
