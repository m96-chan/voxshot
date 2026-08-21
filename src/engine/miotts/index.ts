/**
 * `voxshot/miotts` — the MioTTS engine and the contract for feeding it weights.
 *
 * A subpath rather than part of the main entry, because the WGSL kernels this
 * engine dispatches are large string literals: a consumer running Chatterbox,
 * or bringing their own engine, should not carry them in their bundle.
 *
 * Two things a caller needs before reaching for this:
 *
 * **The weights are yours to fetch.** Nothing under here touches the network,
 * a cache or a filesystem. You answer {@link MioTtsWeightSource.load} with the
 * bytes of a part, from wherever suits your application, and the engine asks
 * for one part at a time so a caller who never clones a voice never pays for
 * the encoder. `examples/` carries a reference implementation.
 *
 * **`speed` does not work.** MioTTS generates speech tokens at a fixed 25 Hz
 * and has no pace control, so {@link SynthesisRequest.speed} is ignored rather
 * than approximated. {@link SynthesisRequest.expressiveness} maps onto the
 * sampling temperature, which is the one delivery control this model has.
 */

export { MioTtsEngine, createMioTtsEngine, requestBrowserGpuDevice } from "./engine.js";
export type { GpuDeviceRequest } from "./engine.js";

export {
  MIOTTS_ENGINE_NAME,
  MIOTTS_SAMPLE_RATE,
  SPEAKER_EMBEDDING_SIZE,
} from "./options.js";
export type { MioTtsEngineOptions } from "./options.js";

export { MIOTTS_WEIGHT_PARTS } from "./weights.js";
export type { MioTtsWeightPart, MioTtsWeightSource } from "./weights.js";
