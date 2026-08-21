import type { ResolvedDevice } from "../../device.js";
import { DeviceUnavailableError, InvalidInputError, VoxShotError } from "../../errors.js";
import type { PcmAudio } from "../../platform.js";
import type { SynthesisEngine, SynthesisRequest } from "../types.js";
import { MIOCODEC_24K, Weights, cpuBackend, decode, type Backend } from "./codec/decoder.js";
import { encodeGlobal } from "./codec/encoder.js";
import { Gpu, gpuBackend } from "./codec/gpu.js";
import { Safetensors } from "./codec/safetensors.js";
import { maxNewFor } from "./constants.js";
import { createGpuEngine, type GpuEngine } from "./lm/gpu-engine.js";
import { createSamplerStats, sampleNextTopK, xorshift32 } from "./lm/sampler.js";
import { loadWeightsQ8, type WeightsQ8Manifest } from "./lm/weights-q8.js";
import {
  MIOTTS_ENGINE_NAME,
  MIOTTS_SAMPLE_RATE,
  assertUsableVoice,
  resolveEngineOptions,
  resolveGeneration,
  type MioTtsEngineOptions,
  type ResolvedMioTtsOptions,
} from "./options.js";
import { normalizeText } from "./text.js";
import { loadTokenizer, speechIndexOf, type Tokenizer, type TokenizerJson } from "./tokenizer.js";
import { loadPart, readJsonPart } from "./weights.js";

/**
 * MioTTS behind {@link SynthesisEngine}: text in, 24 kHz speech out, entirely
 * on the caller's GPU.
 *
 * The pipeline is two models. A Qwen3-0.6B language model (int8, resident on
 * the GPU) turns text into `<|s_n|>` speech tokens at a fixed 25 Hz, and
 * MioCodec's decoder turns those into audio, conditioned on a 128-dimension
 * speaker embedding. The language model never sees the voice — speaker
 * identity enters only at the decoder — which is why the same text produces
 * the same tokens whoever is speaking.
 *
 * Nothing in this file, or anything it imports, touches the network, a cache
 * or a filesystem. Weights arrive through {@link MioTtsWeightSource}.
 */

/** The ids that end a generation: `<|im_end|>` and `<|endoftext|>`. */
const EOS_IDS = new Set([151645, 151643]);

/**
 * How much of a reference clip reaches the encoder.
 *
 * The MioTTS reference server's own `max_reference_seconds`. Not a nicety:
 * WavLM's attention is O(T²), so a three-minute clip means ~3.9 GB of scratch
 * — a dead tab rather than a slow one. `encodeGlobal` carries its own 30 s
 * hard cap for any other caller; this is the engine matching the server it is
 * a port of, and the difference is visible in `lastReferenceSeconds`.
 */
const MAX_REFERENCE_SECONDS = 20;

/** How the engine gets a GPU device when the caller does not supply one. */
export type GpuDeviceRequest = () => Promise<{ device: GPUDevice; adapterInfo: string }>;

/**
 * The browser's own device request.
 *
 * The one place a WebGPU global is read, and it is injectable — the
 * architecture rule is that browser APIs are reached through a seam a test can
 * replace, and a caller who already owns a device passes `device` instead and
 * never reaches this at all.
 */
export const requestBrowserGpuDevice: GpuDeviceRequest = async () => {
  const gpu = (globalThis.navigator as Navigator | undefined)?.gpu;
  if (!gpu) {
    throw new DeviceUnavailableError(
      "webgpu",
      "This environment has no navigator.gpu. The MioTTS engine has no CPU path: the model is " +
        "far too slow to run without a GPU, so it refuses rather than appearing to work.",
    );
  }
  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    throw new DeviceUnavailableError("webgpu", "requestAdapter() returned no WebGPU adapter.");
  }
  // At the adapter's own limits: the packed embedding table alone is ~168 MiB
  // of storage and the largest lm_head chunk binding is ~67 MiB, past some
  // defaults.
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  const info = adapter.info
    ? [adapter.info.vendor, adapter.info.architecture, adapter.info.description]
        .filter(Boolean)
        .join(" ") || "unknown adapter"
    : "unknown adapter";
  return { device, adapterInfo: info };
};

/** Everything `load()` builds, kept together so `dispose` can drop it in one go. */
interface Loaded {
  lm: GpuEngine;
  tokenizer: Tokenizer;
  codecWeights: Weights;
  codecBackend: Backend;
  gpu: Gpu | null;
  /** Only built if the caller supplied the device; ours is ours to destroy. */
  ownsDevice: boolean;
  device: GPUDevice;
}

export class MioTtsEngine implements SynthesisEngine {
  readonly name = MIOTTS_ENGINE_NAME;
  readonly sampleRate = MIOTTS_SAMPLE_RATE;

  /**
   * There is no CPU path, and that is a decision rather than an omission. The
   * language model streams ~600 MiB of weights per token; on a CPU a sentence
   * would take minutes. `VoxShot.create` checks this *before* `load()`, so an
   * unsuitable machine finds out before paying for 1.1 GB of weights.
   */
  readonly requiresGpu = true;

  #options: ResolvedMioTtsOptions;
  #gpuDevice: GPUDevice;
  #adapterInfo: string;
  #ownsDevice: boolean;
  #loaded: Loaded | null = null;
  #device: ResolvedDevice | undefined;
  #encoderWeights: Promise<Weights> | null = null;
  #lastReferenceSeconds = 0;
  #referenceTrimmed = false;

  /**
   * Takes a device rather than finding one, so that the "can this environment
   * run at all" question is settled by {@link createMioTtsEngine} before a
   * caller has arranged a gigabyte of weights.
   */
  constructor(
    options: MioTtsEngineOptions,
    gpu: { device: GPUDevice; adapterInfo: string; owned: boolean },
  ) {
    this.#options = resolveEngineOptions(options);
    this.#gpuDevice = gpu.device;
    this.#adapterInfo = gpu.adapterInfo;
    this.#ownsDevice = gpu.owned;
  }

  /** Which adapter the models are resident on, for logs and reports. */
  get adapterInfo(): string {
    return this.#adapterInfo;
  }

  get loadedDevice(): ResolvedDevice | undefined {
    return this.#device;
  }

  /** Seconds of the reference clip the last `embed()` actually encoded. */
  get lastReferenceSeconds(): number {
    return this.#lastReferenceSeconds;
  }

  /** Whether the last `embed()` cut the clip down to {@link MAX_REFERENCE_SECONDS}. */
  get referenceWasTrimmed(): boolean {
    return this.#referenceTrimmed;
  }

  async load(device: ResolvedDevice): Promise<void> {
    if (this.#loaded) return;
    if (device !== "webgpu") {
      throw new DeviceUnavailableError(
        "webgpu",
        `The "${this.name}" engine requires a GPU and was asked to load on "${device}".`,
      );
    }

    const gpuDevice = this.#gpuDevice;
    const source = this.#options.weights;

    const tokenizerJson = await readJsonPart<TokenizerJson>(source, "tokenizer");
    const tokenizer = await loadTokenizer(tokenizerJson);

    const manifest = await readJsonPart<WeightsQ8Manifest>(source, "lm-manifest");
    const [codes, scales, norms] = await Promise.all([
      loadPart(source, "lm-codes"),
      loadPart(source, "lm-scales"),
      loadPart(source, "lm-norms"),
    ]);
    // sha256 is deliberately not requested: the hasher is synchronous and a
    // ~600 MiB digest costs seconds on the load path. Byte lengths are still
    // checked against the manifest, which catches a truncated transfer.
    const weights = loadWeightsQ8({ manifest, codes, scales, norms });
    const lm = await createGpuEngine(gpuDevice, weights, { maxSeqLen: this.#options.maxSeqLen });

    const checkpoint = await loadPart(source, "codec-decoder");
    const codecWeights = new Weights(Safetensors.parse(checkpoint));

    // The codec shares the language model's device — one device, both models
    // resident, and no second `requestDevice`. `cpuBackend` is the fallback
    // shape and is unreachable here: without a GPU we threw above.
    const gpu = Gpu.fromDevice(gpuDevice, this.#adapterInfo);
    this.#loaded = {
      lm,
      tokenizer,
      codecWeights,
      codecBackend: gpuBackend(gpu) ?? cpuBackend,
      gpu,
      ownsDevice: this.#ownsDevice,
      device: gpuDevice,
    };
    this.#device = "webgpu";
  }

  async embed(audio: PcmAudio): Promise<Float32Array> {
    const loaded = this.#assertLoaded();
    if (audio.sampleRate !== MIOTTS_SAMPLE_RATE) {
      throw new InvalidInputError(
        `Reference audio must be ${MIOTTS_SAMPLE_RATE} Hz; got ${audio.sampleRate}.`,
      );
    }

    // Fetched on the first clone and never before: a caller who synthesises
    // with an embedding they already hold never pays for the encoder's 117 MB.
    // A failure drops the cache so the next attempt retries rather than
    // re-awaiting a permanently rejected promise.
    this.#encoderWeights ??= (async () => {
      const bytes = await loadPart(this.#options.weights, "codec-encoder");
      return new Weights(Safetensors.parse(bytes));
    })().catch((error: unknown) => {
      this.#encoderWeights = null;
      throw error;
    });
    const encoder = await this.#encoderWeights;

    const limit = MAX_REFERENCE_SECONDS * MIOTTS_SAMPLE_RATE;
    this.#referenceTrimmed = audio.samples.length > limit;
    const wave = this.#referenceTrimmed ? audio.samples.slice(0, limit) : audio.samples;
    this.#lastReferenceSeconds = wave.length / MIOTTS_SAMPLE_RATE;

    return await encodeGlobal(wave, encoder, { backend: loaded.codecBackend });
  }

  async synthesize(request: SynthesisRequest): Promise<Float32Array> {
    const loaded = this.#assertLoaded();
    assertUsableVoice(request.voice);
    const generation = resolveGeneration(request, this.#options);

    // Above everything that costs something: an abandoned request should not
    // pay for a prefill nobody is waiting for.
    request.signal?.throwIfAborted();

    // The reference server's own `normalize_text`, on top of whatever
    // canonicalisation VoxShot already did. Not a second guess at the same
    // job: this is the preprocessing the model was served behind, and text it
    // never saw in training is text it renders worse.
    const text = normalizeText(request.text);
    if (text.trim().length === 0) return new Float32Array(0);

    const promptIds = loaded.tokenizer.encodeChat(text);
    if (promptIds.length + 1 >= this.#options.maxSeqLen) {
      throw new InvalidInputError(
        `This chunk is ${promptIds.length} tokens and the engine was built for maxSeqLen=` +
          `${this.#options.maxSeqLen}. Lower VoxShot's maxChunkLength, or build the engine with ` +
          "a larger maxSeqLen.",
      );
    }

    const rng = xorshift32(generation.seed);
    const sampler = {
      mode: "top-p",
      temperature: generation.temperature,
      topP: generation.topP,
      rng,
    } as const;
    const stats = createSamplerStats();

    loaded.lm.reset();

    // Prefill is the decode path one token at a time — a ~15-token prompt does
    // not earn a batched path. Only the last token's logits matter, so every
    // earlier step skips the 164k-row projection and its 657 KB readback.
    let logits: Float32Array | null = null;
    for (let i = 0; i < promptIds.length; i += 1) {
      logits = await loaded.lm.decodeStep(promptIds[i]!, {
        skipLogits: i + 1 < promptIds.length,
      });
    }

    const maxNew = maxNewFor(promptIds.length, generation.maxNewTokens, this.#options.maxSeqLen);
    const speechIndices: number[] = [];
    for (let step = 0; step < maxNew; step += 1) {
      // Checked every step: a generation is up to 700 of these, and a caller
      // who has stopped listening should not wait for all of them.
      request.signal?.throwIfAborted();

      const id = sampleNextTopK(logits!, [], sampler, { stats });
      if (EOS_IDS.has(id)) break;
      const index = speechIndexOf(id);
      if (index !== null) speechIndices.push(index);
      if (step + 1 < maxNew) {
        logits = await loaded.lm.decodeStep(id);
      }
    }

    if (speechIndices.length === 0) {
      throw new VoxShotError(
        "The language model produced no speech tokens for this text.",
        "UNKNOWN",
      );
    }

    const { waveform } = await decode(
      Float32Array.from(speechIndices),
      request.voice.vector,
      // The aligned path: two STFT frames per 25 Hz token, so `n` tokens is
      // `n * 960` samples at 24 kHz.
      2 * speechIndices.length,
      MIOCODEC_24K,
      loaded.codecWeights,
      loaded.codecBackend,
    );
    return waveform;
  }

  async dispose(): Promise<void> {
    const loaded = this.#loaded;
    this.#loaded = null;
    this.#device = undefined;
    this.#encoderWeights = null;
    if (!loaded) return;

    loaded.lm.destroy();
    // Only a device this engine requested is a device this engine may destroy.
    // A caller who supplied one is still using it for something else.
    if (loaded.ownsDevice) loaded.gpu?.destroy();
  }

  #assertLoaded(): Loaded {
    if (!this.#loaded) {
      throw new VoxShotError(
        `The "${this.name}" engine has not been loaded. VoxShot.create() does this; a direct ` +
          "caller must await load() first.",
        "UNKNOWN",
      );
    }
    return this.#loaded;
  }
}

/**
 * Build a MioTTS engine, failing here rather than mid-synthesis when the
 * environment cannot run it.
 *
 * ```ts
 * import { VoxShot } from "voxshot";
 * import { createMioTtsEngine } from "voxshot/miotts";
 *
 * const engine = await createMioTtsEngine({
 *   weights: { load: async (part) => (await fetch(`/models/${part}`)).arrayBuffer() },
 * });
 * const tts = await VoxShot.create({ engine, device: "webgpu" });
 * ```
 *
 * `speed` does not work. MioTTS generates speech tokens at a fixed 25 Hz and
 * has no pace control, so {@link SynthesisRequest.speed} is ignored rather
 * than approximated by resampling — which would shift the pitch of a voice the
 * caller cloned specifically to keep. {@link SynthesisRequest.expressiveness}
 * maps onto the sampling temperature, which is the one delivery control this
 * model has.
 *
 * Chunks are generated independently, so prosody does not carry across a
 * sentence boundary the way a single long utterance would.
 */
export async function createMioTtsEngine(
  options: MioTtsEngineOptions & { requestDevice?: GpuDeviceRequest },
): Promise<MioTtsEngine> {
  // The gate, and the reason this factory is async at all: an environment that
  // cannot run the model is told here, while it still costs nothing, instead
  // of part-way through a load a caller has already arranged 1.1 GB for.
  //
  // Acquiring the device is what asks the question — there is no separate
  // probe that could answer differently from the thing it is standing in for.
  const gpu = options.device
    ? // A caller who supplies a device has already answered it, and owns the
      // device's lifetime: `dispose()` will not destroy it.
      { device: options.device, adapterInfo: "caller-supplied device", owned: false }
    : { ...(await (options.requestDevice ?? requestBrowserGpuDevice)()), owned: true };

  return new MioTtsEngine(options, gpu);
}
