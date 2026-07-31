import type { ResolvedDevice } from "../../device.js";

/**
 * The four ONNX sessions a Chatterbox model is split into, named by their
 * *file* names: Transformers.js resolves per-session `dtype` entries against
 * the ONNX file name (`language_model`), not the session key (`model`) — a
 * mismatched key is silently ignored and the device default (fp32 on WebGPU,
 * 2 GB for the language model) loads instead.
 */
export type ChatterboxSession =
  | "embed_tokens"
  | "speech_encoder"
  | "language_model"
  | "conditional_decoder";

/** Per-session quantization, as accepted by `from_pretrained({ dtype })`. */
export type DtypeConfig = Record<ChatterboxSession, string>;

/** One attempt at loading the model: a device plus a quantization choice. */
export interface LoadPlan {
  readonly device: ResolvedDevice;
  readonly dtype: DtypeConfig;
}

/**
 * Only the language model is quantized. The encoder and decoder stay at fp32
 * because that is what the reference browser demo ships
 * (embed_tokens / speech_encoder / conditional_decoder fp32, language model
 * q4f16 on WebGPU and q4 on WASM).
 */
function plan(device: ResolvedDevice, languageModel: string): LoadPlan {
  return {
    device,
    dtype: {
      embed_tokens: "fp32",
      speech_encoder: "fp32",
      language_model: languageModel,
      conditional_decoder: "fp32",
    },
  };
}

/**
 * Build the ordered list of load attempts for a device.
 *
 * fp16 support varies by GPU, so a WebGPU request degrades to integer-only
 * weights before giving up on the GPU entirely and landing on WASM. A WASM
 * request has nothing to fall back to and yields a single plan.
 *
 * Pass `fp16: false` when the adapter lacks `shader-f16`: an f16 plan on such
 * a device *loads* fine and only fails at the first inference, which the
 * load-time fallback can no longer catch.
 */
export function buildLoadPlans(
  device: ResolvedDevice,
  overrides?: Partial<DtypeConfig>,
  fp16 = true,
  allowWasm = true,
): LoadPlan[] {
  // The wasm tail exists so a GPU that cannot run the model still produces
  // audio. Whether that is a kindness or a trap is the engine's call: this
  // pipeline measures ~5.8x slower than real time there, which is not a
  // degraded experience. An engine that says so drops the tail (#107).
  const webgpu = [...(fp16 ? [plan("webgpu", "q4f16")] : []), plan("webgpu", "q4")];
  const plans =
    device === "webgpu"
      ? [...webgpu, ...(allowWasm ? [plan("wasm", "q4")] : [])]
      : [plan("wasm", "q4")];

  if (!overrides) {
    return plans;
  }
  return plans.map((entry) => ({ device: entry.device, dtype: { ...entry.dtype, ...overrides } }));
}
