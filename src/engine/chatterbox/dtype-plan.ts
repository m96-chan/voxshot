import type { ResolvedDevice } from "../../device.js";

/** The four ONNX sessions a Chatterbox model is split into. */
export type ChatterboxSession = "embed_tokens" | "speech_encoder" | "model" | "conditional_decoder";

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
function plan(device: ResolvedDevice, model: string): LoadPlan {
  return {
    device,
    dtype: {
      embed_tokens: "fp32",
      speech_encoder: "fp32",
      model,
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
 */
export function buildLoadPlans(
  device: ResolvedDevice,
  overrides?: Partial<DtypeConfig>,
): LoadPlan[] {
  const plans =
    device === "webgpu"
      ? [plan("webgpu", "q4f16"), plan("webgpu", "q4"), plan("wasm", "q4")]
      : [plan("wasm", "q4")];

  if (!overrides) {
    return plans;
  }
  return plans.map((entry) => ({ device: entry.device, dtype: { ...entry.dtype, ...overrides } }));
}
