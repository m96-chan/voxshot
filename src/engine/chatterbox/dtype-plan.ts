import type { ResolvedDevice } from "../../device.js";

/**
 * The ONNX sessions a Chatterbox model is split into.
 *
 * Transformers.js reads `dtype` under **two different keys for the same
 * session**, and the language model is the one where they differ — Chatterbox
 * maps the session key `model` to the file `language_model`:
 *
 * - `session.js` builds the session by **file name**, so `language_model` is
 *   what makes the right weights load.
 * - `get_model_files` builds the list of files a load will touch by **session
 *   key**, calling `selectDtype(dtype, "model", device)`. Miss that and it
 *   falls through to the device default.
 *
 * Verified against the vendored 4.2.0:
 *
 * ```
 * without `model`   webgpu -> onnx/language_model.onnx           (fp32, 2.08 GB)
 *                   wasm   -> onnx/language_model_quantized.onnx (404)
 * with `model`      both   -> onnx/language_model_q4.onnx
 * ```
 *
 * The fp32 file is never fetched, but it is seeded into `progress_total`,
 * which is what inflates the download denominator (#55). So both keys are
 * carried, and `model` always mirrors `language_model` — one session, two
 * names, never allowed to drift (#62).
 */
export type ChatterboxSession =
  | "embed_tokens"
  | "speech_encoder"
  | "language_model"
  | "model"
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
      // Same session as `language_model`, under the key the expected-file list
      // uses. See ChatterboxSession.
      model: languageModel,
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
  return plans.map((entry) => {
    const dtype = { ...entry.dtype, ...overrides };
    // Kept in step after overrides too: a caller who quantizes the language
    // model should not have to know it is addressed twice.
    return { device: entry.device, dtype: { ...dtype, model: dtype.language_model } };
  });
}
