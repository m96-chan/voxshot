import { describe, expect, it } from "vitest";

import { buildLoadPlans } from "../../../src/engine/chatterbox/dtype-plan.js";

describe("buildLoadPlans", () => {
  it("starts on WebGPU with the fp16 quantized language model", () => {
    const [first] = buildLoadPlans("webgpu");

    expect(first?.device).toBe("webgpu");
    expect(first?.dtype).toEqual({
      embed_tokens: "fp32",
      speech_encoder: "fp32",
      language_model: "q4f16",
      model: "q4f16",
      conditional_decoder: "fp32",
    });
  });

  it("falls back from fp16 to fp32 weights on the same device", () => {
    const plans = buildLoadPlans("webgpu");

    expect(plans[1]?.device).toBe("webgpu");
    expect(plans[1]?.dtype.language_model).toBe("q4");
  });

  it("ends on wasm so an unsupported GPU still loads", () => {
    const plans = buildLoadPlans("webgpu");
    const last = plans[plans.length - 1];

    expect(last?.device).toBe("wasm");
    expect(last?.dtype.language_model).toBe("q4");
  });

  it("never proposes webgpu when wasm was requested", () => {
    const plans = buildLoadPlans("wasm");

    expect(plans.every((plan) => plan.device === "wasm")).toBe(true);
    expect(plans).toHaveLength(1);
  });

  it("applies a dtype override to every plan", () => {
    const plans = buildLoadPlans("webgpu", { language_model: "fp32" });

    expect(plans.every((plan) => plan.dtype.language_model === "fp32")).toBe(true);
  });

  it("keeps the non overridden sessions untouched", () => {
    const [first] = buildLoadPlans("wasm", { conditional_decoder: "q8" });

    expect(first?.dtype).toEqual({
      embed_tokens: "fp32",
      speech_encoder: "fp32",
      language_model: "q4",
      model: "q4",
      conditional_decoder: "q8",
    });
  });

  it("skips f16 plans when the device cannot run f16 shaders", () => {
    const plans = buildLoadPlans("webgpu", undefined, false);

    expect(plans.map((plan) => `${plan.device}:${plan.dtype.language_model}`)).toEqual([
      "webgpu:q4",
      "wasm:q4",
    ]);
  });

  it("keeps f16 plans when f16 support is unknown", () => {
    expect(buildLoadPlans("webgpu")[0]?.dtype.language_model).toBe("q4f16");
    expect(buildLoadPlans("webgpu", undefined, true)[0]?.dtype.language_model).toBe("q4f16");
  });

  it("ignores the f16 flag on wasm, which never ran f16 plans", () => {
    const plans = buildLoadPlans("wasm", undefined, false);

    expect(plans).toHaveLength(1);
    expect(plans[0]?.dtype.language_model).toBe("q4");
  });
});

describe("an engine that requires a GPU", () => {
  it("drops the wasm tail", () => {
    // Not a preference: this pipeline measures ~5.8x slower than real time on
    // CPU, and it arrives with no error after a ~1.5 GB download. An engine
    // that knows that should be able to say so.
    for (const fp16 of [true, false]) {
      const plans = buildLoadPlans("webgpu", undefined, fp16, false);
      expect(plans.map((plan) => plan.device)).not.toContain("wasm");
      expect(plans.length).toBeGreaterThan(0);
    }
  });

  it("keeps degrading dtype within webgpu", () => {
    // Dropping wasm must not also drop the q4f16 -> q4 step, which is the
    // fallback that actually works.
    expect(
      buildLoadPlans("webgpu", undefined, true, false).map((p) => p.dtype.language_model),
    ).toEqual(["q4f16", "q4"]);
  });

  it("leaves the tail alone for an engine that did not ask", () => {
    const plans = buildLoadPlans("webgpu");
    expect(plans[plans.length - 1]?.device).toBe("wasm");
  });
});

describe("the language model's dtype key", () => {
  it("is carried under the session key as well as the file name", () => {
    // Transformers.js reads dtype by **session key** when it builds the list of
    // files a load will touch: `get_model_files` iterates the session map and
    // calls `selectDtype(dtype, sessionKey, device)`, and Chatterbox maps
    // `model -> language_model`. With only a `language_model` key, that lookup
    // misses and falls through to the device default. Verified against the
    // vendored 4.2.0:
    //
    //   without `model`  webgpu -> onnx/language_model.onnx           (fp32)
    //                    wasm   -> onnx/language_model_quantized.onnx (404)
    //   with `model`     both   -> onnx/language_model_q4.onnx
    //
    // The fp32 file is never fetched but is seeded into `progress_total`, which
    // is what inflates the download denominator (#55).
    for (const plan of [...buildLoadPlans("webgpu"), ...buildLoadPlans("wasm")]) {
      expect(plan.dtype.model).toBe(plan.dtype.language_model);
    }
  });

  it("keeps the file-name key, which builds the actual session", () => {
    // `session.js` resolves the session by file name (`names[name]`), so this
    // has to be an addition. Renaming would regress #11 — the load itself would
    // stop finding its quantization.
    for (const plan of buildLoadPlans("webgpu")) {
      expect(plan.dtype.language_model).toBeDefined();
    }
  });

  it("lets an override reach both", () => {
    const [plan] = buildLoadPlans("webgpu", { language_model: "q8" });

    expect(plan?.dtype.language_model).toBe("q8");
    expect(plan?.dtype.model).toBe("q8");
  });
});
