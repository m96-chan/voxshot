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
