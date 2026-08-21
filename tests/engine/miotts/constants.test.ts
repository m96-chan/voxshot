import { describe, expect, it } from "vitest";

import { MAX_NEW_TOKENS, MAX_SEQ_LEN, maxNewFor } from "../../../src/engine/miotts/constants.js";

describe("shared generation cap", () => {
  it("pins the reference server's limits", () => {
    expect(MAX_NEW_TOKENS).toBe(700);
    expect(MAX_SEQ_LEN).toBe(768);
  });

  it("caps at MAX_NEW_TOKENS for short prompts", () => {
    expect(maxNewFor(15)).toBe(700);
  });

  it("caps at the sequence budget for long prompts", () => {
    expect(maxNewFor(100)).toBe(668);
    expect(maxNewFor(768)).toBe(0);
  });

  it("respects an explicit lower request", () => {
    expect(maxNewFor(15, 128)).toBe(128);
  });
});
