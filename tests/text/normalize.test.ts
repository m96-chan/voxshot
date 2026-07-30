import { describe, expect, it } from "vitest";

import { normalizeText } from "../../src/text/normalize.js";

describe("normalizeText", () => {
  it("trims and collapses whitespace", () => {
    expect(normalizeText("  hello   world \n")).toBe("hello world");
  });

  it("collapses newlines and tabs into single spaces", () => {
    expect(normalizeText("hello\n\n\tworld")).toBe("hello world");
  });

  it("applies NFKC so full width characters become canonical", () => {
    expect(normalizeText("ＶｏｘＳｈｏｔ１２３")).toBe("VoxShot123");
  });

  it("converts half width katakana to full width via NFKC", () => {
    expect(normalizeText("ｵﾝｾｲｺﾞｳｾｲ")).toBe("オンセイゴウセイ");
  });

  it("removes control characters", () => {
    expect(normalizeText("he\u0000l\u0007lo")).toBe("hello");
  });

  it("keeps Japanese punctuation intact", () => {
    expect(normalizeText("こんにちは、世界。")).toBe("こんにちは、世界。");
  });

  it("returns an empty string for whitespace only input", () => {
    expect(normalizeText("   \n\t ")).toBe("");
  });
});
