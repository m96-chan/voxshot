import { describe, expect, it } from "vitest";

import { splitSentences } from "../../src/text/segment.js";

describe("splitSentences", () => {
  it("returns an empty array for empty input", () => {
    expect(splitSentences("")).toEqual([]);
    expect(splitSentences("    ")).toEqual([]);
  });

  it("returns a single sentence when there is no terminator", () => {
    expect(splitSentences("hello world")).toEqual(["hello world"]);
  });

  it("splits on western terminators and keeps them", () => {
    expect(splitSentences("Hi there. How are you? Great!")).toEqual([
      "Hi there.",
      "How are you?",
      "Great!",
    ]);
  });

  it("splits on Japanese terminators and keeps them", () => {
    // NFKC normalization folds full width ？！ into their ASCII equivalents,
    // while 。 and 、 stay as they are.
    expect(splitSentences("こんにちは。元気ですか？はい！")).toEqual([
      "こんにちは。",
      "元気ですか?",
      "はい!",
    ]);
  });

  it("keeps consecutive terminators with the sentence", () => {
    expect(splitSentences("Really?! Yes...")).toEqual(["Really?!", "Yes..."]);
  });

  it("does not split a decimal point between digits", () => {
    expect(splitSentences("Pi is 3.14 exactly. Really.")).toEqual([
      "Pi is 3.14 exactly.",
      "Really.",
    ]);
  });

  it("treats a leading period as a terminator", () => {
    expect(splitSentences(".5 apples")).toEqual([".", "5 apples"]);
  });

  it("treats a trailing period after a digit as a terminator", () => {
    expect(splitSentences("Volume 3. Next")).toEqual(["Volume 3.", "Next"]);
  });

  it("handles an over-long sentence that ends with a comma", () => {
    expect(splitSentences("aaaa, bbbb,", { maxLength: 6 })).toEqual(["aaaa,", "bbbb,"]);
  });

  it("treats newlines as sentence boundaries", () => {
    expect(splitSentences("first line\nsecond line")).toEqual(["first line", "second line"]);
  });

  it("normalizes each sentence", () => {
    expect(splitSentences("  Hello    world.   Bye.  ")).toEqual(["Hello world.", "Bye."]);
  });

  it("splits over-long sentences at comma boundaries", () => {
    const text = "aaaa, bbbb, cccc, dddd";

    expect(splitSentences(text, { maxLength: 12 })).toEqual(["aaaa, bbbb,", "cccc, dddd"]);
  });

  it("splits over-long sentences at Japanese comma boundaries", () => {
    const text = "あああ、いいい、ううう、えええ";

    expect(splitSentences(text, { maxLength: 8 })).toEqual(["あああ、いいい、", "ううう、えええ"]);
  });

  it("hard splits when there is no break opportunity", () => {
    expect(splitSentences("abcdefghij", { maxLength: 4 })).toEqual(["abcd", "efgh", "ij"]);
  });

  it("never returns a chunk longer than maxLength", () => {
    const text = "これは非常に長い文章です、区切りが少ないためハードスプリットされます。";
    const chunks = splitSentences(text, { maxLength: 10 });

    expect(chunks.length).toBeGreaterThan(1);
    for (const chunk of chunks) {
      expect(chunk.length).toBeLessThanOrEqual(10);
    }
    expect(chunks.join("")).toBe(text);
  });

  it("rejects a non positive maxLength", () => {
    expect(() => splitSentences("hello", { maxLength: 0 })).toThrow(/maxLength/);
  });
});
