import { beforeAll, describe, expect, it } from "vitest";

import {
  ENDOFTEXT_ID,
  IM_END_ID,
  IM_START_ID,
  PRE_TOKENIZE,
  SPEECH_TOKEN_BASE,
  SPEECH_TOKEN_COUNT,
  loadTokenizer,
  speechIndexOf,
  speechTokenId,
  type Tokenizer,
  type TokenizerJson,
} from "../../../src/engine/miotts/tokenizer.js";

/**
 * The BPE machinery, exercised against a hand-built vocab rather than the
 * model's 13.8 MB `tokenizer.json`.
 *
 * The real file is not in git and CI cannot download it, so the tests that
 * check this port against the reference HF tokenizer's own output live in
 * `npm run test:models`. What is checked here is everything that does not need
 * the checkpoint: merge ranking, the byte-level map, added-token splitting, the
 * chat template's shape, and the speech-token arithmetic.
 *
 * The synthetic vocab spells its pieces out as literals — `"Ġ"` for a space,
 * `"Ċ"` for a newline — which is also what pins the GPT-2 byte map: if
 * `encode` mapped a space to anything else, the piece would not be in this
 * vocab and encoding would throw rather than quietly produce different ids.
 */

/** Every printable ASCII character maps to itself under the GPT-2 byte map. */
const ASCII_PIECES = Array.from({ length: 0x7e - 0x21 + 1 }, (_, i) =>
  String.fromCharCode(0x21 + i),
);

/** Single-byte pieces plus the three merge results the tests reach for. */
const PIECES = [...ASCII_PIECES, "Ġ", "Ċ", "bc", "ab", "abab"];

/**
 * Ranked lowest first. `b c` deliberately outranks `a b` so that "abc" can
 * only come out as `a` + `bc` if the merge loop really picks by rank.
 */
const MERGES: [string, string][] = [
  ["b", "c"],
  ["a", "b"],
  ["ab", "ab"],
];

function syntheticJson(overrides: Partial<TokenizerJson> = {}): TokenizerJson {
  const vocab: Record<string, number> = {};
  PIECES.forEach((piece, index) => {
    vocab[piece] = index;
  });
  return {
    added_tokens: [
      { id: ENDOFTEXT_ID, content: "<|endoftext|>" },
      { id: IM_START_ID, content: "<|im_start|>" },
      { id: IM_END_ID, content: "<|im_end|>" },
      { id: SPEECH_TOKEN_BASE, content: "<|s_0|>" },
      { id: SPEECH_TOKEN_BASE + 5, content: "<|s_5|>" },
    ],
    model: { vocab, merges: MERGES },
    ...overrides,
  };
}

const id = (piece: string): number => {
  const at = PIECES.indexOf(piece);
  if (at < 0) throw new Error(`the synthetic vocab has no piece ${JSON.stringify(piece)}`);
  return at;
};

const ids = (...pieces: string[]): number[] => pieces.map(id);

describe("BPE over a synthetic vocab", () => {
  let tokenizer: Tokenizer;

  beforeAll(async () => {
    tokenizer = await loadTokenizer(syntheticJson());
  });

  it("applies the lowest-ranked merge first", () => {
    // Both `a b` and `b c` are applicable to "abc"; only the rank decides.
    expect(tokenizer.encode("abc")).toEqual(ids("a", "bc"));
  });

  it("keeps merging the result of a merge", () => {
    // a b a b -> ab ab (both occurrences) -> abab.
    expect(tokenizer.encode("abab")).toEqual(ids("abab"));
  });

  it("stops when no adjacent pair has a rank", () => {
    expect(tokenizer.encode("xyz")).toEqual(ids("x", "y", "z"));
  });

  it("maps a space and a newline through the byte-level table", () => {
    expect(tokenizer.encode("a b")).toEqual(ids("a", "Ġ", "b"));
    expect(tokenizer.encode("a\nb")).toEqual(ids("a", "Ċ", "b"));
  });

  it("round-trips text through encode and decode", () => {
    for (const text of ["abc", "abab", "a b c", "x!y?z", "a\nb"]) {
      expect(tokenizer.decode(tokenizer.encode(text))).toBe(text);
    }
  });

  it("accepts the older space-separated merge spelling", async () => {
    const older = await loadTokenizer(
      syntheticJson({
        model: { vocab: syntheticJson().model.vocab, merges: MERGES.map(([a, b]) => `${a} ${b}`) },
      }),
    );

    expect(older.encode("abc")).toEqual(ids("a", "bc"));
  });

  it("refuses a merge line that does not split into exactly two pieces", async () => {
    const broken = syntheticJson({
      model: { vocab: syntheticJson().model.vocab, merges: ["a b c"] },
    });

    await expect(loadTokenizer(broken)).rejects.toThrow(/unsplittable merge/);
  });

  it("refuses to decode an id that is in neither the vocab nor the added tokens", () => {
    expect(() => tokenizer.decode([PIECES.length + 1])).toThrow(/not in vocab/);
  });
});

describe("added tokens", () => {
  let tokenizer: Tokenizer;

  beforeAll(async () => {
    tokenizer = await loadTokenizer(syntheticJson());
  });

  it("emits one id for an added token and BPEs the text around it", () => {
    expect(tokenizer.encode("a<|im_end|>b")).toEqual([id("a"), IM_END_ID, id("b")]);
  });

  it("recognises every <|s_n|> without listing all 12,800 of them", () => {
    // Only two are in `added_tokens` above; the rest are handled by the
    // regex branch, which is the whole point of that branch.
    expect(tokenizer.encode("<|s_0|>")).toEqual([SPEECH_TOKEN_BASE]);
    expect(tokenizer.encode("<|s_9999|>")).toEqual([SPEECH_TOKEN_BASE + 9999]);
  });

  it("treats a non-canonical or out-of-range <|s_n|> as ordinary text", () => {
    // "<|s_007|>" is not the token "<|s_7|>", and there is no <|s_12800|>.
    expect(tokenizer.encode("<|s_007|>")).not.toContain(SPEECH_TOKEN_BASE + 7);
    expect(tokenizer.encode(`<|s_${SPEECH_TOKEN_COUNT}|>`)).not.toContain(
      SPEECH_TOKEN_BASE + SPEECH_TOKEN_COUNT,
    );
  });

  it("still finds an added token that starts inside a rejected span", async () => {
    // Rejecting a `<|s_n|>` match rescans from the NEXT character rather than
    // from the end of the span, so an added token overlapping it is not lost.
    //
    // The vocab below is contrived on purpose, and that is worth recording: a
    // real rejected span is always `<|s_` + digits + `|>`, and every added
    // token Qwen ships starts with `<|`, which inside such a span only occurs
    // at offset 0. With the model's own tokenizer.json the rescan therefore
    // cannot change any result — this is the only shape that observes it, so
    // without it the line would be untested rather than merely untestable.
    const json = syntheticJson();
    json.added_tokens = [...json.added_tokens, { id: 4242, content: "5|>" }];
    const overlapping = await loadTokenizer(json);

    // "<|s_0075|>" matches the speech branch and is rejected (leading zero),
    // and "5|>" starts at offset 7 inside it.
    expect(overlapping.encode("<|s_0075|>")).toContain(4242);
  });

  it("decodes speech ids and added tokens back to their literal forms", () => {
    expect(tokenizer.decode([SPEECH_TOKEN_BASE + 5, IM_END_ID])).toBe("<|s_5|><|im_end|>");
  });
});

describe("chat template", () => {
  let tokenizer: Tokenizer;

  beforeAll(async () => {
    tokenizer = await loadTokenizer(syntheticJson());
  });

  it("ends with the ids of '<|im_start|>assistant\\n'", () => {
    const tail = tokenizer.encode("<|im_start|>assistant\n");
    const chat = tokenizer.encodeChat("abc");

    expect(chat.slice(chat.length - tail.length)).toEqual(tail);
  });

  it("omits the generation prompt when asked", () => {
    const withPrompt = tokenizer.encodeChat("abc");
    const without = tokenizer.encodeChat("abc", { addGenerationPrompt: false });
    const tail = tokenizer.encode("<|im_start|>assistant\n");

    expect(withPrompt).toEqual([...without, ...tail]);
  });

  it("prepends a system turn when one is given", () => {
    const withSystem = tokenizer.encodeChat("abc", { system: "xyz" });
    const head = tokenizer.encode("<|im_start|>system\nxyz<|im_end|>\n");

    expect(withSystem.slice(0, head.length)).toEqual(head);
  });
});

describe("the Qwen2 pre-tokenizer regex", () => {
  const split = (text: string): string[] => [...text.matchAll(PRE_TOKENIZE)].map(([word]) => word);

  it("keeps contractions attached to the apostrophe", () => {
    expect(split("it's fine")).toEqual(["it", "'s", " fine"]);
  });

  it("folds ſ into the contraction group like the reference", () => {
    expect(split("x'ſy")).toEqual(["x", "'ſ", "y"]);
  });

  it("still stops uppercase contractions where the reference does", () => {
    expect(split("I'VEGOT")).toEqual(["I", "'VE", "GOT"]);
  });
});

describe("speech token arithmetic", () => {
  it("speechTokenId maps the codebook range onto contiguous ids", () => {
    expect(speechTokenId(0)).toBe(151669);
    expect(speechTokenId(12799)).toBe(164468);
  });

  it("speechTokenId rejects out-of-range indices", () => {
    expect(() => speechTokenId(-1)).toThrow();
    expect(() => speechTokenId(12800)).toThrow();
  });

  it("speechIndexOf inverts speechTokenId", () => {
    expect(speechIndexOf(151669)).toBe(0);
    expect(speechIndexOf(164468)).toBe(12799);
  });

  it("speechIndexOf returns null on non-speech ids", () => {
    expect(speechIndexOf(151643)).toBeNull(); // <|endoftext|>
    expect(speechIndexOf(151668)).toBeNull(); // </think>, one below the range
    expect(speechIndexOf(164469)).toBeNull(); // first dead padding row
    expect(speechIndexOf(0)).toBeNull();
  });

  it("exports the ChatML special ids", () => {
    expect(ENDOFTEXT_ID).toBe(151643);
    expect(IM_START_ID).toBe(151644);
    expect(IM_END_ID).toBe(151645);
    expect(SPEECH_TOKEN_BASE).toBe(151669);
    expect(SPEECH_TOKEN_COUNT).toBe(12800);
  });
});
