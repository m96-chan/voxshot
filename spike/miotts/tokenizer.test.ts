import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it } from "vitest";

import {
  ENDOFTEXT_ID,
  IM_END_ID,
  IM_START_ID,
  loadTokenizer,
  PRE_TOKENIZE,
  SPEECH_TOKEN_BASE,
  SPEECH_TOKEN_COUNT,
  speechIndexOf,
  speechTokenId,
  type Tokenizer,
} from "./tokenizer";

/**
 * Golden vectors from the reference HF tokenizer (`dump_tokenizer_vectors.py`).
 *
 * Not in git — regenerable — so a missing file has to fail loudly with the
 * rebuild command rather than letting the suite skip itself green.
 */

const HERE = dirname(fileURLToPath(import.meta.url));

const SNAPSHOT = join(
  process.env.HOME ?? "/home/m96-chan",
  ".cache/huggingface/hub/models--Aratako--MioTTS-0.6B/snapshots",
  "901ee12c50efd68ea3db6a6780cd29da197cd0da",
);

interface EncodeVector {
  text: string;
  ids: number[];
}
interface ChatVector {
  user: string;
  ids: number[];
}
interface DecodeVector {
  ids: number[];
  text: string;
}
interface Vectors {
  _rebuild: string;
  /** sha256 of the exact tokenizer.json the vectors were dumped from. */
  tokenizer_sha256: string;
  encode: EncodeVector[];
  chat: ChatVector[];
  decode: DecodeVector[];
}

function missing(what: string): never {
  throw new Error(
    `${what} is missing. The goldens are deliberately not in git; rebuild them with\n` +
      `  cd spike/miotts && python3 dump_tokenizer_vectors.py`,
  );
}

function loadVectors(): Vectors {
  try {
    return JSON.parse(
      readFileSync(join(HERE, "golden", "tokenizer_vectors.json"), "utf8"),
    ) as Vectors;
  } catch {
    missing("golden/tokenizer_vectors.json");
  }
}

const vectors = loadVectors();

let tokenizer: Tokenizer;

beforeAll(async () => {
  const json = JSON.parse(readFileSync(join(SNAPSHOT, "tokenizer.json"), "utf8"));
  tokenizer = await loadTokenizer(json);
});

describe("tokenizer.json pin", () => {
  it("the file under test is byte-identical to the one the vectors were dumped from", () => {
    // Every consumer resolves tokenizer.json independently (this test pins
    // the snapshot path, expected-tokens.ts and serve.mjs resolve refs/main)
    // — a silent drift between them would make the vectors test the wrong
    // file. Loud failure here means: rerun dump_tokenizer_vectors.py against
    // the file you actually want, and check the other resolvers.
    const bytes = readFileSync(join(SNAPSHOT, "tokenizer.json"));
    const sha = createHash("sha256").update(bytes).digest("hex");
    expect(sha).toBe(vectors.tokenizer_sha256);
  });
});

describe("encode against golden vectors", () => {
  for (const { text, ids } of vectors.encode) {
    it(`encodes ${JSON.stringify(text.length > 40 ? text.slice(0, 40) + "…" : text)}`, () => {
      expect(tokenizer.encode(text)).toEqual(ids);
    });
  }
});

describe("decode against golden vectors", () => {
  for (const { ids, text } of vectors.decode) {
    it(`decodes [${ids.slice(0, 6).join(",")}${ids.length > 6 ? ",…" : ""}]`, () => {
      expect(tokenizer.decode(ids)).toBe(text);
    });
  }
});

describe("chat template against golden vectors", () => {
  for (const { user, ids } of vectors.chat) {
    it(`encodeChat(${JSON.stringify(user)})`, () => {
      expect(tokenizer.encodeChat(user)).toEqual(ids);
    });
  }
});

describe("pre-tokenizer Unicode case folding (contraction group)", () => {
  // The reference's Rust `(?i:'s|...)` applies Unicode simple case folding,
  // so U+017F (ſ) folds to 's' and "'ſ" is a contraction piece. The split is
  // NOT observable in encode ids with this vocab — ſ's byte-mapped pieces
  // ("Å", "¿") merge only with each other, so no BPE merge ever crosses the
  // boundary — which is why this test pins the pre-tokenizer split itself.
  // Expected splits below are the reference tokenizer's own
  // `pre_tokenizer.pre_tokenize_str` output, measured 2026-08-21 against the
  // pinned snapshot (901ee12c…): "x'ſy" → ["x", "'ſ", "y"].
  const split = (text: string) => Array.from(text.matchAll(PRE_TOKENIZE), (m) => m[0]);

  it("folds ſ into the contraction group like the reference", () => {
    expect(split("x'ſy")).toEqual(["x", "'ſ", "y"]);
    expect(split("it'ſ fine")).toEqual(["it", "'ſ", " fine"]);
  });

  it("still stops uppercase contractions where the reference does", () => {
    expect(split("I'VEGOT")).toEqual(["I", "'VE", "GOT"]);
    expect(split("we'rex")).toEqual(["we", "'re", "x"]);
  });
});

describe("speech token arithmetic (no golden needed)", () => {
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
    expect(speechIndexOf(89015)).toBeNull(); // こんにちは
  });

  it("exports the ChatML special ids", () => {
    expect(ENDOFTEXT_ID).toBe(151643);
    expect(IM_START_ID).toBe(151644);
    expect(IM_END_ID).toBe(151645);
    expect(SPEECH_TOKEN_BASE).toBe(151669);
    expect(SPEECH_TOKEN_COUNT).toBe(12800);
  });
});

describe("encodeChat structure (no golden needed)", () => {
  it("ends with the ids of '<|im_start|>assistant\\n'", () => {
    const tail = tokenizer.encode("<|im_start|>assistant\n");
    const chat = tokenizer.encodeChat("こんにちは");
    expect(chat.slice(chat.length - tail.length)).toEqual(tail);
  });

  it("omits the generation prompt when asked", () => {
    const withPrompt = tokenizer.encodeChat("こんにちは");
    const without = tokenizer.encodeChat("こんにちは", { addGenerationPrompt: false });
    const tail = tokenizer.encode("<|im_start|>assistant\n");
    expect(withPrompt).toEqual([...without, ...tail]);
    expect(without[without.length - 1]).toBe(tokenizer.encode("<|im_end|>\n").at(-1));
  });
});
