import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it } from "vitest";

import {
  PRE_TOKENIZE,
  loadTokenizer,
  type Tokenizer,
} from "../../src/engine/miotts/tokenizer.js";

/**
 * The MioTTS tokenizer port, checked against vectors dumped from the reference
 * HF tokenizer (`spike/miotts/dump_tokenizer_vectors.py`).
 *
 * This file is part of `npm run test:models`, NOT of `npm test`. It needs the
 * 13.8 MB `tokenizer.json` from the model repo and a golden file that is
 * deliberately not in git, neither of which CI can produce.
 *
 * It does not skip when they are absent — it fails, with the command that
 * rebuilds them. A suite that goes green by checking nothing is worse than one
 * that is not run at all, because the green is indistinguishable from a real
 * pass. See `vitest.models.config.ts` for why this directory is invisible to
 * the CI suite rather than merely excluded from it by convention.
 */

const HERE = dirname(fileURLToPath(import.meta.url));

/** Where `dump_tokenizer_vectors.py` writes, and where it is run from. */
const GOLDEN_DIR = join(HERE, "..", "..", "spike", "miotts", "golden");
const REBUILD = "cd spike/miotts && python3 dump_tokenizer_vectors.py";

/**
 * The exact snapshot the vectors were dumped from. Pinned by commit rather
 * than `refs/main`: the vectors are only meaningful against the file they
 * were produced from, and the sha256 check below would otherwise report a
 * model update as a port regression.
 */
const SNAPSHOT = join(
  process.env.HF_HOME ?? join(homedir(), ".cache", "huggingface"),
  "hub",
  "models--Aratako--MioTTS-0.6B",
  "snapshots",
  "901ee12c50efd68ea3db6a6780cd29da197cd0da",
);

interface Vectors {
  _rebuild: string;
  /** sha256 of the exact tokenizer.json the vectors were dumped from. */
  tokenizer_sha256: string;
  encode: { text: string; ids: number[] }[];
  chat: { user: string; ids: number[] }[];
  decode: { ids: number[]; text: string }[];
}

function read(path: string, what: string, rebuild: string): Buffer {
  try {
    return readFileSync(path);
  } catch (cause) {
    throw new Error(`${what} is missing (${path}). Rebuild it with\n  ${rebuild}`, { cause });
  }
}

const vectors = JSON.parse(
  read(join(GOLDEN_DIR, "tokenizer_vectors.json"), "golden/tokenizer_vectors.json", REBUILD).toString(
    "utf8",
  ),
) as Vectors;

let tokenizer: Tokenizer;

beforeAll(async () => {
  const bytes = read(
    join(SNAPSHOT, "tokenizer.json"),
    "the model's tokenizer.json",
    'python3 -c "from huggingface_hub import snapshot_download; snapshot_download(\'Aratako/MioTTS-0.6B\')"',
  );
  tokenizer = await loadTokenizer(JSON.parse(bytes.toString("utf8")));
});

describe("tokenizer.json pin", () => {
  it("the file under test is byte-identical to the one the vectors were dumped from", () => {
    // Every consumer resolves tokenizer.json independently, so a silent drift
    // between them would make the vectors test the wrong file. Loud failure
    // here means: rerun the dump against the file you actually want.
    const bytes = readFileSync(join(SNAPSHOT, "tokenizer.json"));

    expect(createHash("sha256").update(bytes).digest("hex")).toBe(vectors.tokenizer_sha256);
  });
});

describe("encode against golden vectors", () => {
  for (const { text, ids } of vectors.encode) {
    it(`encodes ${JSON.stringify(text.length > 40 ? `${text.slice(0, 40)}…` : text)}`, () => {
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
  // The reference's Rust `(?i:'s|...)` applies Unicode simple case folding, so
  // U+017F (ſ) folds to 's' and "'ſ" is a contraction piece. The split is NOT
  // observable in encode ids with this vocab — ſ's byte-mapped pieces merge
  // only with each other, so no BPE merge ever crosses the boundary — which is
  // why this pins the pre-tokenizer split itself. Expected splits are the
  // reference tokenizer's own `pre_tokenizer.pre_tokenize_str` output,
  // measured 2026-08-21 against the pinned snapshot.
  const split = (text: string): string[] => Array.from(text.matchAll(PRE_TOKENIZE), (m) => m[0]);

  it("folds ſ into the contraction group like the reference", () => {
    expect(split("x'ſy")).toEqual(["x", "'ſ", "y"]);
    expect(split("it'ſ fine")).toEqual(["it", "'ſ", " fine"]);
  });

  it("still stops uppercase contractions where the reference does", () => {
    expect(split("I'VEGOT")).toEqual(["I", "'VE", "GOT"]);
    expect(split("we'rex")).toEqual(["we", "'re", "x"]);
  });
});
