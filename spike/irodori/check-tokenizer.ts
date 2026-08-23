import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { normalizeText } from "./text.js";
import { loadTokenizer, type TokenizerJson } from "./tokenizer.js";

/**
 * The text path against the reference, both halves separately.
 *
 *     cd spike/irodori && npm run check:tokenizer
 *
 * Separately, because they fail differently. A normalisation difference is a
 * rule transcribed wrongly and shows up as a different string. A tokenisation
 * difference on identical input is the lattice — a whole algorithm — and shows
 * up as different ids for text that looks the same. Reporting one number for
 * both would say "the text path is wrong" and leave the useful half out.
 */

const HERE = dirname(fileURLToPath(import.meta.url));
const REVISION = "77675fc96a7e445e982e2ba90246b816efc74ec6";
const REPO_DIR = join(
  homedir(),
  ".cache/huggingface/hub/models--sbintuitions--modernbert-ja-310m/snapshots",
  REVISION,
);

interface Vectors {
  _rebuild: string;
  bos_id: number;
  eos_id: number;
  vectors: { text: string; normalized: string; ids: number[] }[];
}

function load<T>(path: string, what: string, rebuild: string): T {
  if (!existsSync(path)) throw new Error(`${what} is missing (${path}). Rebuild it with\n  ${rebuild}`);
  return JSON.parse(readFileSync(path, "utf8")) as T;
}

const vectors = load<Vectors>(
  join(HERE, "golden", "tokenizer_vectors.json"),
  "golden/tokenizer_vectors.json",
  "cd spike/irodori && IRODORI_REPO=... ../dacvae/.venv/bin/python dump_tokenizer.py",
);
const json = load<TokenizerJson>(
  join(REPO_DIR, "tokenizer.json"),
  "ModernBERT-ja's tokenizer.json",
  `python3 -c "from huggingface_hub import hf_hub_download as d; d('sbintuitions/modernbert-ja-310m','tokenizer.json',revision='${REVISION}')"`,
);

const tokenizer = loadTokenizer(json);

let normalizeFailures = 0;
let encodeFailures = 0;

for (const vector of vectors.vectors) {
  const label = JSON.stringify(vector.text.length > 24 ? `${vector.text.slice(0, 24)}…` : vector.text);

  const mine = normalizeText(vector.text).trim();
  if (mine !== vector.normalized) {
    normalizeFailures += 1;
    console.log(`FAIL normalize ${label}`);
    console.log(`       mine ${JSON.stringify(mine)}`);
    console.log(`  reference ${JSON.stringify(vector.normalized)}`);
    continue;
  }

  // Encoded from the REFERENCE's normalized text, not from ours. They agree
  // here — the check above just said so — but taking the reference's keeps a
  // normalisation slip from being reported twice, once as itself and once as a
  // tokenizer failure it did not cause.
  const ids = tokenizer.encode(vector.normalized);
  if (ids.length !== vector.ids.length || ids.some((id, i) => id !== vector.ids[i])) {
    encodeFailures += 1;
    console.log(`FAIL encode ${label}  ${JSON.stringify(vector.normalized)}`);
    console.log(`       mine [${ids.join(", ")}]`);
    console.log(`  reference [${vector.ids.join(", ")}]`);
    continue;
  }

  const roundTrip = tokenizer.decode(ids);
  const marker = roundTrip === vector.normalized ? "ok  " : "ok* ";
  console.log(`${marker} ${label.padEnd(28)} ${ids.length} ids${marker === "ok* " ? "  (decode differs)" : ""}`);
}

console.log();
if (normalizeFailures === 0 && encodeFailures === 0) {
  console.log(
    `all ${vectors.vectors.length} vectors agree — normalization and Unigram both, ` +
      `against ModernBERT-ja @ ${REVISION.slice(0, 8)}`,
  );
} else {
  console.log(`${normalizeFailures} normalization and ${encodeFailures} encode failures`);
  process.exitCode = 1;
}
