import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { MAX_NEW_TOKENS, maxNewFor } from "./constants.js";
import { generateQ8 } from "./model-q8.js";
import { normalizeText } from "./text.js";
import { loadTokenizer, speechIndexOf, type TokenizerJson } from "./tokenizer.js";
import { loadWeightsQ8FromDir } from "./weights-q8-node.js";

/**
 * The E2E oracle: raw text in, the q8 CPU model's greedy token ids out.
 *
 *   npx tsx expected-tokens.ts "<raw text>" [--max-new 700] [--dump-margins]
 *
 * Pipeline = the reference server's, minus sampling temperature (greedy, so
 * the output is reproducible and another script can diff against it):
 * `normalize_text` -> ChatML user turn -> greedy generateQ8 with stopAtEos ->
 * one JSON line on stdout:
 *
 *   {"promptIds":[...],"ids":[...],"speech":[...codec indices...],"eosReached":bool}
 *
 * With `--dump-margins` the line also carries per-step argmax margins,
 * `"margins":[{top1,top2,relGap},...]` — one entry per generated id, where
 * relGap = (logit(top1) - logit(top2)) / max(|top1|,|top2| logits). A checker
 * that saw a GPU/CPU divergence reruns with this flag to tell a near-tie
 * argmax flip (tiny relGap at the divergence step) from a real bug.
 *
 * The requested --max-new is capped exactly like browser.ts caps its runs:
 * min(maxNew, MAX_SEQ_LEN - promptIds.length) — see constants.ts.
 *
 * Everything that is not that JSON line goes to stderr, so callers can pipe
 * stdout straight into JSON.parse. 700 is the reference server's max_tokens.
 */

const REPO_ID = "Aratako/MioTTS-0.6B";
const HERE = dirname(fileURLToPath(import.meta.url));

function usage(): never {
  process.stderr.write('usage: npx tsx expected-tokens.ts "<raw text>" [--max-new 700] [--dump-margins]\n');
  process.exit(2);
}

function parseArgs(argv: string[]): { text: string; maxNew: number; dumpMargins: boolean } {
  let text: string | null = null;
  let maxNew = MAX_NEW_TOKENS;
  let dumpMargins = false;
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]!;
    if (arg === "--max-new") {
      const value = argv[i + 1];
      if (value === undefined) usage();
      maxNew = Number(value);
      if (!Number.isInteger(maxNew) || maxNew <= 0) usage();
      i += 1;
    } else if (arg === "--dump-margins") {
      dumpMargins = true;
    } else if (text === null) {
      text = arg;
    } else {
      usage();
    }
  }
  if (text === null) usage();
  return { text, maxNew, dumpMargins };
}

/** tokenizer.json from wherever `huggingface_hub` put it (weights-cache style). */
function loadTokenizerJson(): TokenizerJson {
  const hub = join(homedir(), ".cache", "huggingface", "hub");
  const repo = `models--${REPO_ID.replace("/", "--")}`;
  let raw: Buffer;
  try {
    const revision = readFileSync(join(hub, repo, "refs", "main"), "utf8").trim();
    raw = readFileSync(join(hub, repo, "snapshots", revision, "tokenizer.json"));
  } catch {
    throw new Error(
      `${REPO_ID}'s tokenizer.json is not in the HF cache. It arrives with the golden:\n` +
        `  cd spike/miotts && python3 dump_golden.py`,
    );
  }
  // refs/main can silently move — pin against the sha the golden vectors were
  // dumped from, so this oracle never tokenizes with a different file than
  // the one the port was verified against.
  let pinned: string | undefined;
  try {
    pinned = (
      JSON.parse(readFileSync(join(HERE, "golden", "tokenizer_vectors.json"), "utf8")) as {
        tokenizer_sha256?: string;
      }
    ).tokenizer_sha256;
  } catch {
    /* handled below — one message for missing file and missing field */
  }
  if (pinned === undefined) {
    throw new Error(
      "golden/tokenizer_vectors.json (with tokenizer_sha256) is missing — rebuild it with\n" +
        "  cd spike/miotts && python3 dump_tokenizer_vectors.py",
    );
  }
  const actual = createHash("sha256").update(raw).digest("hex");
  if (actual !== pinned) {
    throw new Error(
      `tokenizer.json drift: the HF-cache file (refs/main) hashes to ${actual}\n` +
        `but golden/tokenizer_vectors.json was dumped from ${pinned}.\n` +
        `The snapshot moved under us — re-verify the port against the new file and rerun\n` +
        `  cd spike/miotts && python3 dump_tokenizer_vectors.py`,
    );
  }
  return JSON.parse(raw.toString("utf8")) as TokenizerJson;
}

interface Margin {
  top1: number;
  top2: number;
  relGap: number;
}

/** Argmax margin of one logits vector: the top-2 ids and their relative gap. */
function marginOf(logits: Float32Array): Margin {
  let top1 = 0;
  let top2 = -1;
  for (let i = 1; i < logits.length; i += 1) {
    if (logits[i]! > logits[top1]!) {
      top2 = top1;
      top1 = i;
    } else if (top2 < 0 || logits[i]! > logits[top2]!) {
      top2 = i;
    }
  }
  const l1 = logits[top1]!;
  const l2 = logits[top2]!;
  const relGap = (l1 - l2) / Math.max(Math.abs(l1), Math.abs(l2), Number.MIN_VALUE);
  return { top1, top2, relGap };
}

async function main(): Promise<void> {
  const { text, maxNew, dumpMargins } = parseArgs(process.argv.slice(2));

  const tokenizer = await loadTokenizer(loadTokenizerJson());
  const weights = loadWeightsQ8FromDir(join(HERE, "q8"));

  const normalized = normalizeText(text);
  const promptIds = tokenizer.encodeChat(normalized);
  // The identical cap browser.ts applies — a one-sided cap would fake a divergence.
  const cappedMaxNew = maxNewFor(promptIds.length, maxNew);

  const margins: Margin[] = [];
  const started = performance.now();
  const ids = generateQ8(promptIds, cappedMaxNew, weights, {
    stopAtEos: true,
    ...(dumpMargins ? { onLogits: (_step: number, logits: Float32Array) => margins.push(marginOf(logits)) } : {}),
  });
  const elapsed = performance.now() - started;
  process.stderr.write(
    `${ids.length} ids in ${(elapsed / 1000).toFixed(1)}s ` +
      `(${(elapsed / ids.length).toFixed(0)} ms/token)\n`,
  );

  const speech = ids
    .map((id) => speechIndexOf(id))
    .filter((n): n is number => n !== null);
  const eosReached = weights.config.eosIds.includes(ids[ids.length - 1]!);

  const payload = dumpMargins
    ? { promptIds, ids, speech, eosReached, margins }
    : { promptIds, ids, speech, eosReached };
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}

main().catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
});
