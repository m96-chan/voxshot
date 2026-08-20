import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { generateQ8 } from "./model-q8.js";
import { normalizeText } from "./text.js";
import { loadTokenizer, speechIndexOf, type TokenizerJson } from "./tokenizer.js";
import { loadWeightsQ8FromDir } from "./weights-q8-node.js";

/**
 * The E2E oracle: raw text in, the q8 CPU model's greedy token ids out.
 *
 *   npx tsx expected-tokens.ts "<raw text>" [--max-new 700]
 *
 * Pipeline = the reference server's, minus sampling temperature (greedy, so
 * the output is reproducible and another script can diff against it):
 * `normalize_text` -> ChatML user turn -> greedy generateQ8 with stopAtEos ->
 * one JSON line on stdout:
 *
 *   {"promptIds":[...],"ids":[...],"speech":[...codec indices...],"eosReached":bool}
 *
 * Everything that is not that JSON line goes to stderr, so callers can pipe
 * stdout straight into JSON.parse. 700 is the reference server's max_tokens.
 */

const REPO_ID = "Aratako/MioTTS-0.6B";
const HERE = dirname(fileURLToPath(import.meta.url));

function usage(): never {
  process.stderr.write('usage: npx tsx expected-tokens.ts "<raw text>" [--max-new 700]\n');
  process.exit(2);
}

function parseArgs(argv: string[]): { text: string; maxNew: number } {
  let text: string | null = null;
  let maxNew = 700;
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]!;
    if (arg === "--max-new") {
      const value = argv[i + 1];
      if (value === undefined) usage();
      maxNew = Number(value);
      if (!Number.isInteger(maxNew) || maxNew <= 0) usage();
      i += 1;
    } else if (text === null) {
      text = arg;
    } else {
      usage();
    }
  }
  if (text === null) usage();
  return { text, maxNew };
}

/** tokenizer.json from wherever `huggingface_hub` put it (weights-cache style). */
function loadTokenizerJson(): TokenizerJson {
  const hub = join(homedir(), ".cache", "huggingface", "hub");
  const repo = `models--${REPO_ID.replace("/", "--")}`;
  try {
    const revision = readFileSync(join(hub, repo, "refs", "main"), "utf8").trim();
    return JSON.parse(
      readFileSync(join(hub, repo, "snapshots", revision, "tokenizer.json"), "utf8"),
    ) as TokenizerJson;
  } catch {
    throw new Error(
      `${REPO_ID}'s tokenizer.json is not in the HF cache. It arrives with the golden:\n` +
        `  cd spike/miotts && python3 dump_golden.py`,
    );
  }
}

async function main(): Promise<void> {
  const { text, maxNew } = parseArgs(process.argv.slice(2));

  const tokenizer = await loadTokenizer(loadTokenizerJson());
  const weights = loadWeightsQ8FromDir(join(HERE, "q8"));

  const normalized = normalizeText(text);
  const promptIds = tokenizer.encodeChat(normalized);

  const started = performance.now();
  const ids = generateQ8(promptIds, maxNew, weights, { stopAtEos: true });
  const elapsed = performance.now() - started;
  process.stderr.write(
    `${ids.length} ids in ${(elapsed / 1000).toFixed(1)}s ` +
      `(${(elapsed / ids.length).toFixed(0)} ms/token)\n`,
  );

  const speech = ids
    .map((id) => speechIndexOf(id))
    .filter((n): n is number => n !== null);
  const eosReached = weights.config.eosIds.includes(ids[ids.length - 1]!);

  process.stdout.write(`${JSON.stringify({ promptIds, ids, speech, eosReached })}\n`);
}

main().catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
});
