import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadBertWeights } from "./bert-weights.js";
import { realLength, runModernBert } from "./modernbert.js";

/**
 * ModernBERT-ja against the reference, layer by layer.
 *
 *     cd spike/irodori && npm run check:bert
 *
 * Twenty-five layers and one number at the end is not a check, it is a coin
 * flip with extra steps: "wrong" tells you nothing about where. Every captured
 * layer is compared on its own, and the report stops at the first one that
 * disagrees — everything after it is downstream of the same fault and would
 * only pad the output.
 *
 * Errors are peak-relative. The hidden states run to |35|, so an absolute
 * threshold would be either meaningless here or impossible elsewhere.
 */

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN = join(HERE, "golden", "bert");
/**
 * How close is close enough, relative to each stage's peak.
 *
 * Set from what the port actually achieves — the worst stage over both cases
 * lands near 7e-7 — rather than from what would be tolerable. A loose bound is
 * not a safe default here: at 2e-4 this check could not tell exact `gelu` from
 * its tanh approximation, which is a real difference in the arithmetic, and it
 * said "ok" to a run using the wrong one.
 */
const TOLERANCE = 5e-6;

interface Entry {
  shape: number[];
  bytes: number;
}
interface Case {
  text: string;
  normalized: string;
  real_positions: number;
}
interface Index {
  max_text_len: number;
  keep_layers: number[];
  cases: Record<string, Case>;
  weights: Record<string, Entry>;
}

if (!existsSync(join(GOLDEN, "index.json"))) {
  throw new Error(
    `golden/bert is missing. Rebuild it with\n` +
      `  cd spike/irodori && IRODORI_REPO=... ../dacvae/.venv/bin/python dump_bert.py`,
  );
}

function golden(name: string): Float32Array {
  const bytes = readFileSync(join(GOLDEN, `${name}.f32`));
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return new Float32Array(copy.buffer);
}

/** Largest absolute difference over the first `rows` rows, and the peak there. */
function compare(mine: Float32Array, theirs: Float32Array, rows: number, width: number) {
  let worst = 0;
  let peak = 0;
  let at = -1;
  for (let index = 0; index < rows * width; index += 1) {
    const gap = Math.abs(mine[index]! - theirs[index]!);
    if (gap > worst) {
      worst = gap;
      at = index;
    }
    peak = Math.max(peak, Math.abs(theirs[index]!));
  }
  return { worst, peak, relative: peak > 0 ? worst / peak : worst, row: Math.floor(at / width) };
}

const index = JSON.parse(readFileSync(join(GOLDEN, "index.json"), "utf8")) as Index;
const started = Date.now();
const { weights } = loadBertWeights(GOLDEN);
console.log(
  `${weights.config.numLayers} layers, ${(weights.floats * 4) / 1e6 | 0} MB of weights, ` +
    `loaded in ${((Date.now() - started) / 1000).toFixed(1)}s`,
);

const full = process.argv.includes("--full");
const width = weights.config.hiddenSize;
const names = ["embeddings", ...index.keep_layers.map((layer) => `layers.${layer}`), "final_norm"];
let failures = 0;
let checked = 0;

for (const [name, entry] of Object.entries(index.cases)) {
  const paddedIds = Int32Array.from(golden(`${name}/input_ids`), (value) => value | 0);
  const paddedKeep = Array.from(golden(`${name}/mask`), (value) => value !== 0);
  const real = realLength(paddedKeep);
  const inputIds = full ? paddedIds : paddedIds.subarray(0, real);
  const keep = full ? paddedKeep : paddedKeep.slice(0, real);

  const preview = entry.normalized.length > 30 ? `${entry.normalized.slice(0, 30)}…` : entry.normalized;
  console.log(`── ${name}: ${JSON.stringify(preview)}`);
  console.log(
    full
      ? `   all ${paddedIds.length} padded positions (--full)`
      : `   ${real} of ${paddedIds.length} positions — padding is exact to drop, see realLength()`,
  );

  const runStarted = Date.now();
  const run = runModernBert({
    weights,
    inputIds,
    keep,
    capture: names,
    onLayer: (at, total) => process.stdout.write(`\r   layer ${at + 1}/${total}   `),
  });
  process.stdout.write("\r".padEnd(26) + "\r");
  console.log(`   ran in ${((Date.now() - runStarted) / 1000).toFixed(1)}s`);

  for (const stage of names) {
    const mine = stage === "final_norm" ? run.hidden : run.captured.get(stage)!;
    const { worst, peak, relative, row } = compare(mine, golden(`${name}/${stage}`), run.length, width);
    checked += 1;
    const ok = relative <= TOLERANCE;
    console.log(
      `   ${ok ? "ok  " : "FAIL"} ${stage.padEnd(14)} max |diff| ${worst.toExponential(2)} ` +
        `of peak ${peak.toFixed(2)}  (${relative.toExponential(2)} relative, worst at row ${row})`,
    );
    if (!ok) {
      // Later layers are downstream of this one and would all fail for the same
      // reason. The first disagreement in a case is the only informative one.
      failures += 1;
      console.log(`   first disagreement in ${name} — everything after it is downstream`);
      break;
    }
  }
  console.log();
}

if (failures > 0) {
  console.log(`${failures} of ${Object.keys(index.cases).length} cases disagree`);
  process.exitCode = 1;
} else {
  console.log(
    `${checked} stage comparisons agree within ${TOLERANCE.toExponential(0)} of peak, ` +
      `against Irodori's own ModernBERT-ja`,
  );
  if (!full) console.log(`re-run with --full to check the truncation itself over all 256 positions`);
}
