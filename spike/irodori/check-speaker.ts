import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadModelWeights } from "./model-weights.js";
import { runSpeakerEncoder } from "./speaker-encoder.js";

/**
 * The speaker encoder against the reference, block by block.
 *
 *     cd spike/irodori && npm run check:speaker
 *
 * This runs on the reference's *recorded input* — the patched reference latent
 * `dump_golden.py` captured with a pre-hook — rather than on anything this port
 * computed. The DACVAE encoder is not ported, and it does not need to be for
 * this stage to be checkable. That is the whole reason the pre-hooks exist.
 */

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN = join(HERE, "golden");
const MODEL = join(GOLDEN, "model");
// Same basis as check-bert.ts: set from what the port achieves, not from what
// would be tolerable. A loose bound here would not see a missing gate.
const TOLERANCE = 5e-6;

for (const [path, how] of [
  [join(MODEL, "index.json"), "../dacvae/.venv/bin/python dump_model.py"],
  [join(GOLDEN, "index.json"), "IRODORI_REPO=... ../dacvae/.venv/bin/python dump_golden.py --text ... --ref-wav ..."],
] as const) {
  if (!existsSync(path)) throw new Error(`${path} is missing. Rebuild it with\n  cd spike/irodori && ${how}`);
}

function golden(name: string): Float32Array {
  const bytes = readFileSync(join(GOLDEN, `${name}.f32`));
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return new Float32Array(copy.buffer);
}

function compare(mine: Float32Array, theirs: Float32Array) {
  let worst = 0;
  let peak = 0;
  for (let index = 0; index < theirs.length; index += 1) {
    worst = Math.max(worst, Math.abs(mine[index]! - theirs[index]!));
    peak = Math.max(peak, Math.abs(theirs[index]!));
  }
  return { worst, peak, relative: peak > 0 ? worst / peak : worst };
}

const started = Date.now();
const weights = loadModelWeights(MODEL);
const { speaker } = weights;
console.log(
  `speaker encoder: ${speaker.layers} blocks, dim ${speaker.dim}, ` +
    `${speaker.heads} heads, SwiGLU hidden ${speaker.mlpHidden}, ` +
    `loaded in ${((Date.now() - started) / 1000).toFixed(1)}s`,
);

// The recorded input: already patched, so `[frames, latentDim * patch]`.
const latent = golden("speaker_encoder.in.latent");
const keep = Array.from(golden("speaker_encoder.in.mask"), (value) => value !== 0);
const frames = keep.length;
console.log(`input ${frames} patched frames of ${latent.length / frames} features, ${keep.filter(Boolean).length} kept\n`);

const stages = ["blocks.0", "blocks.5"];
const runStarted = Date.now();
const run = runSpeakerEncoder({ weights, latent, keep, capture: stages });
console.log(`ran in ${((Date.now() - runStarted) / 1000).toFixed(1)}s\n`);

let failed: string | null = null;
for (const stage of [...stages, ""]) {
  const name = stage ? `speaker_encoder.${stage}` : "speaker_encoder";
  const mine = stage ? run.captured.get(stage)! : run.state;
  const { worst, peak, relative } = compare(mine, golden(name));
  const ok = relative <= TOLERANCE;
  console.log(
    `${ok ? "ok  " : "FAIL"} ${name.padEnd(26)} max |diff| ${worst.toExponential(2)} ` +
      `of peak ${peak.toFixed(2)}  (${relative.toExponential(2)} relative)`,
  );
  if (!ok) {
    failed = name;
    break;
  }
}

console.log();
if (failed) {
  console.log(`first disagreement at ${failed} — everything after it is downstream`);
  process.exitCode = 1;
} else {
  console.log(`the speaker encoder agrees within ${TOLERANCE.toExponential(0)} of peak`);
}
