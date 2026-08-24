import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { type Context, ditBlock, projectContexts } from "./dit.js";
import { loadModelWeights } from "./model-weights.js";

/**
 * One `DiffusionBlock` against the reference, on the reference's own inputs.
 *
 *     cd spike/irodori && npm run check:dit
 *
 * Three blocks were recorded — the first, the sixth and the last — at three of
 * the 32 flow steps. Each is checked in isolation: `x`, `cond_embed` and all
 * three context states come from the golden, so a block is right or wrong on
 * its own rather than as the twelfth link in a chain that does not exist yet.
 *
 * That is deliberate. The flow loop is not ported, the timestep embedding is
 * not ported, and neither has to be for the expensive half of the model to be
 * shown correct.
 */

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN = join(HERE, "golden");
const MODEL = join(GOLDEN, "model");
const TOLERANCE = 5e-6;

if (!existsSync(join(MODEL, "index.json"))) {
  throw new Error(`golden/model is missing. Rebuild it with\n  cd spike/irodori && ../dacvae/.venv/bin/python dump_model.py`);
}

function golden(name: string): Float32Array {
  const bytes = readFileSync(join(GOLDEN, `${name}.f32`));
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return new Float32Array(copy.buffer);
}

function flags(name: string): boolean[] {
  return Array.from(golden(name), (value) => value !== 0);
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
const { dit } = weights;
console.log(
  `DiT: ${dit.blocks.length} blocks, dim ${dit.shape.dim}, ${dit.shape.heads} heads ` +
    `(${Math.floor(dit.shape.heads / 2)} rotated), SwiGLU hidden ${dit.shape.mlpHidden}, ` +
    `AdaLN rank ${dit.shape.rank}, loaded in ${((Date.now() - started) / 1000).toFixed(1)}s`,
);

const index = JSON.parse(readFileSync(join(GOLDEN, "index.json"), "utf8")) as {
  tensors: Record<string, { shape: number[] }>;
};
// Whatever `dump_golden.py` recorded: `blocks.N` and `blocks.N__stepM`.
const cases = Object.keys(index.tensors)
  .filter((name) => /^blocks\.\d+(__step\d+)?$/.test(name))
  .sort();
console.log(`${cases.length} recorded block outputs to check\n`);

let failures = 0;
for (const name of cases) {
  const block = Number(/^blocks\.(\d+)/.exec(name)![1]);
  const suffix = name.slice(`blocks.${block}`.length);
  const at = (argument: string) => `blocks.${block}.in.${argument}${suffix}`;

  const x = golden(at("x"));
  const cond = golden(at("cond_embed"));
  // Batch comes from the recorded shape, not from the length. The first fire of
  // every step-0 module is the CFG batch of three; reading `[3, 97, 1280]` as
  // 291 tokens lets the three variants attend to each other.
  const [batch, tokens] = index.tensors[at("x")]!.shape as [number, number];
  const contexts: Record<string, Context> = {
    text: { state: golden(at("text_state")), keep: flags(at("text_mask")) },
    speaker: { state: golden(at("speaker_state")), keep: flags(at("speaker_mask")) },
    caption: { state: golden(at("caption_state")), keep: flags(at("caption_mask")) },
  };

  const weightsFor = dit.blocks[block]!;
  const context = projectContexts(contexts, weightsFor, dit.shape, batch);
  const runStarted = Date.now();
  const mine = ditBlock(x, cond, weightsFor, context, tokens, dit.shape);
  const { worst, peak, relative } = compare(mine, golden(name));
  const ok = relative <= TOLERANCE;
  if (!ok) failures += 1;
  console.log(
    `${ok ? "ok  " : "FAIL"} ${name.padEnd(22)} batch ${batch}, ${tokens} latent + ${context.tokens} context keys  ` +
      `max |diff| ${worst.toExponential(2)} of peak ${peak.toFixed(2)} ` +
      `(${relative.toExponential(2)} relative, ${((Date.now() - runStarted) / 1000).toFixed(1)}s)`,
  );
}

console.log();
if (failures > 0) {
  console.log(`${failures} of ${cases.length} recorded blocks disagree`);
  process.exitCode = 1;
} else {
  console.log(`all ${cases.length} recorded blocks agree within ${TOLERANCE.toExponential(0)} of peak`);
}
