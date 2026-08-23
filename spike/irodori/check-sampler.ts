import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import type { Context } from "./dit.js";
import { loadModelWeights } from "./model-weights.js";
import { condModule, linearSchedule, sample, timestepEmbedding } from "./sampler.js";

/**
 * The whole flow loop against the reference, from the reference's own noise.
 *
 *     cd spike/irodori && npm run check:sampler
 *
 * Thirty-two Euler steps through twelve DiT blocks, sixteen of them at a batch
 * of three. On web-xpu-ops' CPU reference that is roughly twenty minutes; it is
 * the definition of correct and the slowest thing available.
 *
 * The initial noise comes from the golden. `torch.randn` with a seeded
 * generator is not reproducible in JavaScript, and it does not need to be:
 * `dump_golden.py` recorded `in_proj`'s input at step 0, which is `x_t` before
 * anything touched it. Sampling from the reference's own noise is also the only
 * way to compare trajectories rather than distributions.
 *
 * `x_t` was recorded again at steps 16 and 31, so the loop is checked at three
 * points rather than one — enough to separate a wrong schedule (which drifts)
 * from a wrong guidance combination (which is wrong immediately).
 */

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN = join(HERE, "golden");
const MODEL = join(GOLDEN, "model");
const STEPS = 32;

/**
 * Looser than the 5e-6 the static stages hold to, and for a reason worth
 * naming.
 *
 * `get_timestep_embedding` multiplies the timestep by frequencies up to 1000,
 * so `cos` is evaluated near **999 radians** — where float32 keeps about four
 * digits of the argument and loses the rest. torch computes it in float32 and
 * lands ~6e-5 from the true value; this port computes it in float64, because
 * JavaScript numbers are float64, and lands on the true value. Neither is
 * wrong; they differ by torch's error, and no rounding order reproduces it
 * without reimplementing torch's `cos`.
 *
 * That difference enters `cond_module`, becomes the AdaLN shift, scale and
 * gate, and is then integrated over 32 steps. The bound below is what that
 * costs, and `check-dit.ts` remains at 5e-6 because it is handed the
 * reference's own `cond_embed` and never sees it.
 */
const TOLERANCE = 5e-3;

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

function report(label: string, mine: Float32Array, theirs: Float32Array, bound: number): boolean {
  const { worst, peak, relative } = compare(mine, theirs);
  const ok = relative <= bound;
  console.log(
    `${ok ? "ok  " : "FAIL"} ${label.padEnd(30)} max |diff| ${worst.toExponential(2)} ` +
      `of peak ${peak.toFixed(3)}  (${relative.toExponential(2)} relative)`,
  );
  return ok;
}

const weights = loadModelWeights(MODEL);
const { config, dit } = weights;
const schedule = linearSchedule(STEPS);
console.log(
  `${STEPS} Euler steps, t from ${schedule[0]!.toFixed(3)} to ${schedule[STEPS]!.toFixed(3)}, ` +
    `${dit.blocks.length} blocks each`,
);

let failures = 0;

// The timestep embedding and `cond_module` on their own, so the float32 gap
// above is a measured number rather than an explanation.
const t0 = schedule[0]!;
const embedded = timestepEmbedding(t0, config.timestep_embed_dim, 3);
if (!report("timestep embedding (t=0.999)", embedded, golden("cond_module.in.input"), 1e-4)) failures += 1;
if (!report("cond_module", condModule(embedded, weights.cond, dit.shape.dim, 3), golden("cond_module"), 1e-3)) {
  failures += 1;
}
console.log();

// The guided stack, as the reference built it: member 0 conditional, member 1
// with the text dropped, member 2 with the speaker dropped. Read off the
// golden — every member's caption is zero here because the request had none,
// which is why the batch is three and not four.
const at = (argument: string, step = "") => `blocks.0.in.${argument}${step}`;
const guided: Record<string, Context> = {
  text: { state: golden(at("text_state")), keep: flags(at("text_mask")) },
  speaker: { state: golden(at("speaker_state")), keep: flags(at("speaker_mask")) },
  caption: { state: golden(at("caption_state")), keep: flags(at("caption_mask")) },
};
const plain: Record<string, Context> = {
  text: { state: golden(at("text_state", "__step31")), keep: flags(at("text_mask", "__step31")) },
  speaker: { state: golden(at("speaker_state", "__step31")), keep: flags(at("speaker_mask", "__step31")) },
  caption: { state: golden(at("caption_state", "__step31")), keep: flags(at("caption_mask", "__step31")) },
};

const noiseStack = golden("in_proj.in.input");
const tokens = noiseStack.length / (3 * config.latent_dim);
// Member 0 of the stack is `x_t` itself; the other two are copies of it.
const noise = noiseStack.slice(0, tokens * config.latent_dim);

const recorded = new Map<number, Float32Array>([
  [16, golden("in_proj.in.input__step16")],
  [31, golden("in_proj.in.input__step31")],
]);

console.log(`sampling ${tokens} latent frames from the reference's own noise`);
const started = Date.now();
const final = sample({
  weights,
  cond: weights.cond,
  noise,
  conditions: { guided, plain, scales: [3.0, 5.0] },
  steps: STEPS,
  onStep: (step, t, x) => {
    const theirs = recorded.get(step);
    const elapsed = ((Date.now() - started) / 1000).toFixed(0);
    process.stdout.write(`\r  step ${String(step).padStart(2)}/${STEPS}  t=${t.toFixed(3)}  ${elapsed}s   `);
    if (!theirs) return;
    process.stdout.write("\r".padEnd(48) + "\r");
    if (!report(`x_t at step ${step}`, x, theirs, TOLERANCE)) failures += 1;
  },
});
process.stdout.write("\r".padEnd(48) + "\r");
console.log(`\n${STEPS} steps in ${((Date.now() - started) / 1000).toFixed(0)}s\n`);

console.log(`final latent: ${final.length / config.latent_dim} frames of ${config.latent_dim}`);
if (failures > 0) {
  console.log(`\n${failures} comparisons disagree`);
  process.exitCode = 1;
} else {
  console.log(`\nthe flow loop tracks the reference to ${TOLERANCE.toExponential(0)} of peak over all ${STEPS} steps`);
}
