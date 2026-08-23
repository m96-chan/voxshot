import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { type Context, projectContexts } from "./dit.js";
import { loadModelWeights } from "./model-weights.js";
import { condModule, linearSchedule, sample, timestepEmbedding, velocityFor } from "./sampler.js";

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
 * `x_t` is recorded at steps 1, 16 and 31. Step 1 is where guidance and the
 * Euler update are already wrong if they are wrong at all — about a minute in,
 * against eighteen for step 16 — and 16 and 31 are what tell a schedule that
 * drifts apart from arithmetic that was wrong immediately.
 *
 * `--stop-at <step>` ends the run once every golden up to that step has been
 * compared, which is what makes checking the guidance combination affordable.
 */

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN = join(HERE, "golden");
const MODEL = join(GOLDEN, "model");
const STEPS = 32;

/**
 * The same 5e-6 the static stages hold to — which was not the expectation.
 *
 * `get_timestep_embedding` multiplies the timestep by frequencies up to 1000,
 * so `cos` is evaluated near **999 radians**, where float32 keeps about four
 * digits of the argument. torch computes it in float32 and lands ~6e-5 from the
 * true value; this port computes it in float64, because JavaScript numbers are
 * float64, and lands on the true value. Neither is wrong, and no rounding order
 * reproduces torch's without reimplementing its `cos`.
 *
 * That looked like it would cost precision through 32 integrated steps, and it
 * does not. `cond_module`'s first `Linear` contracts the difference to 7.5e-8
 * before it reaches a single AdaLN, and the loop finishes at 1.1e-6 — no worse
 * than a stage that never sees it. The bound is set from that measurement.
 */
const TOLERANCE = 5e-6;

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

// The step-0 velocity, before guidance combines it. One guided forward pass
// exercises the timestep embedding, `cond_module`, `in_proj`, all twelve
// blocks, `out_norm` and `out_proj` — everything except the schedule and the
// combination — in about a minute rather than the loop's twenty-four.
const quick = process.argv.includes("--quick");
const stopAt = (() => {
  const at = process.argv.indexOf("--stop-at");
  return at >= 0 ? Number(process.argv[at + 1]) : Infinity;
})();

const noiseStack = golden("in_proj.in.input");
const tokens = noiseStack.length / (3 * config.latent_dim);
// Member 0 of the stack is `x_t` itself; the other two are copies of it.
const noise = noiseStack.slice(0, tokens * config.latent_dim);

/**
 * `x_t` as the reference had it, at each recorded step.
 *
 * Trimmed to the first batch member, because a *guided* step records the
 * stacked `[3, 97, 32]` and `x_t` is one `[97, 32]` copied three times.
 * Comparing against the whole stack reads past the end of the port's array and
 * reports NaN — which is what it did, and is the same batch confusion that cost
 * the DiT block a wrong first result.
 */
const width = tokens * config.latent_dim;
const recorded = new Map<number, Float32Array>([
  // Step 1 is the cheap one and the reason the dump records it: guidance and
  // the Euler update both go wrong on the *first* step, so a fault that only
  // showed at step 16 cost eighteen minutes an attempt to see.
  [1, golden("in_proj.in.input__step1").subarray(0, width)],
  [16, golden("in_proj.in.input__step16").subarray(0, width)],
  [31, golden("in_proj.in.input__step31").subarray(0, width)],
]);

/**
 * Where guidance stops, checked against the shapes rather than against `rf.py`.
 *
 * The reference recorded `blocks.0.in.x` as `[3, 97, 1280]` at step 0 and
 * `[1, 97, 1280]` at steps 16 and 31, so the batch width at a recorded step
 * *is* whether guidance ran there. This costs nothing and pins the schedule and
 * `cfg_min_t` together — a wrong `0.999`, a wrong step count or a wrong
 * threshold all move the crossing.
 */
const shapes = (JSON.parse(readFileSync(join(GOLDEN, "index.json"), "utf8")) as {
  tensors: Record<string, { shape: number[] }>;
}).tensors;
for (const [step, key] of [[0, "blocks.0.in.x"], [16, "blocks.0.in.x__step16"], [31, "blocks.0.in.x__step31"]] as const) {
  const t = schedule[step]!;
  const mine = t >= 0.5 && t <= 1.0;
  const theirs = shapes[key]!.shape[0]! > 1;
  const ok = mine === theirs;
  if (!ok) failures += 1;
  console.log(
    `${ok ? "ok  " : "FAIL"} guidance at step ${String(step).padStart(2)}          ` +
      `t=${t.toFixed(3)} -> ${mine ? "guided" : "plain"}, reference ran batch ${shapes[key]!.shape[0]}`,
  );
}
console.log();

if (quick) {
  const guidedContext = dit.blocks.map((block) =>
    projectContexts(guided, block, dit.shape, 3),
  );
  const started_ = Date.now();
  const v = velocityFor(noiseStack, t0, weights, guidedContext, tokens, 3);
  console.log(`one guided velocity in ${((Date.now() - started_) / 1000).toFixed(0)}s`);
  if (!report("velocity at step 0", v, golden("out_proj"), TOLERANCE)) failures += 1;
  console.log();
  console.log(
    failures > 0
      ? `${failures} comparisons disagree`
      : `the velocity function agrees within ${TOLERANCE.toExponential(0)} of peak ` +
        `(--quick: the schedule and the guidance combination are not exercised)`,
  );
  if (failures > 0) process.exitCode = 1;
  process.exit();
}

class StopEarly extends Error {}

console.log(`sampling ${tokens} latent frames from the reference's own noise`);
const started = Date.now();
let final: Float32Array | null = null;
try {
  final = sample({
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
    if (step >= stopAt) throw new StopEarly();
  },
  });
} catch (error) {
  if (!(error instanceof StopEarly)) throw error;
}
process.stdout.write("\r".padEnd(48) + "\r");
console.log(`\n${((Date.now() - started) / 1000).toFixed(0)}s\n`);

if (final) console.log(`final latent: ${final.length / config.latent_dim} frames of ${config.latent_dim}`);
else console.log(`stopped at step ${stopAt} (--stop-at); later steps were not run`);
if (failures > 0) {
  console.log(`\n${failures} comparisons disagree`);
  process.exitCode = 1;
} else {
  console.log(`\nthe flow loop tracks the reference to ${TOLERANCE.toExponential(0)} of peak over all ${STEPS} steps`);
}
