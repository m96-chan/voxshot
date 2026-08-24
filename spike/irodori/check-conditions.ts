import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { conditionNorm, prependMeanToken, project } from "./conditions.js";
import { loadModelWeights } from "./model-weights.js";
import { patchReference, runSpeakerEncoder } from "./speaker-encoder.js";

/**
 * Everything between the encoders and the DiT, against the reference.
 *
 *     cd spike/irodori && npm run check:conditions
 *
 * `encode_conditions` is three short paths that are easy to get almost right:
 *
 *   text     backbone -> projector -> text_norm
 *   speaker  patch -> speaker_encoder -> speaker_norm -> prepend mean token
 *   caption  backbone -> projector -> caption_norm
 *
 * Each is checked against what the DiT was actually handed — `blocks.0.in.*`,
 * captured by pre-hook — rather than against the module's own output, because
 * the norms and the mean token happen between the two and a check against the
 * module output would step over them.
 */

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN = join(HERE, "golden");
const MODEL = join(GOLDEN, "model");
/**
 * Per path, because they are not the same depth.
 *
 * The text projector is two matmuls over the backbone's recorded output. The
 * speaker path is eight blocks and a norm, so it carries eight blocks' worth of
 * float32 reordering against CUDA — it lands at 5.7e-6, and a bound that called
 * that a failure would be measuring the depth of the stack rather than the
 * correctness of the port.
 */
const TOLERANCE = { shallow: 5e-6, speaker: 2e-5 };

function golden(name: string): Float32Array {
  const bytes = readFileSync(join(GOLDEN, `${name}.f32`));
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return new Float32Array(copy.buffer);
}

function flags(name: string): boolean[] {
  return Array.from(golden(name), (value) => value !== 0);
}

/**
 * Compare, and say plainly when there was nothing to compare.
 *
 * A golden that is entirely zero cannot distinguish a correct port from one
 * that returns zeros — the caption path is exactly that here, because this
 * request had no caption and the mask zeroes the whole tensor. Reporting it as
 * `ok` would be counting a comparison that observes nothing, so it gets its own
 * verdict and does not count towards either total.
 */
function report(label: string, mine: Float32Array, theirs: Float32Array, bound: number): boolean | null {
  let worst = 0;
  let peak = 0;
  for (let index = 0; index < theirs.length; index += 1) {
    worst = Math.max(worst, Math.abs(mine[index]! - theirs[index]!));
    peak = Math.max(peak, Math.abs(theirs[index]!));
  }
  if (peak === 0) {
    const mineZero = mine.every((value) => value === 0);
    console.log(
      `--   ${label.padEnd(26)} the golden is all zeros — nothing to observe ` +
        `(this port also returns ${mineZero ? "zeros" : "non-zeros, which is a real difference"})`,
    );
    return mineZero ? null : false;
  }
  const relative = worst / peak;
  const ok = relative <= bound;
  console.log(
    `${ok ? "ok  " : "FAIL"} ${label.padEnd(26)} max |diff| ${worst.toExponential(2)} ` +
      `of peak ${peak.toFixed(3)}  (${relative.toExponential(2)} relative)`,
  );
  return ok;
}

const weights = loadModelWeights(MODEL);
const { config, normEps } = weights;
const index = JSON.parse(readFileSync(join(GOLDEN, "index.json"), "utf8")) as {
  tensors: Record<string, { shape: number[] }>;
};
console.log(
  `projectors ${config.pretrained_hidden} -> ${config.text_dim} with a ` +
    `${config.projector_hidden}-wide residual branch (${config.pretrained_projector_type})\n`,
);

let failures = 0;
let unobserved = 0;

// Text and caption: the backbone's masked output is what the projector sees,
// and it is recorded as the backbone module's own output.
for (const which of ["text", "caption"] as const) {
  const inputName = which === "text" ? "text_encoder.in" : "caption_encoder.in";
  const keep = flags(`${inputName}.mask`);
  // The backbone's output before masking; `PretrainedTextBackbone.forward`
  // applies the mask, and so does this.
  const state = golden("pretrained_text_backbone.backbone");
  const dim = config.pretrained_hidden;
  const masked = state.slice(0, keep.length * dim);
  for (let row = 0; row < keep.length; row += 1) {
    if (!keep[row]) masked.fill(0, row * dim, (row + 1) * dim);
  }

  if (which === "text") {
    const projected = project(masked, keep, weights.projectors.text, normEps);
    if (report("text_encoder", projected, golden("text_encoder"), TOLERANCE.shallow) === false) failures += 1;
    const normed = conditionNorm(projected, weights.norms.text, keep.length, config.text_dim, normEps);
    // What the DiT was handed. Member 0 of the guided stack is the conditional.
    const theirs = golden("blocks.0.in.text_state").subarray(0, keep.length * config.text_dim);
    if (report("text_norm -> DiT", normed, theirs, TOLERANCE.shallow) === false) failures += 1;
  } else {
    // The caption backbone pass used different ids, so the recorded text
    // backbone output is the wrong input. Only the norm is checkable here.
    const captionDim = config.caption_dim ?? config.text_dim;
    const captionKeep = flags("caption_norm.in.x").length > 0 ? keep : keep;
    const normed = conditionNorm(
      golden("caption_norm.in.x"),
      weights.norms.caption,
      captionKeep.length,
      captionDim,
      normEps,
    );
    if (report("caption_norm", normed, golden("caption_norm"), TOLERANCE.shallow) === false) failures += 1;
    unobserved += 1;
  }
}

// Speaker: the whole path, from the reference latent the pre-hook recorded.
{
  const rawLatent = golden("speaker_encoder.in.latent");
  const rawKeep = flags("speaker_encoder.in.mask");
  // Already patched by the time `speaker_encoder` sees it, so `patchReference`
  // is exercised on the shape it produced rather than re-applied.
  const patched = patchReference(rawLatent, rawKeep, rawLatent.length / rawKeep.length, 1);
  const run = runSpeakerEncoder({ weights, latent: patched.latent, keep: patched.keep, capture: [] });
  const normed = conditionNorm(run.state, weights.speaker.outNorm, rawKeep.length, config.speaker_dim, normEps);
  if (report("speaker_norm", normed, golden("speaker_norm"), TOLERANCE.speaker) === false) failures += 1;

  const withMean = prependMeanToken(normed, rawKeep, config.speaker_dim);
  const theirs = golden("blocks.0.in.speaker_state").subarray(0, withMean.keep.length * config.speaker_dim);
  if (report("mean token -> DiT", withMean.state, theirs, TOLERANCE.speaker) === false) failures += 1;
  const theirMask = flags("blocks.0.in.speaker_mask").slice(0, withMean.keep.length);
  const maskOk = withMean.keep.every((value, at) => value === theirMask[at]);
  console.log(`${maskOk ? "ok  " : "FAIL"} ${"speaker mask".padEnd(26)} ${withMean.keep.length} flags`);
  if (!maskOk) failures += 1;
}

console.log();
if (failures > 0) {
  console.log(`${failures} comparisons disagree`);
  process.exitCode = 1;
} else {
  console.log(
    `the conditioning path agrees — text within ${TOLERANCE.shallow.toExponential(0)} of peak, ` +
      `speaker within ${TOLERANCE.speaker.toExponential(0)} after eight blocks`,
  );
  if (unobserved > 0) {
    console.log(
      `${unobserved} comparison observed nothing: this request had no caption, so the ` +
        `caption path is zero end to end. It needs a request with one to be checked at all.`,
    );
  }
}
void index;
