import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { framesFrom, predictDuration } from "./duration.js";
import { loadModelWeights } from "./model-weights.js";

/**
 * The duration predictor against the reference.
 *
 *     cd spike/irodori && npm run check:duration
 *
 * One scalar, which sounds like a weak check and is not: it is a sum over 256
 * per-token softplus outputs, after three blocks each modulated by a speaker
 * and a caption vector that are added rather than concatenated. Every one of
 * those choices moves the number.
 *
 * It is also the check with the most consequence per digit. A wrong duration
 * does not mispronounce anything — it stretches or clips the whole utterance,
 * and nothing downstream can notice.
 */

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN = join(HERE, "golden");

function golden(name: string): Float32Array {
  const bytes = readFileSync(join(GOLDEN, `${name}.f32`));
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return new Float32Array(copy.buffer);
}

const weights = loadModelWeights(join(GOLDEN, "model"));
const { config } = weights;

const textState = golden("duration_predictor.in.text_state");
const textKeep = Array.from(golden("duration_predictor.in.text_mask"), (value) => value !== 0);
const speakerState = golden("duration_predictor.in.speaker_state");
const hasSpeaker = golden("duration_predictor.in.has_speaker")[0] !== 0;
const hasCaption = golden("duration_predictor.in.has_caption")[0] !== 0;

console.log(
  `${weights.duration.blocks.length} blocks of ${weights.duration.dim}, ` +
    `${textKeep.filter(Boolean).length} real of ${textKeep.length} text tokens, ` +
    `speaker ${hasSpeaker ? "present" : "absent"}, caption ${hasCaption ? "present" : "absent"}`,
);

// `_speaker_vec` takes the speaker state's first token — the mean summary —
// rather than pooling again.
const speakerVec = hasSpeaker ? speakerState.subarray(0, config.speaker_dim) : null;
// `_caption_vec` masks by `caption_mask & has_caption`; with no caption every
// flag is false, the denominator clamps to one, and the result is `null_caption`.
const captionVec = null;

const mine = predictDuration({ weights, textState, textKeep, speakerVec, captionVec });
const theirs = golden("duration_predictor")[0]!;
const gap = Math.abs(mine - theirs);
const relative = gap / Math.abs(theirs);
const ok = relative <= 1e-5;

console.log();
console.log(`     mine  log1p ${mine.toFixed(6)}  ->  ${framesFrom(mine).toFixed(2)} latent frames`);
console.log(`reference  log1p ${theirs.toFixed(6)}  ->  ${framesFrom(theirs).toFixed(2)} latent frames`);
console.log(
  `${ok ? "\nok" : "\nFAIL"} — ${gap.toExponential(2)} apart (${relative.toExponential(2)} relative), ` +
    `${Math.abs(framesFrom(mine) - framesFrom(theirs)).toFixed(3)} frames`,
);
if (!ok) process.exitCode = 1;
