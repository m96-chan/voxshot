import { beforeAll, describe, expect, it } from "vitest";
import { decode, MIOCODEC_24K, type Tensor } from "./decoder.js";
import { GoldenCase, loadCase, loadIndex, worstDifference } from "./golden.js";
import { loadWeights } from "./weights-cache.js";

/**
 * The decoder, stage by stage, against the reference implementation's own
 * intermediates.
 *
 * Every stage is compared, not only the waveform: checked at the output alone a
 * port learns that something is wrong and nothing about where, because every
 * stage's error arrives there together. The stages here are ordered as the graph
 * runs, so the **first** failure names the culprit.
 */

const index = loadIndex();

const weights = loadWeights(index.repo_id);

/**
 * The stages, in graph order, with the layout each one has in the golden.
 *
 * `torchShape` is what the reference emits; the port keeps a batch-free version
 * of the same thing. Recording both is what makes a transposed-but-otherwise-
 * correct stage fail loudly rather than by a factor of nothing.
 */
const STAGES: { name: string; note: string }[] = [
  { name: "content_embedding", note: "FSQ decode" },
  { name: "after_prenet", note: "Transformer 768, 6L x 12H, window 65 -> 512" },
  { name: "after_conv_upsample", note: "ConvTranspose1d k=2 s=2" },
  { name: "after_interpolate", note: "linear resample to the STFT length" },
  { name: "after_prior_net", note: "ResNet x2, GroupNorm(32)" },
  { name: "after_decoder", note: "Transformer 512, 8L x 8H, AdaLN-Zero" },
  { name: "after_post_net", note: "ResNet x2" },
  { name: "istft_linear", note: "Linear [512 -> 1922]" },
  { name: "spec_real", note: "exp(log-magnitude) * cos(phase)" },
  { name: "spec_imag", note: "exp(log-magnitude) * sin(phase)" },
  { name: "waveform", note: 'iSTFT, "same" padding' },
];

/**
 * `[B, bins, frames]` or `[B, C, L]` from torch, against `[frames, bins]` or
 * `[C, L]` here — and `[B, L, C]` for the attention stages, which the port also
 * keeps as `[L, C]`. Dropping the leading 1 is the only reshaping done before
 * comparison; anything beyond that would be papering over a real layout bug.
 */
function expectedFor(golden: GoldenCase, name: string): { data: Float32Array; shape: number[] } {
  const tensor = golden.tensor(name);
  const shape = tensor.shape[0] === 1 && tensor.shape.length > 1 ? tensor.shape.slice(1) : tensor.shape;
  return { data: tensor.data, shape };
}

/**
 * A transposed pair of a stage, so a comparison cannot pass by coincidence of
 * length. Spec stages are bin-major upstream and frame-major here — a
 * documented difference, not a bug — so they are turned before comparing.
 */
function alignToGolden(name: string, actual: Tensor, expectedShape: number[]): Float32Array {
  if (name === "spec_real" || name === "spec_imag") {
    const [bins, frames] = expectedShape as [number, number];
    const out = new Float32Array(actual.data.length);
    for (let f = 0; f < frames; f += 1) {
      for (let b = 0; b < bins; b += 1) out[b * frames + f] = actual.data[f * bins + b]!;
    }
    return out;
  }
  return actual.data;
}

/**
 * What these comparisons do **not** observe, measured rather than guessed.
 *
 * `GROUP_NORM_EPS` is 1e-6 because `ResNetBlock` passes that explicitly, and
 * torch's own default is 1e-5. Changing it here to 1e-5 leaves every one of
 * these tests green: inside `sqrt(var + eps)` with a variance of order one, the
 * two differ far below the f32 noise the stages already carry. The value is
 * still right, and it is still worth matching upstream — but nothing here is
 * guarding it, and saying so is cheaper than someone later assuming otherwise.
 *
 * The attention window is only observable in the `windowed` case. At 25 tokens
 * a 65-wide window reaches every position, so `aligned` and `resampled` cannot
 * tell a windowed decoder from an unwindowed one; disabling the mask leaves
 * their prenet stage green and fails only where the sequence is longer.
 */
describe("MioCodec decoder against the golden", () => {
  for (const name of Object.keys(index.cases)) {
    const golden = loadCase(name);
    const manifest = golden.manifest;

    describe(`[${name}] ${manifest.num_tokens} tokens -> ${manifest.stft_length} frames`, () => {
      // Decoded once per case, in `beforeAll` rather than in the `describe`
      // body. Running it during collection makes a throw a collection error
      // with no test attached — measured: breaking the iSTFT's padding mode
      // reported "1 failed" and named no stage at all. Running it inside the
      // first `it` instead just moves the problem, since one decode on the
      // reference implementations takes several seconds and trips the default
      // per-test timeout.
      let result: Awaited<ReturnType<typeof decode>>;
      beforeAll(async () => {
        result = await decode(
          golden.tensor("tokens").data,
          golden.tensor("global_embedding").data,
          manifest.stft_length,
          MIOCODEC_24K,
          weights,
        );
        // These are the *reference* op implementations — deliberately the
        // slowest expression of each. Minutes here would mean something is
        // wrong, but seconds are the design.
      }, 120_000);

      for (const stage of STAGES) {
        it(`${stage.name} — ${stage.note}`, () => {
          const expected = expectedFor(golden, stage.name);
          const actual = result.stages[stage.name];
          expect(actual, `stage ${stage.name} was not produced`).toBeDefined();
          expect(actual!.data.length).toBe(expected.data.length);

          const aligned = alignToGolden(stage.name, actual!, expected.shape);
          const worst = worstDifference(aligned, expected.data);
          // Relative to the stage's own peak. Measured against a zero
          // tolerance, the worst element across all three cases and every
          // stage sits between **2.1e-6 and 2.2e-5** — f32 round-off
          // accumulating through fourteen transformer layers, and a float64
          // JavaScript reference against float32 torch tensors.
          //
          // So 1e-4 is roughly **5x** headroom on the worst stage, not the
          // orders of magnitude it might look like. It is enough because the
          // failures this is guarding against are not subtle: measured, a
          // wrong RoPE theta, a disabled window mask, a dropped `1 +` in
          // AdaLN and an ungated residual each move the answer by tens of
          // percent. It is *not* enough to catch a change of the same size as
          // the noise — see the group-norm epsilon below.
          expect(worst.rel, `worst: ${JSON.stringify(worst)}`).toBeLessThan(1e-4);
        });
      }
    });
  }
});
