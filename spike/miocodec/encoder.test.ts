import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it } from "vitest";
import { cpuBackend, type Backend } from "./decoder.js";
import { calculateWaveformPadding, encodeGlobal } from "./encoder.js";
import { GoldenCase, worstDifference, type CaseManifest } from "./golden.js";
import { loadEncoderWeights } from "./weights-cache.js";

/**
 * The encoder's global path, stage by stage, against the reference
 * implementation's own intermediates.
 *
 * Same shape as `decoder.test.ts`, for the same reason: checked at the
 * embedding alone a port learns that something is wrong and nothing about
 * where. The stages are ordered as the graph runs, so the **first** failure
 * names the culprit.
 */

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "golden-encoder");

interface EncoderCaseManifest extends Omit<CaseManifest, "num_tokens" | "stft_length" | "interpolate_is_identity"> {
  audio_length: number;
  waveform_padding: number;
}

interface EncoderIndex {
  repo_id: string;
  cases: Record<string, EncoderCaseManifest>;
}

/**
 * Spelled out, as in `golden.ts`: tests that skip themselves when the fixtures
 * are absent report success for having checked nothing.
 */
function loadEncoderIndex(): EncoderIndex {
  try {
    return JSON.parse(readFileSync(join(ROOT, "index.json"), "utf8")) as EncoderIndex;
  } catch {
    throw new Error(
      `golden-encoder/index.json is missing. The goldens are deliberately not in git; rebuild with\n` +
        `  cd spike/miocodec && .venv/bin/python dump_encoder_golden.py\n` +
        `See spike/miocodec/README.md for how the venv is put together.`,
    );
  }
}

const index = loadEncoderIndex();

// Throws with the export command when encoder-weights.safetensors is absent.
const weights = loadEncoderWeights();

/**
 * The stages, in graph order. Layouts: the golden's conv stages
 * (`convnext_block*`, `attn_weights`) are channel-major `[1, C, T]` and the
 * port keeps `[C, T]` for them, so after dropping the leading batch 1 every
 * comparison is element-for-element — no transpose is needed anywhere, and the
 * length assertion below keeps that claim honest.
 */
const STAGES: { name: string; note: string; tolerance?: number }[] = [
  { name: "waveform_24k", note: "input echo — the 24 kHz waveform as handed in" },
  { name: "after_resample", note: "pad + polyphase 24k -> 16k, conv [2,1,23] stride 3" },
  { name: "after_feature_extractor", note: "7x Conv1d frontend, GroupNorm(512) after block 0, exact GELU" },
  { name: "after_feature_projection", note: "LayerNorm(512) -> Linear 512 -> 768" },
  { name: "after_pos_conv", note: "grouped Conv1d k128 g16, crop 1, GELU" },
  { name: "ssl_layer1", note: "WavLM layer 1, post-norm, gated relative position bias" },
  { name: "ssl_layer2", note: "WavLM layer 2, bias reused from layer 1, own gate" },
  { name: "global_input", note: "mean(layer1, layer2) — no z-norm on this branch" },
  { name: "convnext_block1", note: "depthwise k7 -> LN(1e-6) -> MLP 384/1152 -> gamma -> +res" },
  // The ConvNeXt tail carries a wider tolerance, and it is the *reference's*
  // doing, not the port's: torch itself, run in f64 against its own f32 on
  // jp_ref1, disagrees by 1.0e-5 / 4.1e-5 / 4.7e-5 / 9.9e-5 across blocks 1-4
  // (measured; the layer-scale residual chain amplifies f32 round-off on that
  // clip). The port lands at 1.0e-5 / 3.9e-5 / 4.8e-5 / 10.0e-5 — the same
  // conditioning floor — so a 1e-4 bar here would sit exactly on the noise
  // with no headroom at all. 5e-4 is ~5x headroom, and still three orders of
  // magnitude below a real mistake: the deliberate LayerNorm break described
  // below moved every one of these blocks by 4.3e-1 to 5.3e-1, measured.
  { name: "convnext_block2", note: "ConvNeXt block 2", tolerance: 5e-4 },
  { name: "convnext_block3", note: "ConvNeXt block 3", tolerance: 5e-4 },
  { name: "convnext_block4", note: "ConvNeXt block 4", tolerance: 5e-4 },
  { name: "after_backbone", note: "final LayerNorm(384, eps 1e-6)", tolerance: 5e-4 },
  { name: "attn_weights", note: "attentive stats pool: softmax over *time*" },
  { name: "pooled_stats", note: "cat(mean, std), std clamped to [1e-4, 1e4]" },
  { name: "global_embedding", note: "Linear 768 -> 128 -> LayerNorm(128)" },
];

/** Drop the leading batch of 1 the reference carries; nothing else is reshaped. */
function expectedFor(golden: GoldenCase, name: string): { data: Float32Array; shape: number[] } {
  const tensor = golden.tensor(name);
  const shape = tensor.shape[0] === 1 && tensor.shape.length > 1 ? tensor.shape.slice(1) : tensor.shape;
  return { data: tensor.data, shape };
}

describe("MioCodec encoder (global path) against the golden", () => {
  for (const name of Object.keys(index.cases)) {
    const manifest = index.cases[name]!;
    const golden = new GoldenCase(manifest as unknown as CaseManifest, join(ROOT, name));

    describe(`[${name}] ${manifest.audio_length} samples -> global_embedding`, () => {
      // Encoded once per case in `beforeAll`, exactly as the decoder suite
      // does: a throw during collection names no stage, and one encode on the
      // reference conv1d chews through ~1e10 multiplies — seconds to a couple
      // of minutes, which is the design, not a bug. The hook timeout (600 s in
      // vitest.config.ts) is the budget.
      const stages: Record<string, Float32Array> = {};
      // The encode runs through a counting wrapper around the reference
      // backend: same numbers on every stage, but the counts observe that
      // `encodeGlobal` actually routes its heavy ops through the injected
      // seam. An implementation that quietly used `cpuBackend` directly would
      // pass every stage comparison and fail only the counter test below —
      // verified red by exactly that mutation.
      const calls = { matmul: 0, conv1d: 0 };
      const countingBackend: Backend = {
        name: "counting (reference)",
        matmul: (...args) => {
          calls.matmul += 1;
          return cpuBackend.matmul(...args);
        },
        conv1d: (...args) => {
          calls.conv1d += 1;
          return cpuBackend.conv1d(...args);
        },
        istft: (...args) => cpuBackend.istft(...args),
      };
      beforeAll(async () => {
        await encodeGlobal(golden.tensor("waveform_24k").data, weights, {
          backend: countingBackend,
          trace: (stage, data) => {
            stages[stage] = data;
          },
        });
      }, 600_000);

      it("routes the heavy ops through the injected backend", () => {
        // Exact counts by graph shape: convs = 1 resample + 7 frontend +
        // 1 pos_conv + 1 embed + 4 depthwise; matmuls = 1 projection +
        // 2 layers x 5 (gate, qkv, out, 2 FFN) + 4 blocks x 2 pointwise +
        // 1 final projection. Exact rather than `> 0`, so an op that silently
        // drops off the seam (or doubles) is caught, not just total absence.
        expect(calls.conv1d).toBe(14);
        expect(calls.matmul).toBe(20);
      });

      it("reproduces the recorded waveform padding", () => {
        // The golden records what `_calculate_waveform_padding` produced so a
        // drifting reimplementation is caught here, by name, rather than as an
        // off-by-a-few-samples resample failure two stages later.
        expect(calculateWaveformPadding(manifest.audio_length)).toBe(manifest.waveform_padding);
      });

      for (const stage of STAGES) {
        it(`${stage.name} — ${stage.note}`, () => {
          const expected = expectedFor(golden, stage.name);
          const actual = stages[stage.name];
          expect(actual, `stage ${stage.name} was not produced`).toBeDefined();
          expect(actual!.length).toBe(expected.data.length);

          const worst = worstDifference(actual!, expected.data);
          // Relative to the stage's own peak, as in `decoder.test.ts`.
          // Measured against a zero tolerance, the worst element sits between
          // **1.7e-7 and 6.8e-6** on every stage up to `global_input` in all
          // three cases, and between **6.1e-7 and 1.0e-4** downstream (the
          // 1e-4 extreme is jp_ref1's ConvNeXt tail — the reference's own
          // f32-vs-f64 disagreement, see the STAGES table). So the default
          // 1e-4 is 15x+ headroom where it applies. The failures it guards
          // against are not subtle: measured, disabling the encoder-level
          // LayerNorm after the pos_conv residual moved ssl_layer1 by
          // **1.12e0** relative (and every later stage with it), and
          // softmaxing the pooling weights over channels instead of time
          // moved attn_weights by **7.8e-1** — four orders of magnitude
          // above either tolerance, with every upstream stage still green,
          // so the first failure named the culprit both times.
          expect(worst.rel, `worst: ${JSON.stringify(worst)}`).toBeLessThan(stage.tolerance ?? 1e-4);
        });
      }
    });
  }
});
