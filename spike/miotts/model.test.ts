import { beforeAll, describe, expect, it } from "vitest";
import { loadCase, loadIndex, worstDifference } from "./golden.js";
import {
  buildWeights,
  greedyGenerate,
  MIOTTS_06B,
  Qwen3Runner,
  type ModelWeights,
} from "./model.js";
import { loadWeights } from "./weights-cache.js";

/**
 * The Qwen3-0.6B forward pass, stage by stage, against the reference model's
 * own intermediates (`dump_golden.py`).
 *
 * Checked only at the sampler a port learns that something is wrong and
 * nothing about where — every layer's error arrives at the logits together.
 * The stages here are ordered as the graph runs, so the **first** failure
 * names the culprit; layer 0's attention is opened further (q_proj, q_norm,
 * RoPE, o_proj) because those are the ops most likely to be subtly wrong
 * (half-rotation pairing, per-head QK-norm, GQA head mapping) in ways that
 * still produce plausible hidden states.
 */

const index = loadIndex();
const cfg = MIOTTS_06B;

// One load + bf16->f32 expansion for the whole file. ~2.4 GB of f32 — fine on
// this machine, and the alternative is paying the conversion once per describe.
const weights: ModelWeights = buildWeights(loadWeights(index.repo_id), cfg);

/** Golden id tensors are f32 on disk; the ids themselves are exact integers. */
function idsOf(golden: ReturnType<typeof loadCase>, name: string): number[] {
  return Array.from(golden.tensor(name).data);
}

/**
 * Greedy continuation with a step-by-step divergence diagnosis: on the first
 * id that disagrees with `expected`, records the logit margin between the
 * expected and produced id at that step — the number that says whether the
 * miss is a real bug (large margin) or a knife-edge tie (tiny one).
 */
function greedyDiagnosed(
  promptIds: number[],
  expected: number[],
  stopAtEos: boolean,
): { ids: number[]; report: string } {
  const runner = new Qwen3Runner(weights, cfg, promptIds.length + expected.length + 1);
  let logits = runner.prefill(promptIds);
  const ids: number[] = [];
  for (let step = 0; step < expected.length; step += 1) {
    let best = 0;
    for (let i = 1; i < logits.length; i += 1) if (logits[i]! > logits[best]!) best = i;
    ids.push(best);
    const want = expected[step]!;
    if (best !== want) {
      const margin = logits[want]! - logits[best]!;
      return {
        ids,
        report:
          `step ${step}: produced ${best} (logit ${logits[best]!.toFixed(4)}), ` +
          `expected ${want} (logit ${logits[want]!.toFixed(4)}, margin ${margin.toExponential(3)})`,
      };
    }
    if (stopAtEos && cfg.eosIds.includes(best)) break;
    if (step + 1 < expected.length) logits = runner.decodeStep(best);
  }
  return { ids, report: "" };
}

/** The stage tensors, in golden (graph) order. */
const STAGES: string[] = [
  "embedding",
  ...Array.from({ length: cfg.numLayers }, (_, i) => `layer${String(i).padStart(2, "0")}`),
  "l0_q_proj",
  "l0_q_normed",
  "l0_q_roped",
  "l0_attn_out",
  "final_norm",
  "logits_last",
];

/**
 * Peak-relative bound, one for every stage. Measured noise floors (worst rel
 * over both cases, this port's f64-accumulating reference ops against the
 * torch f32 eager reference, 2026-08-21):
 *
 *   embedding        0           (gather is copying, bit-exact)
 *   l0_q_proj        4.7e-7
 *   l0_q_normed      5.6e-7
 *   l0_q_roped       5.6e-7     (f64 cos/sin against torch's f32 table)
 *   l0_attn_out      3.1e-7
 *   layer00..27      <= 7.8e-7   (flat with depth, ~4.3e-7 typical)
 *   final_norm       1.8e-6
 *   logits_last      2.3e-6
 *
 * 1e-4 sits ~50x above the worst of those — loose enough not to flake, tight
 * enough that a wrong RoPE pairing (rel ~1e0, measured with RoPE unwired) or
 * a swapped norm cannot hide.
 */
const REL = 1e-4;

describe("MioTTS-0.6B LM against the golden", () => {
  // The config the golden was dumped under must be the config this port
  // hardcodes; a checkpoint swap should fail here, not as a shape error.
  it("index config matches the port's config", () => {
    expect(index.config.num_layers).toBe(cfg.numLayers);
    expect(index.config.hidden).toBe(cfg.hidden);
    expect(index.config.heads).toBe(cfg.heads);
    expect(index.config.kv_heads).toBe(cfg.kvHeads);
    expect(index.config.head_dim).toBe(cfg.headDim);
    expect(index.config.ffn).toBe(cfg.ffn);
    expect(index.config.vocab).toBe(cfg.vocab);
    expect(index.config.rope_theta).toBe(cfg.ropeTheta);
    expect(index.config.rms_eps).toBe(cfg.rmsEps);
    expect(index.config.eos_ids).toEqual(cfg.eosIds);
    expect(index.config.tie_word_embeddings).toBe(true);
  });

  for (const name of Object.keys(index.cases)) {
    const golden = loadCase(name);
    const manifest = golden.manifest;

    describe(`[${name}] "${manifest.content}" (${manifest.prompt_tokens} prompt tokens)`, () => {
      describe("prefill stages", () => {
        // One prefill per case, traced, in `beforeAll` rather than the
        // describe body — a throw during collection names no stage at all
        // (measured on the miocodec suite), and the reference ops take
        // seconds, which the per-test timeout would eat.
        const stages = new Map<string, Float32Array>();
        beforeAll(() => {
          const runner = new Qwen3Runner(weights, cfg, manifest.prompt_tokens);
          runner.prefill(idsOf(golden, "prompt_ids"), (stage, data) => stages.set(stage, data));
        });

        for (const stage of STAGES) {
          it(stage, () => {
            const expected = golden.tensor(stage);
            const actual = stages.get(stage);
            expect(actual, `stage ${stage} was not traced`).toBeDefined();
            expect(actual!.length).toBe(expected.data.length);
            const worst = worstDifference(actual!, expected.data);
            expect(
              worst.rel,
              `${stage}[${worst.index}] = ${worst.actual} vs ${worst.expected} (abs ${worst.abs})`,
            ).toBeLessThan(REL);
          });
        }
      });

      describe("greedy decode (KV-cache path)", () => {
        // Exact id equality over every step — a single wrong logit ordering
        // anywhere in 28 layers x 64 steps shows up here. Generated once per
        // case; ~0.6 GFLOP-equivalent per step on the reference ops.
        let ids64: number[] = [];
        let diagnosis = "";
        beforeAll(() => {
          const prompt = idsOf(golden, "prompt_ids");
          const expected = idsOf(golden, "greedy64");
          const started = performance.now();
          // The golden's manual-loop semantics: argmax, never stopping at eos.
          ids64 = greedyGenerate(prompt, 64, weights, cfg, { stopAtEos: false });
          const elapsed = performance.now() - started;
          console.log(`[${name}] greedy64: ${(elapsed / 64).toFixed(0)} ms/token`);
          if (ids64.join(",") !== expected.join(",")) {
            diagnosis = greedyDiagnosed(prompt, expected, false).report;
          }
        });

        it("greedy64 ids are exactly the golden's", () => {
          expect(ids64, diagnosis).toEqual(idsOf(golden, "greedy64"));
        });

        if (manifest.tensors["greedy_full"]) {
          let idsFull: number[] = [];
          let fullDiagnosis = "";
          beforeAll(() => {
            const prompt = idsOf(golden, "prompt_ids");
            const expected = idsOf(golden, "greedy_full");
            const started = performance.now();
            // generate()'s semantics: stop at (and include) eos. maxNew is the
            // dump's 512 cap, so a broken stop condition runs long and fails
            // on length rather than passing by construction.
            idsFull = greedyGenerate(prompt, 512, weights, cfg, { stopAtEos: true });
            const elapsed = performance.now() - started;
            console.log(
              `[${name}] greedy_full: ${idsFull.length} ids, ` +
                `${(elapsed / idsFull.length).toFixed(0)} ms/token`,
            );
            if (idsFull.join(",") !== expected.join(",")) {
              fullDiagnosis = greedyDiagnosed(prompt, expected, true).report;
            }
          });

          it("greedy_full ids are exactly the golden's, ending at eos", () => {
            const expected = idsOf(golden, "greedy_full");
            expect(idsFull, fullDiagnosis).toEqual(expected);
            expect(cfg.eosIds).toContain(idsFull[idsFull.length - 1]);
          });
        }
      });
    });
  }

  describe("decode step against prefill (ja)", () => {
    const golden = loadCase("ja");
    let decodeLogits: Float32Array;
    let prefillLogits: Float32Array;
    let decoded: Qwen3Runner;
    let recomputed: Qwen3Runner;
    let promptLength = 0;
    let kBefore: Float32Array;
    let kAfter: Float32Array;

    beforeAll(() => {
      const prompt = idsOf(golden, "prompt_ids");
      const next = idsOf(golden, "greedy64")[0]!;
      promptLength = prompt.length;

      // Path A: prefill the prompt, then one decode step for `next`.
      decoded = new Qwen3Runner(weights, cfg, prompt.length + 1);
      decoded.prefill(prompt);
      kBefore = decoded.cache.read(0, prompt.length).k;
      decodeLogits = decoded.decodeStep(next);
      kAfter = decoded.cache.read(0, prompt.length).k;

      // Path B: prefill the extended sequence from scratch.
      recomputed = new Qwen3Runner(weights, cfg, prompt.length + 1);
      prefillLogits = recomputed.prefill([...prompt, next]);
    });

    it("one decode step's logits match recomputing the extended prefill", () => {
      // Same ops, same accumulation order — the two paths differ only in how
      // K/V reach the attention (cache vs freshly projected), so anything
      // beyond noise here is a cache-indexing bug.
      const worst = worstDifference(decodeLogits, prefillLogits);
      expect(worst.rel, `logits[${worst.index}]: ${worst.actual} vs ${worst.expected}`).toBeLessThan(
        1e-6,
      );
    });

    it("a decode step appends to the KV cache without disturbing prefilled entries", () => {
      expect(kAfter.length).toBe(kBefore.length);
      const worst = worstDifference(kAfter, kBefore);
      expect(worst.abs, `k[${worst.index}] moved by ${worst.abs}`).toBe(0);
    });

    it("the cache holds the same K/V a from-scratch prefill computes", () => {
      for (const layer of [0, cfg.numLayers - 1]) {
        const a = decoded.cache.read(layer, promptLength + 1);
        const b = recomputed.cache.read(layer, promptLength + 1);
        const worstK = worstDifference(a.k, b.k);
        expect(worstK.rel, `layer ${layer} k[${worstK.index}]`).toBeLessThan(1e-6);
        const worstV = worstDifference(a.v, b.v);
        expect(worstV.rel, `layer ${layer} v[${worstV.index}]`).toBeLessThan(1e-6);
      }
    });
  });
});
