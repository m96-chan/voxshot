import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { matvec } from "web-xpu-ops/ops/matvec";
import { loadCase, loadIndex, worstDifference } from "./golden.js";
import { buildWeights, MIOTTS_06B, type ModelWeights } from "./model.js";
import {
  generateQ8,
  GraphRunner,
  Qwen3RunnerQ8,
  type Graph,
  type GraphConfig,
} from "./model-q8.js";
import { SPEECH_TOKEN_BASE, SPEECH_TOKEN_COUNT } from "./tokenizer.js";
import { loadWeights } from "./weights-cache.js";
import { loadWeightsQ8FromDir } from "./weights-q8-node.js";

/**
 * The q8 CPU oracle (`model-q8.ts`) against the goldens, in three steps that
 * separate three failure modes:
 *
 *  1. **Graph parity** — the adjacent-pair (ops/rope) graph run over
 *     f32 weights the *test* permutes must match the golden to reference
 *     precision. Quantization is out of the picture, so a failure here is a
 *     wiring bug (wrong permutation, wrong rope convention, wrong norm),
 *     never "int8 is lossy".
 *  2. **Q8 accuracy** — the same graph over the real q8 artifacts, measured
 *     against the golden logits. A failure here (with 1 green) is
 *     quantization damage, not wiring.
 *  3. **Q8 greedy behaviour** — what the TTS pipeline actually consumes:
 *     speech tokens until eos, deterministic.
 */

const index = loadIndex();
const cfg = MIOTTS_06B;

// Loud when spike/miotts/q8 is absent (the loader names the rebuild command).
const HERE = new URL(".", import.meta.url).pathname;
const q8 = loadWeightsQ8FromDir(`${HERE}q8`);

/** Golden id tensors are f32 on disk; the ids themselves are exact integers. */
function idsOf(golden: ReturnType<typeof loadCase>, name: string): number[] {
  return Array.from(golden.tensor(name).data);
}

const graphConfig: GraphConfig = {
  numLayers: cfg.numLayers,
  hidden: cfg.hidden,
  heads: cfg.heads,
  kvHeads: cfg.kvHeads,
  headDim: cfg.headDim,
  ffn: cfg.ffn,
  vocab: cfg.vocab,
  ropeTheta: cfg.ropeTheta,
  rmsEps: cfg.rmsEps,
  eosIds: cfg.eosIds,
};

/**
 * The rotate-half -> adjacent-pair channel relabelling, written here from
 * first principles (j < 64 goes to 2j; j >= 64 goes to 2(j-64)+1) rather than
 * imported from `llm/weights.ts` — importing the implementation under test's
 * own permutation would make this a tautology.
 */
function pi(j: number, headDim: number): number {
  const half = headDim / 2;
  return j < half ? 2 * j : 2 * (j - half) + 1;
}

function permuteHeadRows(
  weight: Float32Array,
  heads: number,
  headDim: number,
  cols: number,
): Float32Array {
  const out = new Float32Array(weight.length);
  for (let h = 0; h < heads; h += 1) {
    for (let j = 0; j < headDim; j += 1) {
      const from = (h * headDim + j) * cols;
      const to = (h * headDim + pi(j, headDim)) * cols;
      out.set(weight.subarray(from, from + cols), to);
    }
  }
  return out;
}

function permuteGamma(gamma: Float32Array, headDim: number): Float32Array {
  const out = new Float32Array(headDim);
  for (let j = 0; j < headDim; j += 1) out[pi(j, headDim)] = gamma[j]!;
  return out;
}

/** f32 weight handle for the injected-linear graph: a plain [rows, cols] matrix. */
interface F32W {
  data: Float32Array;
  rows: number;
  cols: number;
}

/**
 * The q8 graph's exact structure over exact f32 arithmetic: the REAL
 * checkpoint's weights, with wq/wk rows and the q/k gammas permuted by the
 * test's own `pi`, and `matvec` injected as the linear. If this matches the
 * golden, the adjacent-pair graph *is* the HF rotate-half graph — the only
 * thing left between it and `Qwen3RunnerQ8` is int8.
 */
function permutedF32Graph(w: ModelWeights): Graph<F32W> {
  const { heads, kvHeads, headDim, hidden, vocab } = cfg;
  const asW = (data: Float32Array, rows: number, cols: number): F32W => ({ data, rows, cols });
  return {
    config: graphConfig,
    layers: w.layers.map((L) => ({
      attnNorm: L.inputNorm,
      wq: asW(permuteHeadRows(L.wq.data, heads, headDim, hidden), heads * headDim, hidden),
      wk: asW(permuteHeadRows(L.wk.data, kvHeads, headDim, hidden), kvHeads * headDim, hidden),
      wv: asW(L.wv.data, kvHeads * headDim, hidden),
      wo: asW(L.wo.data, hidden, heads * headDim),
      qNorm: permuteGamma(L.qNorm, headDim),
      kNorm: permuteGamma(L.kNorm, headDim),
      ffnNorm: L.postAttnNorm,
      wGate: asW(L.wGate.data, cfg.ffn, hidden),
      wUp: asW(L.wUp.data, cfg.ffn, hidden),
      wDown: asW(L.wDown.data, hidden, cfg.ffn),
    })),
    finalNorm: w.finalNorm,
    embedRow: (id) => w.embed.data.slice(id * hidden, (id + 1) * hidden),
    linear: (weight, vector) =>
      matvec({ matrix: weight.data, vector, M: weight.rows, K: weight.cols }),
    logits: (row) => matvec({ matrix: w.embed.data, vector: row, M: vocab, K: hidden }),
  };
}

describe("graph parity: permuted-f32 through the q8 graph equals the golden", () => {
  // Same bound as model.test.ts's stage checks: ~50x above the f32 noise
  // floor, far under the ~1e0 a wrong pairing produces.
  const REL = 1e-4;

  // One bf16->f32 load + permutation for both cases; dropped in afterAll so
  // the ~2.8 GB does not sit around while the q8 describes run.
  let parityGraph: Graph<F32W> | null = null;
  beforeAll(() => {
    parityGraph = permutedF32Graph(buildWeights(loadWeights(index.repo_id), cfg));
  });
  afterAll(() => {
    parityGraph = null;
  });

  for (const name of Object.keys(index.cases)) {
    it(`[${name}] prefill logits_last matches the golden at rel < ${REL}`, () => {
      const golden = loadCase(name);
      const prompt = idsOf(golden, "prompt_ids");
      const runner = new GraphRunner(parityGraph!, prompt.length);
      const logits = runner.prefill(prompt);
      const worst = worstDifference(logits, golden.tensor("logits_last").data);
      expect(
        worst.rel,
        `logits[${worst.index}] = ${worst.actual} vs ${worst.expected} (abs ${worst.abs})`,
      ).toBeLessThan(REL);
    });
  }
});

describe("q8 prefill accuracy against the golden logits", () => {
  // Peak-relative error of the q8 logits, per case, measured 2026-08-21 with
  // the pinned q8 artifacts (shas in q8/manifest.json):
  //   ja  worst rel 4.778e-2  (abs 1.47 on a ~30-peak logit vector)
  //   en  worst rel 1.249e-2  (abs 0.35)
  // The bound is ~2x the worst measurement: loose enough for a re-quantized
  // artifact's jitter, tight enough that a wiring regression (rel ~1e0,
  // measured by unwiring the permuted gammas) cannot hide behind "int8 is
  // lossy".
  const Q8_REL = 1e-1;

  for (const name of Object.keys(index.cases)) {
    it(`[${name}] logits_last within rel ${Q8_REL}, argmax = golden greedy64[0]`, () => {
      const golden = loadCase(name);
      const prompt = idsOf(golden, "prompt_ids");
      const runner = new Qwen3RunnerQ8(q8, prompt.length);
      const logits = runner.prefill(prompt);
      const expected = golden.tensor("logits_last").data;
      const worst = worstDifference(logits, expected);
      console.log(`[${name}] q8 logits_last worst rel ${worst.rel.toExponential(3)}`);
      expect(
        worst.rel,
        `logits[${worst.index}] = ${worst.actual} vs ${worst.expected} (abs ${worst.abs})`,
      ).toBeLessThan(Q8_REL);

      // Measured to hold for both cases with the pinned artifacts: int8's
      // logit error (max |Δ| ~1.5) does not flip the top-1 at the prompt's
      // last position.
      let best = 0;
      for (let i = 1; i < logits.length; i += 1) if (logits[i]! > logits[best]!) best = i;
      expect(best).toBe(idsOf(golden, "greedy64")[0]);
    });
  }
});

describe("q8 greedy generation (ja)", () => {
  const golden = loadCase("ja");
  const goldenFull = idsOf(golden, "greedy_full");
  let ids: number[] = [];

  beforeAll(() => {
    const prompt = idsOf(golden, "prompt_ids");
    const started = performance.now();
    ids = generateQ8(prompt, 128, q8, { stopAtEos: true });
    const elapsed = performance.now() - started;
    console.log(
      `[ja] q8 greedy: ${ids.length} ids, ${(elapsed / ids.length).toFixed(0)} ms/token`,
    );
  });

  it("emits only speech tokens, then eos", () => {
    expect(cfg.eosIds).toContain(ids[ids.length - 1]);
    for (const id of ids.slice(0, -1)) {
      expect(id, `id ${id} is not a speech token`).toBeGreaterThanOrEqual(SPEECH_TOKEN_BASE);
      expect(id).toBeLessThan(SPEECH_TOKEN_BASE + SPEECH_TOKEN_COUNT);
    }
  });

  it("lands in a sane length band for a short greeting", () => {
    // The f32 golden emits 84 ids (83 speech + eos) for this text at 25
    // tokens/s. A q8 run may diverge in *which* tokens, but a drastically
    // different length means the model lost the plot, not just a logit edge.
    expect(ids.length).toBeGreaterThanOrEqual(40);
    expect(ids.length).toBeLessThanOrEqual(128);
  });

  it("agrees with the f32 golden's greedy prefix", () => {
    let agree = 0;
    while (agree < ids.length && agree < goldenFull.length && ids[agree] === goldenFull[agree]) {
      agree += 1;
    }
    console.log(
      `[ja] q8-vs-golden greedy agreement: ${agree} ids ` +
        `(q8 ${ids.length}, golden ${goldenFull.length})`,
    );
    // Measured 2026-08-21 with the pinned artifacts: agreement prefix is 6
    // ids (q8 generated 88, golden 84 — greedy argmax under int8 diverges at
    // step 6 and finds a different, equally speech-token path to eos).
    // Floor = min(8, measured) = 6.
    expect(agree).toBeGreaterThanOrEqual(6);
  });

  it(
    "is deterministic: the same call twice yields identical ids",
    { timeout: 360_000 }, // two prefill+12-step runs at ~2.4 s/token
    () => {
      const prompt = idsOf(golden, "prompt_ids");
      // onLogits observes each step's pre-sampling logits: exactly one call
      // per emitted id, and under greedy the emitted id IS that vector's
      // argmax — this is what --dump-margins in expected-tokens.ts builds on.
      const observed: { step: number; argmax: number }[] = [];
      const a = generateQ8(prompt, 12, q8, {
        stopAtEos: true,
        onLogits: (step, logits) => {
          let best = 0;
          for (let i = 1; i < logits.length; i += 1) if (logits[i]! > logits[best]!) best = i;
          observed.push({ step, argmax: best });
        },
      });
      const b = generateQ8(prompt, 12, q8, { stopAtEos: true });
      expect(a).toEqual(b);
      expect(observed.map((o) => o.step)).toEqual(a.map((_, i) => i));
      expect(observed.map((o) => o.argmax)).toEqual(a);
      // And the long run above starts the same way — no state leaks between
      // runner instances.
      expect(ids.slice(0, a.length)).toEqual(a);
    },
  );
});
