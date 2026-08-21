/**
 * What q4 costs, measured — the numbers web-xpu-ops #137 asks for before any
 * q4 kernel exists.
 *
 * ## Why there is no kernel here
 *
 * A quantization format's damage can be observed without implementing its
 * GEMV: quantize the weights, dequantize them straight back to f32, and run
 * the **existing f32 reference forward** (`model.ts`) over the result. What
 * comes out is the quantization error alone — no kernel arithmetic, no packing
 * convention, nothing that could later be blamed for a number measured here.
 * W8A32/W4A32 makes this exact rather than approximate: `matvecQ8` keeps the
 * activation in f32 and applies the row scale after the dot product
 * (`ops/matvec/reference.ts`), so `sum_k (code * scale) * x_k` and
 * `(sum_k code * x_k) * scale` differ only in f32 rounding — which the q8
 * baseline check below confirms empirically against the numbers the real q8
 * artifacts produced.
 *
 * ## The baseline that makes the rest trustworthy
 *
 * Config `q8-row` reproduces the shipped q8 pipeline (`convert_weights.py` ->
 * `model-q8.ts`), whose measurements are recorded in `model-q8.test.ts`:
 * logits peak-relative 4.778e-2 (ja) / 1.249e-2 (en), and a greedy trajectory
 * that agrees with the f32 golden for 6 ids. If this harness does not
 * reproduce those, every other row of its output is noise. `f32` (no
 * quantization at all) is measured first for the same reason from the other
 * side: it must land on the golden at the reference noise floor (2.3e-6 for
 * logits_last, per model.test.ts) and reproduce all 84 golden greedy ids.
 *
 * ## Quantization conventions implemented
 *
 * All symmetric absmax, no zero point, matching `ops/quantize/reference.ts`'s
 * rounding bit-for-bit (`Math.round` semantics = ties toward +Infinity,
 * computed as `floor(x) + (x - floor(x) >= 0.5)` to avoid the double rounding
 * `floor(x + 0.5)` introduces; see `llm/tools/quant_common.py`'s module doc).
 *
 *   - `row`      : one scale per output row (the shipped q8 convention).
 *   - `g<N>`     : one scale per N consecutive **input** channels within a row.
 *                  Grouping runs along K, the dot product's reduction axis,
 *                  which is what makes a group's scale usable by a GEMV that
 *                  streams a row.
 *   - range      : `[-7, 7]` for q4 (`scale = absmax/7`), `[-127, 127]` for q8
 *                  — the existing `ops/quantize` convention, one code wasted at
 *                  the negative end in exchange for a symmetric grid.
 *   - `q4_0`     : the alternative the issue's decision 2 names, implemented as
 *                  llama.cpp's Q4_0 actually defines it (`ggml-quants.c`,
 *                  `quantize_row_q4_0_ref`, read not recalled): `d = max / -8`
 *                  where `max` is the **signed** value of largest magnitude, so
 *                  the extreme element lands on code -8 and never clamps; the
 *                  opposite tail is what gets clipped, at `d` granularity.
 *                  Codes span `[-8, 7]`, using all 16 bins. llama.cpp stores
 *                  `d` as fp16 and fixes the block at 32; this harness keeps
 *                  the f32 scale and the configured group size so the *range*
 *                  is the only thing that differs from the row above it.
 *
 * An all-zero group gets `scale = 1` and codes of 0, `quantize()`'s own guard.
 *
 * The two conventions also disagree about the reciprocal used before rounding
 * (`div/absmax` in f64 versus llama.cpp's `1.0f/d` from the narrowed f32
 * scale). That is not cosmetic — see `quantizeGroup`.
 *
 * ## Running
 *
 *   cd spike/miotts && npx tsx measure_q4.ts                  # everything
 *   npx tsx measure_q4.ts --configs f32,q8-row                # a subset
 *   npx tsx measure_q4.ts --no-family                         # skip per-family
 *   npx tsx measure_q4.ts --configs none --family-specs q4-row  # only the sweep
 *   npx tsx measure_q4.ts --max-new 128 --json out.json
 *
 * A config takes ~35-50 s (two 15-token prompts plus one greedy continuation
 * of up to `--max-new` tokens, at ~300 ms/token through the f32 reference).
 * Everything is deterministic: two runs of the same config produce identical
 * numbers, verified by re-running the table.
 *
 * Needs the goldens and the checkpoint (`python3 dump_golden.py`).
 */
import { writeFileSync } from "node:fs";
import { loadCase, loadIndex, worstDifference } from "./golden.js";
import { buildWeights, MIOTTS_06B, Qwen3Runner, type ModelWeights } from "./model.js";
import { SPEECH_TOKEN_BASE, SPEECH_TOKEN_COUNT } from "./tokenizer.js";
import { loadWeights, type Tensor } from "./weights-cache.js";

// ---------------------------------------------------------------- quantizing

export interface QuantSpec {
  /** Codes per weight. 8 -> [-127, 127], 4 -> [-7, 7] (or Q4_0's [-8, 7]). */
  bits: 4 | 8;
  /** Input channels sharing one scale; `null` = the whole row. */
  groupSize: number | null;
  /** q4 only: llama.cpp Q4_0's `d = max/-8`, [-8, 7], instead of absmax/7. */
  q4_0?: boolean;
}

/** Peak and RMS error of a quantized tensor, both relative to its absmax. */
export interface WeightError {
  peak: number;
  rms: number;
}

/**
 * `dst[i] = dequantize(quantize(src[i]))` over one group.
 *
 * Returns the group's absmax, its worst absolute error, and the summed squared
 * error: the peak is what `convert_weights.py` reports, but the peak of a q4
 * group is *always* the range's own bound, so it cannot distinguish group 64
 * from group 128 — the RMS can, and that is the number the group-size decision
 * actually turns on.
 */
function quantizeGroup(
  src: Float32Array,
  dst: Float32Array,
  at: number,
  len: number,
  spec: QuantSpec,
): { absmax: number; worst: number; sumSq: number } {
  let absmax = 0;
  // llama.cpp's Q4_0 keeps the *signed* value of largest magnitude (strict
  // `>`, first one wins on a tie) — the sign is what makes `d = max/-8` put
  // that element on code -8 rather than clamping it.
  let signedMax = 0;
  for (let i = 0; i < len; i += 1) {
    const v = src[at + i]!;
    const a = Math.abs(v);
    if (a > absmax) {
      absmax = a;
      signedMax = v;
    }
  }
  if (absmax === 0) {
    for (let i = 0; i < len; i += 1) dst[at + i] = 0;
    return { absmax: 0, worst: 0, sumSq: 0 };
  }
  const div = spec.bits === 8 ? 127 : spec.q4_0 ? 8 : 7;
  const lo = spec.bits === 8 ? -127 : spec.q4_0 ? -8 : -7;
  const hi = spec.bits === 8 ? 127 : 7;
  // The scale narrows to f32 (that is how it goes on the wire); the code ->
  // value product is f64 and narrows again on the Float32Array store, exactly
  // as `quant_common.dequantize_per_row` into an f32 output does.
  const scale = Math.fround(spec.q4_0 ? signedMax / -div : absmax / div);
  // The two conventions disagree about the reciprocal, and it is not cosmetic:
  // `ops/quantize` (and `quant_common.quantize_per_row`) divides by the *exact*
  // absmax (`div/absmax`, f64), while llama.cpp computes `id = 1.0f/d` from the
  // already-narrowed f32 scale. Using one for the other moves codes at the .5
  // boundaries — measured: it shifted the q8 baseline's ja logits error from
  // 4.778e-2 to 4.695e-2, i.e. enough to break the reproduction this harness is
  // validated by. Each convention keeps its own.
  const inverse = spec.q4_0 ? Math.fround(1 / scale) : div / absmax;
  let worst = 0;
  let sumSq = 0;
  for (let i = 0; i < len; i += 1) {
    const value = src[at + i]!;
    const scaled = value * inverse;
    const floored = Math.floor(scaled);
    let code = scaled - floored >= 0.5 ? floored + 1 : floored;
    if (code < lo) code = lo;
    else if (code > hi) code = hi;
    dst[at + i] = code * scale;
    const err = dst[at + i]! - value;
    const a = Math.abs(err);
    if (a > worst) worst = a;
    sumSq += err * err;
  }
  return { absmax, worst, sumSq };
}

/** One `[rows, cols]` matrix through quantize -> dequantize, src -> dst. */
export function quantizeMatrix(
  src: Float32Array,
  dst: Float32Array,
  rows: number,
  cols: number,
  spec: QuantSpec,
): WeightError {
  const group = spec.groupSize ?? cols;
  let tensorAbsmax = 0;
  let tensorWorst = 0;
  let sumSq = 0;
  for (let r = 0; r < rows; r += 1) {
    const base = r * cols;
    for (let c = 0; c < cols; c += group) {
      const len = Math.min(group, cols - c);
      const g = quantizeGroup(src, dst, base + c, len, spec);
      if (g.absmax > tensorAbsmax) tensorAbsmax = g.absmax;
      if (g.worst > tensorWorst) tensorWorst = g.worst;
      sumSq += g.sumSq;
    }
  }
  if (tensorAbsmax === 0) return { peak: 0, rms: 0 };
  return {
    peak: tensorWorst / tensorAbsmax,
    rms: Math.sqrt(sumSq / (rows * cols)) / tensorAbsmax,
  };
}

// ------------------------------------------------------------------ families

/**
 * The eight quantizable tensor families. `embed` is the embedding table and,
 * because this checkpoint ties them, the lm_head matrix too — a q4 embed
 * degrades both the input rows and every logit.
 */
export const FAMILIES = [
  "embed",
  "q_proj",
  "k_proj",
  "v_proj",
  "o_proj",
  "gate_proj",
  "up_proj",
  "down_proj",
] as const;
export type Family = (typeof FAMILIES)[number];

/** Every tensor of a family, as `[src, dst]` pairs plus its shape. */
function tensorsOf(
  family: Family,
  master: ModelWeights,
  work: ModelWeights,
): { src: Tensor; dst: Tensor }[] {
  if (family === "embed") return [{ src: master.embed, dst: work.embed }];
  const pick = (L: ModelWeights["layers"][number]): Tensor => {
    switch (family) {
      case "q_proj":
        return L.wq;
      case "k_proj":
        return L.wk;
      case "v_proj":
        return L.wv;
      case "o_proj":
        return L.wo;
      case "gate_proj":
        return L.wGate;
      case "up_proj":
        return L.wUp;
      case "down_proj":
        return L.wDown;
    }
  };
  return master.layers.map((L, i) => ({ src: pick(L), dst: pick(work.layers[i]!) }));
}

/** Copies a family back to its unquantized values (used between configs). */
function restoreFamily(family: Family, master: ModelWeights, work: ModelWeights): void {
  for (const { src, dst } of tensorsOf(family, master, work)) dst.data.set(src.data);
}

function applyFamily(
  family: Family,
  master: ModelWeights,
  work: ModelWeights,
  spec: QuantSpec,
): WeightError {
  // Worst tensor of the family on each statistic; a family's tensors all have
  // the same shape, so the two maxima are directly comparable across families.
  let out: WeightError = { peak: 0, rms: 0 };
  for (const { src, dst } of tensorsOf(family, master, work)) {
    const [rows, cols] = src.shape as [number, number];
    const e = quantizeMatrix(src.data, dst.data, rows, cols, spec);
    out = { peak: Math.max(out.peak, e.peak), rms: Math.max(out.rms, e.rms) };
  }
  return out;
}

// ----------------------------------------------------------------- measuring

const cfg = MIOTTS_06B;

export interface CaseResult {
  name: string;
  /** Worst |logit - golden| / peak(|golden|) — the issue's headline statistic. */
  rel: number;
  abs: number;
  /** RMS of the same difference over all 164480 logits, also peak-relative. */
  rmsRel: number;
  argmaxMatches: boolean;
}

/** RMS difference, normalised by the expected vector's peak magnitude. */
function rmsRelative(actual: Float32Array, expected: Float32Array): number {
  let peak = 0;
  let sumSq = 0;
  for (let i = 0; i < expected.length; i += 1) peak = Math.max(peak, Math.abs(expected[i]!));
  for (let i = 0; i < actual.length; i += 1) {
    const d = actual[i]! - expected[i]!;
    sumSq += d * d;
  }
  return Math.sqrt(sumSq / actual.length) / peak;
}

/** What each family is quantized to; a family absent from the map stays f32. */
export type Recipe = Partial<Record<Family, QuantSpec>>;

/** The same spec for every quantizable family — configs 1-4 of the issue. */
function uniform(spec: QuantSpec): Recipe {
  return Object.fromEntries(FAMILIES.map((f) => [f, spec])) as Recipe;
}

export interface Measurement {
  config: string;
  recipe: Recipe;
  /** Peak and RMS `|dequant - src| / absmax` per quantized family. */
  weightError: Partial<Record<Family, WeightError>>;
  cases: CaseResult[];
  greedy: {
    ids: number;
    agree: number;
    eos: boolean;
    /**
     * Every emitted id before eos is a speech token — what the codec stage
     * actually requires, and a coarser question than "did it match the golden".
     * A run can diverge from the golden at step 1 and still be usable; a run
     * that emits text tokens has lost the plot.
     */
    speechOnly: boolean;
    goldenIds: number;
    /** The ids themselves, so a reader can diff two runs without re-running. */
    emitted: number[];
    msPerToken: number;
  } | null;
  seconds: number;
}

function idsOf(golden: ReturnType<typeof loadCase>, name: string): number[] {
  return Array.from(golden.tensor(name).data);
}

function argmax(logits: Float32Array): number {
  let best = 0;
  for (let i = 1; i < logits.length; i += 1) if (logits[i]! > logits[best]!) best = i;
  return best;
}

/**
 * The prompt fed one token at a time rather than as a batch.
 *
 * `Qwen3Runner.prefill` with T > 1 goes through `matmul`, which needs a
 * transposed copy of every weight — and `model.ts` memoises those by array
 * identity, so a harness that mutates weights in place between configs would
 * silently reuse the *previous* config's transposes. Feeding the prompt
 * through the KV-cached decode path keeps every projection on `matvec`, which
 * reads the weight in place. The two paths accumulate identically (both in
 * f64 over ascending k, per `model.ts#linear`), and the f32 config verifies
 * that end to end against the golden.
 */
function promptLogits(weights: ModelWeights, prompt: number[], maxSeq: number): {
  runner: Qwen3Runner;
  logits: Float32Array;
} {
  const runner = new Qwen3Runner(weights, cfg, maxSeq);
  let logits = runner.prefill([prompt[0]!]);
  for (let i = 1; i < prompt.length; i += 1) logits = runner.decodeStep(prompt[i]!);
  return { runner, logits };
}

function greedyFrom(runner: Qwen3Runner, logits: Float32Array, maxNew: number): number[] {
  const out: number[] = [];
  let current = logits;
  for (let step = 0; step < maxNew; step += 1) {
    const id = argmax(current);
    out.push(id);
    if (cfg.eosIds.includes(id)) break;
    if (step + 1 < maxNew) current = runner.decodeStep(id);
  }
  return out;
}

interface Options {
  maxNew: number;
  greedy: boolean;
}

function measure(
  label: string,
  recipe: Recipe,
  master: ModelWeights,
  work: ModelWeights,
  opts: Options,
): Measurement {
  const started = performance.now();
  const weightError: Partial<Record<Family, WeightError>> = {};
  for (const f of FAMILIES) {
    const spec = recipe[f];
    if (spec) weightError[f] = applyFamily(f, master, work, spec);
  }

  const cases: CaseResult[] = [];
  let greedy: Measurement["greedy"] = null;
  for (const name of Object.keys(loadIndex().cases)) {
    const golden = loadCase(name);
    const prompt = idsOf(golden, "prompt_ids");
    const wantGreedy = opts.greedy && Boolean(golden.manifest.tensors["greedy_full"]);
    const maxSeq = prompt.length + (wantGreedy ? opts.maxNew : 1);
    const { runner, logits } = promptLogits(work, prompt, maxSeq);
    const expected = golden.tensor("logits_last").data;
    const worst = worstDifference(logits, expected);
    cases.push({
      name,
      rel: worst.rel,
      abs: worst.abs,
      rmsRel: rmsRelative(logits, expected),
      argmaxMatches: argmax(logits) === idsOf(golden, "greedy64")[0],
    });
    if (wantGreedy) {
      const at = performance.now();
      const ids = greedyFrom(runner, logits, opts.maxNew);
      const goldenFull = idsOf(golden, "greedy_full");
      let agree = 0;
      while (agree < ids.length && agree < goldenFull.length && ids[agree] === goldenFull[agree]) {
        agree += 1;
      }
      const eos = cfg.eosIds.includes(ids[ids.length - 1]!);
      const body = eos ? ids.slice(0, -1) : ids;
      greedy = {
        ids: ids.length,
        agree,
        eos,
        speechOnly: body.every(
          (id) => id >= SPEECH_TOKEN_BASE && id < SPEECH_TOKEN_BASE + SPEECH_TOKEN_COUNT,
        ),
        goldenIds: goldenFull.length,
        emitted: ids,
        msPerToken: (performance.now() - at) / ids.length,
      };
    }
  }

  // Back to f32 so the next config starts from the checkpoint, not from
  // whatever the last one left behind.
  for (const f of FAMILIES) if (recipe[f]) restoreFamily(f, master, work);

  return {
    config: label,
    recipe,
    weightError,
    cases,
    greedy,
    seconds: (performance.now() - started) / 1000,
  };
}

// -------------------------------------------------------------------- report

function fmt(m: Measurement): string {
  const cases = m.cases
    .map(
      (c) =>
        `${c.name} peak ${c.rel.toExponential(3)} rms ${c.rmsRel.toExponential(3)} ` +
        `(abs ${c.abs.toFixed(3)}, argmax ${c.argmaxMatches ? "ok" : "FLIPPED"})`,
    )
    .join("  |  ");
  const g = m.greedy
    ? `greedy ${m.greedy.ids} ids (golden ${m.greedy.goldenIds}), agree ${m.greedy.agree}, ` +
      `eos ${m.greedy.eos}, speech-only ${m.greedy.speechOnly}`
    : "greedy skipped";
  const w = Object.entries(m.weightError)
    .map(([k, v]) => `${k} ${(v as WeightError).rms.toExponential(2)}`)
    .join(" ");
  return `${m.config.padEnd(22)} ${cases}\n${" ".repeat(22)} ${g}${w ? `\n${" ".repeat(22)} weight rms err/absmax: ${w}` : ""}\n${" ".repeat(22)} ${m.seconds.toFixed(1)}s`;
}

// ---------------------------------------------------------------------- main

const Q8_ROW: QuantSpec = { bits: 8, groupSize: null };
const Q4_G128: QuantSpec = { bits: 4, groupSize: 128 };

const CONFIGS: { label: string; recipe: Recipe }[] = [
  { label: "f32", recipe: {} },
  { label: "q8-row", recipe: uniform(Q8_ROW) },
  { label: "q4-row", recipe: uniform({ bits: 4, groupSize: null }) },
  { label: "q4-g64", recipe: uniform({ bits: 4, groupSize: 64 }) },
  { label: "q4-g128", recipe: uniform(Q4_G128) },
  // The range decision (issue #137's open question 2), holding everything else
  // at q4-g128: llama.cpp Q4_0's [-8, 7] against ops/quantize's [-7, 7]. The
  // g32 row is Q4_0's own block size, i.e. the format the ecosystem ships.
  { label: "q4_0-g128", recipe: uniform({ bits: 4, groupSize: 128, q4_0: true }) },
  { label: "q4_0-g32", recipe: uniform({ bits: 4, groupSize: 32, q4_0: true }) },
  // Mixed width: q4 everywhere except the tensors the per-family sweep found
  // most damaging. `embed` is the tied lm_head, so it pays twice.
  { label: "mix-embed8", recipe: { ...uniform(Q4_G128), embed: Q8_ROW } },
  {
    label: "mix-embed8-vo8",
    recipe: { ...uniform(Q4_G128), embed: Q8_ROW, v_proj: Q8_ROW, o_proj: Q8_ROW },
  },
];

/**
 * Wire size of the whole model under one spec: codes at `bits` bits each plus
 * one f32 scale per group.
 *
 * Scales are counted as f32 because that is what `ops/quantize` emits and what
 * `matvecQ8` reads; llama.cpp stores its block scale as fp16, which would halve
 * the second term (noted rather than assumed away — it is a format decision the
 * issue has not made).
 */
function wireBytes(weights: ModelWeights, recipe: Recipe): { bytes: number; bpw: number } {
  let bytes = 0;
  let params = 0;
  for (const f of FAMILIES) {
    const spec = recipe[f];
    for (const { src } of tensorsOf(f, weights, weights)) {
      const [rows, cols] = src.shape as [number, number];
      params += rows * cols;
      if (!spec) {
        bytes += rows * cols * 4;
        continue;
      }
      const groups = rows * Math.ceil(cols / (spec.groupSize ?? cols));
      bytes += Math.ceil((rows * cols * spec.bits) / 8) + groups * 4;
    }
  }
  return { bytes, bpw: (bytes * 8) / params };
}

function parseArgs(argv: string[]): {
  configs: string[] | null;
  familySpecs: string[];
  maxNew: number;
  json: string | null;
  greedy: boolean;
} {
  const get = (flag: string): string | null => {
    const at = argv.indexOf(flag);
    return at >= 0 && at + 1 < argv.length ? argv[at + 1]! : null;
  };
  return {
    configs: get("--configs")?.split(",") ?? null,
    // Which specs the per-family sweep uses; "" disables it.
    familySpecs: argv.includes("--no-family")
      ? []
      : (get("--family-specs") ?? "q4-row,q4-g128").split(",").filter(Boolean),
    maxNew: Number(get("--max-new") ?? 128),
    json: get("--json"),
    greedy: !argv.includes("--no-greedy"),
  };
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  const index = loadIndex();
  console.log(`[measure] checkpoint ${index.repo_id}, cases ${Object.keys(index.cases).join(", ")}`);

  const t0 = performance.now();
  const master = buildWeights(loadWeights(index.repo_id), cfg);
  // The forward runs over `work`; `master` stays at the checkpoint's f32
  // values so every config quantizes the same source.
  const clone = (t: Tensor): Tensor => ({ data: Float32Array.from(t.data), shape: [...t.shape] });
  const work: ModelWeights = {
    embed: clone(master.embed),
    finalNorm: master.finalNorm,
    layers: master.layers.map((L) => ({
      ...L,
      wq: clone(L.wq),
      wk: clone(L.wk),
      wv: clone(L.wv),
      wo: clone(L.wo),
      wGate: clone(L.wGate),
      wUp: clone(L.wUp),
      wDown: clone(L.wDown),
    })),
  };
  console.log(`[measure] weights loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

  for (const { label, recipe } of CONFIGS) {
    const { bytes, bpw } = wireBytes(master, recipe);
    console.log(
      `[size] ${label.padEnd(16)} ${(bytes / 1024 / 1024).toFixed(1).padStart(7)} MiB  ` +
        `${bpw.toFixed(3)} bits/weight (quantizable tensors only; norms are f32 either way)`,
    );
  }

  const opts: Options = { maxNew: args.maxNew, greedy: args.greedy };
  const results: Measurement[] = [];
  for (const { label, recipe } of CONFIGS) {
    if (args.configs && !args.configs.includes(label)) continue;
    const m = measure(label, recipe, master, work, opts);
    results.push(m);
    console.log(fmt(m));
  }

  // Layer sensitivity: one family at a time, everything else left at f32.
  for (const specLabel of args.familySpecs) {
    const base = CONFIGS.find((c) => c.label === specLabel)?.recipe;
    const spec = base?.[FAMILIES[0]];
    if (!spec) throw new Error(`--family-specs names ${specLabel}, which is not a uniform config`);
    for (const family of FAMILIES) {
      const m = measure(`${specLabel}:${family}`, { [family]: spec }, master, work, opts);
      results.push(m);
      console.log(fmt(m));
    }
  }

  if (args.json) {
    writeFileSync(args.json, `${JSON.stringify(results, null, 2)}\n`);
    console.log(`[measure] wrote ${args.json}`);
  }
}

main();
