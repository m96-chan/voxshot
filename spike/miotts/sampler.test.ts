import { describe, expect, it } from "vitest";
import { sampleNext, type Constraint, type SamplerOptions } from "../../../web-xpu-ops/llm/sampler.js";
import { loadCase } from "./golden.js";
import {
  DEFAULT_TOP_K,
  createSamplerStats,
  sampleNextTopK,
  topKMass,
  xorshift32,
} from "../../src/engine/miotts/lm/sampler.js";

/**
 * `sampler.ts` against the upstream `llm/sampler.ts` it stands in front of.
 *
 * The whole value of the wrapper is that it is FASTER while picking the SAME
 * id, so the suite is built around exactly those two observations:
 *
 *  - **agreement** — same logits + same draw => same id, over hundreds of
 *    draws spanning the whole [0, 1) range and every logits vector to hand. A
 *    wrapper that quietly changed the distribution would still look fine on a
 *    single draw, and would still produce speech; only a sweep catches it, and
 *    only one that reaches past the window (see `agreementSweep`).
 *  - **speed** — a floor on the speedup, because "same id" is trivially
 *    satisfiable by delegating everything to upstream, which is what the
 *    fallback path does. Without this assertion the agreement test would stay
 *    green for a change that fixed nothing.
 *
 * The tail-mass test is the third leg: it records what the truncation
 * actually costs on this model, so `DEFAULT_TOP_K` is a measured number and
 * not a folk one, and so a future temperature/vocabulary change that
 * invalidates the measurement fails here instead of silently degrading.
 *
 * The two golden vectors are the real logits available to a checkout. The
 * agreement was ALSO swept over 60 consecutive decode steps of a real sampled
 * generation while #120 was being investigated — 240 draws each, 14,400 in
 * total, **0 disagreements**, 56 fallbacks (0.39%, against a measured mean
 * tail of 4.1e-3). Those vectors are 39 MB of raw floats and are not in the
 * repo, so that sweep lives in the ISSUE rather than here. To repeat it, run
 * `generateQ8` with `onLogits` and feed each step through `agreementSweep`.
 */

/** The real decode-step logits from the goldens (`dump_golden.py`'s prefill last position). */
const goldenLogits: { name: string; data: Float32Array }[] = ["ja", "en"].map((name) => ({
  name,
  data: loadCase(name).tensor("logits_last").data,
}));

/**
 * The reference server's own sampling settings — the only ones the TTS page
 * ever asks for, so they are the ones the agreement sweep must cover.
 */
const MIOTTS_SAMPLER: SamplerOptions = { mode: "top-p", temperature: 0.8, topP: 1.0 };

/** An rng that always returns the same value — one point of the draw's range. */
function fixedDraw(options: SamplerOptions, u: number): SamplerOptions {
  return options.mode === "greedy" ? options : { ...options, rng: () => u };
}

/**
 * Both samplers over an even grid of `draws` values of the uniform draw, and
 * where they differed.
 *
 * **The grid, rather than a sweep of seeds, is the whole point.** The obvious
 * version of this test — `xorshift32(seed)` for seed = 1..300 — is green
 * against a sampler with no fallback at all, because a freshly seeded
 * xorshift32 returns `seed * 6.3e-5` on its FIRST call: 300 small seeds probe
 * only u < 0.02, the extreme head of the distribution, and never reach the
 * tail where the window ends and the two implementations could part company.
 * (Measured: that sweep reported 0/300 fallbacks on a vector whose tail beyond
 * the window is 7.6%, i.e. it should have fallen back ~23 times.) A uniform
 * grid covers the head, the body and the region past the window in proportion.
 */
function agreementSweep(
  logits: Float32Array,
  options: SamplerOptions,
  draws: number,
  topK?: number,
): { disagreements: number[]; fallbacks: number } {
  const stats = createSamplerStats();
  const disagreements: number[] = [];
  for (let i = 0; i < draws; i += 1) {
    const u = (i + 0.5) / draws;
    const mine = sampleNextTopK(logits, [], fixedDraw(options, u), {
      ...(topK === undefined ? {} : { topK }),
      stats,
    });
    const theirs = sampleNext(logits, [], fixedDraw(options, u));
    if (mine !== theirs) disagreements.push(u);
  }
  return { disagreements, fallbacks: stats.fallbacks };
}

describe("xorshift32", () => {
  it("is deterministic and stays inside [0, 1)", () => {
    const a = xorshift32(42);
    const b = xorshift32(42);
    for (let i = 0; i < 100; i += 1) {
      const v = a();
      expect(v).toBe(b());
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
    expect(xorshift32(43)()).not.toBe(xorshift32(42)());
    // 0 is not a usable xorshift state; the seeded stream must still move.
    expect(xorshift32(0)()).toBeGreaterThan(0);
  });
});

describe("topKMass — what the truncation actually costs on this model", () => {
  /**
   * The numbers behind `DEFAULT_TOP_K`. `k=256` (the value the ISSUE guessed
   * at) leaves ~8% of the mass outside the window on the golden logits and up
   * to 45% on a real mid-utterance step: this model's next-token distribution
   * at temperature 0.8 is genuinely flat over a 164,480-token vocabulary, so
   * a truncation that DROPPED the tail would change the voice, not just its
   * timing. That is why the implementation falls back instead of truncating,
   * and why k is chosen for fallback RATE rather than for accuracy.
   */
  it("records the tail left outside the window at each k", () => {
    for (const { name, data } of goldenLogits) {
      const mass = topKMass(data, 0.8, [256, 1024, 2048, 4096]);
      // Recorded, not asserted away: printed so a run shows the real profile.
      console.log(`${name}: ` + mass.map((m, i) => `tail@${[256, 1024, 2048, 4096][i]}=${(1 - m).toExponential(2)}`).join(" "));
      expect(mass).toHaveLength(4);
      // Monotone: a wider window can only capture more.
      for (let i = 1; i < mass.length; i += 1) expect(mass[i]!).toBeGreaterThanOrEqual(mass[i - 1]!);
      expect(mass[mass.length - 1]!).toBeLessThanOrEqual(1);
    }
  });

  it("keeps the default window's fallback rate low, and shows why k=256 would not do", () => {
    // Two bounds, and the second is the load-bearing one.
    //
    // tail@2048 is the FALLBACK rate — how often a step pays for upstream's
    // sort — so it only has to be small. Measured: ja 6.3e-3, en 7.6e-2, and
    // on 60 real decode steps mean 4.1e-3 / worst 4.2e-2. The goldens are the
    // prefill's last position, i.e. the first generated token, where the model
    // is least constrained and the distribution is flattest of all; 1e-1 is a
    // ceiling over that worst case.
    //
    // tail@256 asserts the OPPOSITE direction on purpose: at the k the ISSUE
    // proposed, a plain truncate-and-renormalise would throw away more than 1%
    // of the mass (measured: ja 8.0e-2, en 3.4e-1). This is the assertion that
    // fails if someone "simplifies" the fallback away — the distribution is
    // genuinely long-tailed and dropping it is not free.
    for (const { name, data } of goldenLogits) {
      const [atDefault, at256] = topKMass(data, 0.8, [DEFAULT_TOP_K, 256]);
      expect(1 - atDefault!, `${name} tail beyond k=${DEFAULT_TOP_K}`).toBeLessThan(1e-1);
      expect(1 - at256!, `${name} tail beyond k=256`).toBeGreaterThan(1e-2);
    }
  });
});

describe("sampleNextTopK picks the same id as upstream sampleNext", () => {
  it("agrees over 200 draws spanning the whole range on every golden logits vector", () => {
    for (const { name, data } of goldenLogits) {
      const { disagreements, fallbacks } = agreementSweep(data, MIOTTS_SAMPLER, 200);
      console.log(`${name}: ${fallbacks}/200 draws took the exact fallback`);
      // Both regions have to be exercised or the sweep proves nothing: the
      // window for the fast path, past it for the fallback.
      expect(fallbacks, `${name} never left the window`).toBeGreaterThan(0);
      expect(fallbacks, `${name} never stayed inside the window`).toBeLessThan(200);
      expect(disagreements, `${name} disagreed at u = ${disagreements.slice(0, 8).join(",")}`).toEqual([]);
    }
  });

  it("stays in step with upstream over a shared, seeded rng stream", () => {
    // The other half of "same rng consumption": one generator per sampler,
    // seeded identically, driven for 150 CONSECUTIVE draws. If either ever
    // took a different number of rng values than the other — a second draw on
    // the fallback path, say — the streams would desynchronise and every id
    // after that point would differ. Independent per-draw comparisons cannot
    // see that at all.
    const stats = createSamplerStats();
    for (const { data } of goldenLogits) {
      const mineRng = xorshift32(42);
      const theirsRng = xorshift32(42);
      const mine: number[] = [];
      const theirs: number[] = [];
      for (let step = 0; step < 150; step += 1) {
        mine.push(sampleNextTopK(data, [], { ...MIOTTS_SAMPLER, rng: mineRng } as SamplerOptions, { stats }));
        theirs.push(sampleNext(data, [], { ...MIOTTS_SAMPLER, rng: theirsRng } as SamplerOptions));
      }
      expect(mine).toEqual(theirs);
    }
    // Counted across both vectors, not per vector: at ja's 0.6% tail a
    // 150-draw stream is as likely as not to stay inside the window the whole
    // way, and a per-vector assertion would be flaky for no extra coverage.
    expect(stats.fallbacks, "the stream never reached past the window").toBeGreaterThan(0);
  });

  it("agrees with a nucleus that fits inside the window (topP 0.9)", () => {
    // The other branch: topP < 1 finds its cutoff among the top k, so the
    // whole draw is served from the window with no fallback at all.
    for (const { data } of goldenLogits) {
      const { disagreements, fallbacks } = agreementSweep(
        data,
        { mode: "top-p", temperature: 0.8, topP: 0.9 },
        100,
      );
      expect(disagreements).toEqual([]);
      expect(fallbacks).toBe(0);
    }
  });

  it("agrees when the nucleus does NOT fit inside the window", () => {
    // topP 0.99 needs ~1500 tokens on these logits; with k=16 the cutoff is
    // not reachable from the window and every draw must fall back — still the
    // same id, because the fallback replays the SAME rng draw upstream.
    const { disagreements, fallbacks } = agreementSweep(
      goldenLogits[0]!.data,
      { mode: "top-p", temperature: 0.8, topP: 0.99 },
      100,
      16,
    );
    expect(disagreements).toEqual([]);
    expect(fallbacks).toBe(100);
  });

  it("agrees on a tie-heavy vector, where the ordering of equal logits decides", () => {
    // The selection has to break ties exactly the way upstream's stable sort
    // does — equal probability, lower index first. A heap that kept the LAST
    // of a tie would pass every test above (real logits rarely tie) and pick a
    // different id here.
    // 64 groups of 64 exactly-equal logits, one logit apart. With k=128 the
    // window holds groups 0 and 1 (~92% of the mass), so most draws are served
    // from the fast path and land on a tie.
    const n = 4096;
    const logits = new Float32Array(n);
    for (let i = 0; i < n; i += 1) logits[i] = 8 - Math.floor(i / 64);
    const { disagreements, fallbacks } = agreementSweep(logits, MIOTTS_SAMPLER, 300, 128);
    expect(disagreements).toEqual([]);
    expect(fallbacks, "the tie-heavy vector never exercised the fast path").toBeLessThan(150);
  });

  it("agrees when k exceeds the vocabulary", () => {
    const logits = Float32Array.from([0.1, 3, -2, 1.5, 0.9]);
    const { disagreements } = agreementSweep(logits, MIOTTS_SAMPLER, 200, 1024);
    expect(disagreements).toEqual([]);
  });
});

describe("greedy is untouched", () => {
  it("returns upstream's argmax on the golden logits", () => {
    for (const { data } of goldenLogits) {
      expect(sampleNextTopK(data, [], { mode: "greedy" })).toBe(sampleNext(data, [], { mode: "greedy" }));
    }
  });

  it("never counts a fallback, because greedy never reaches the top-k path", () => {
    const stats = createSamplerStats();
    sampleNextTopK(goldenLogits[0]!.data, [], { mode: "greedy" }, { stats });
    expect(stats).toEqual({ calls: 1, fallbacks: 0 });
  });
});

describe("constraints and errors behave exactly as upstream", () => {
  const only = (ids: number[]): Constraint => ({ nextAllowed: () => new Set(ids) });

  it("picks the same constrained id as upstream", () => {
    const allowed = [151669, 151670, 151700, 152000, 160000];
    for (const { data } of goldenLogits) {
      for (let i = 0; i < 30; i += 1) {
        const options = fixedDraw(MIOTTS_SAMPLER, (i + 0.5) / 30);
        expect(sampleNextTopK(data, [], options, { constraint: only(allowed) })).toBe(
          sampleNext(data, [], options, only(allowed)),
        );
      }
    }
  });

  it("rejects an empty constraint the way upstream does", () => {
    const empty: Constraint = { nextAllowed: () => new Set<number>() };
    expect(() => sampleNextTopK(goldenLogits[0]!.data, [], MIOTTS_SAMPLER, { constraint: empty })).toThrow(
      /constraint allows no token/,
    );
  });

  it("rejects the same bad parameters with the same messages", () => {
    const data = goldenLogits[0]!.data;
    for (const options of [
      { mode: "top-p", temperature: 0, topP: 1 },
      { mode: "top-p", temperature: -1, topP: 1 },
      { mode: "top-p", temperature: 0.8, topP: 0 },
      { mode: "top-p", temperature: 0.8, topP: 1.5 },
    ] as SamplerOptions[]) {
      let upstream = "";
      try {
        sampleNext(data, [], options);
      } catch (error) {
        upstream = (error as Error).message;
      }
      expect(upstream).not.toBe("");
      expect(() => sampleNextTopK(data, [], options)).toThrow(upstream);
    }
  });

  it("rejects an all-(-Infinity) vector the way upstream does", () => {
    const dead = Float32Array.from([-Infinity, -Infinity, -Infinity]);
    expect(() => sampleNextTopK(dead, [], MIOTTS_SAMPLER)).toThrow(/no finite logit/);
    expect(() => sampleNextTopK(dead, [], { mode: "greedy" })).toThrow(/no finite logit/);
  });

  it("rejects a non-positive topK rather than silently sampling from nothing", () => {
    expect(() => sampleNextTopK(goldenLogits[0]!.data, [], MIOTTS_SAMPLER, { topK: 0 })).toThrow(/topK/);
  });
});

describe("speed — the reason the wrapper exists", () => {
  /**
   * A floor, not a measurement: the wrapper is ~12-14x faster than upstream on
   * this vocabulary here, and the assertion is set at 4x so a slower machine
   * or a noisy CI box does not turn a real improvement into a red suite. What
   * it does catch is the regression that matters — a k so small that most
   * draws fall back, or a reintroduced full-vocabulary sort, both of which
   * land near 1x.
   *
   * **Each call continues ONE rng stream**, the way a real generation does,
   * rather than seeding a fresh generator per call. That is not a detail: a
   * freshly seeded xorshift32 returns ~1e-4 on its first call, so a per-call
   * seed keeps every draw at the head of the distribution and the fallback
   * never runs. Timed that way this test stayed green at 12x with
   * `DEFAULT_TOP_K` set to **2** — a sampler that must fall back on ~83% of
   * real draws and is therefore no faster than upstream at all. The stream is
   * what makes the amortised fallback cost show up in the number.
   */
  it("is at least 4x faster than upstream on a 164,480-token vocabulary", () => {
    const data = goldenLogits[0]!.data;
    const stats = createSamplerStats();
    const time = (run: () => void): number => {
      for (let i = 0; i < 5; i += 1) run(); // warm-up: let the JIT settle before the clock starts
      const started = performance.now();
      for (let i = 0; i < 30; i += 1) run();
      return (performance.now() - started) / 30;
    };
    const mineRng = xorshift32(7);
    const theirsRng = xorshift32(7);
    const mine = time(() =>
      sampleNextTopK(data, [], { ...MIOTTS_SAMPLER, rng: mineRng } as SamplerOptions, { stats }),
    );
    const theirs = time(() => sampleNext(data, [], { ...MIOTTS_SAMPLER, rng: theirsRng } as SamplerOptions));
    console.log(
      `sampleNextTopK ${mine.toFixed(2)} ms vs sampleNext ${theirs.toFixed(2)} ms ` +
        `(${(theirs / mine).toFixed(1)}x, ${(1000 / mine).toFixed(0)} vs ${(1000 / theirs).toFixed(0)} tok/s ceiling; ` +
        `${stats.fallbacks}/${stats.calls} fell back)`,
    );
    expect(theirs / mine).toBeGreaterThan(4);
  });
});
