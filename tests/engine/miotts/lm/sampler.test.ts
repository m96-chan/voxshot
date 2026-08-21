import { describe, expect, it } from "vitest";
import { sampleNext, type SamplerOptions } from "web-xpu-ops/llm/sampler";

import {
  DEFAULT_TOP_K,
  createSamplerStats,
  sampleNextTopK,
  topKMass,
  xorshift32,
} from "../../../../src/engine/miotts/lm/sampler.js";
import { pseudoRandom } from "../../../helpers/safetensors.js";

/**
 * `sampler.ts` against the upstream `sampleNext` it stands in front of.
 *
 * The wrapper's whole claim is that it is faster while picking the *same* id,
 * so that is what is checked: agreement over a sweep of draws, on synthetic
 * logits rather than the model's. The sweep matters more than any single
 * draw — a wrapper that quietly changed the distribution would still agree on
 * most draws and would still produce speech.
 *
 * Agreement on the model's own logits, and the measured speedup that justifies
 * the wrapper existing at all, need real weights and live in
 * `npm run test:models`.
 */

/** A vocabulary with a flat tail, like the model's — no one dominant token. */
function logitsOf(size: number, seed = 7): Float32Array {
  return pseudoRandom(size, seed, 4);
}

/** Ids from both implementations for the same draw, over the whole [0, 1) range. */
function sweep(
  logits: ArrayLike<number>,
  options: Omit<SamplerOptions & { mode: "top-p" }, "rng">,
  draws: number,
  tuning: Parameters<typeof sampleNextTopK>[3] = {},
): { fast: number[]; upstream: number[] } {
  const fast: number[] = [];
  const upstream: number[] = [];
  for (let i = 0; i < draws; i += 1) {
    const u = i / draws;
    fast.push(sampleNextTopK(logits, [], { ...options, rng: () => u }, tuning));
    upstream.push(sampleNext(logits, [], { ...options, rng: () => u }));
  }
  return { fast, upstream };
}

describe("sampleNextTopK agrees with upstream", () => {
  it("picks the same id across the whole draw range at topP = 1", () => {
    const { fast, upstream } = sweep(logitsOf(400), { mode: "top-p", temperature: 0.8, topP: 1 }, 200);

    expect(fast).toEqual(upstream);
  });

  it("picks the same id for a nucleus narrower than the vocabulary", () => {
    const { fast, upstream } = sweep(logitsOf(400, 11), { mode: "top-p", temperature: 0.8, topP: 0.9 }, 200);

    expect(fast).toEqual(upstream);
  });

  it("picks the same id when the temperature flattens the distribution", () => {
    const { fast, upstream } = sweep(logitsOf(300, 3), { mode: "top-p", temperature: 5, topP: 1 }, 150);

    expect(fast).toEqual(upstream);
  });

  it("still agrees once the window is too small to hold the draw", () => {
    // The window is a cache of the sort's prefix, not a truncation: a draw
    // landing past it delegates to upstream, replaying the value already
    // drawn. If it were a truncation, these ids would differ.
    const stats = createSamplerStats();
    const { fast, upstream } = sweep(
      logitsOf(400, 5),
      { mode: "top-p", temperature: 0.8, topP: 1 },
      200,
      { topK: 4, stats },
    );

    expect(fast).toEqual(upstream);
    expect(stats.fallbacks).toBeGreaterThan(0);
    expect(stats.calls).toBe(200);
  });

  it("consumes exactly one rng value per call, fallback or not", () => {
    // The sweep above cannot see this: it hands both implementations a
    // constant `() => u`, which is unaffected by how many times it is called.
    // Driving each from its own identically seeded STREAM is what makes an
    // extra draw observable — and it has to be, because a fallback that drew
    // again would desynchronise a seeded run from upstream at the first one,
    // and every id after it would differ.
    const logits = logitsOf(400, 5);
    const options = { mode: "top-p", temperature: 0.8, topP: 1 } as const;
    const stats = createSamplerStats();
    const fastRng = xorshift32(1234);
    const upstreamRng = xorshift32(1234);

    const fast: number[] = [];
    const upstream: number[] = [];
    for (let i = 0; i < 200; i += 1) {
      fast.push(sampleNextTopK(logits, [], { ...options, rng: fastRng }, { topK: 4, stats }));
      upstream.push(sampleNext(logits, [], { ...options, rng: upstreamRng }));
    }

    expect(fast).toEqual(upstream);
    expect(stats.fallbacks).toBeGreaterThan(0);
  });

  it("breaks probability ties by lower index, the way a stable sort does", () => {
    // Exact ties are rare in real logits and change the sampled id wherever
    // they occur, so the tie-break is stressed deliberately rather than left
    // to chance.
    //
    // Getting a tie to matter takes some care, and the shape below is the
    // reason. A vocabulary of uniformly equal logits observes nothing: the
    // heap admits a candidate only when it beats the root strictly, so with
    // every entry equal nothing is ever evicted and the tie-break is never
    // consulted. What consults it is an eviction while the root is tied — so
    // the window fills with equal entries, and then two better ones arrive and
    // displace the worst of them. Which two get displaced is exactly what the
    // index tie-break decides, and upstream's stable sort keeps the lowest.
    const tied = new Float32Array(64).fill(1);
    tied[60] = 5;
    tied[61] = 5;
    const { fast, upstream } = sweep(tied, { mode: "top-p", temperature: 1, topP: 1 }, 100, {
      topK: 8,
    });

    expect(fast).toEqual(upstream);
  });

  it("skips entries whose probability underflows, as upstream does", () => {
    const logits = logitsOf(200, 13);
    logits[17] = -1e30;
    const { fast, upstream } = sweep(logits, { mode: "top-p", temperature: 0.5, topP: 1 }, 120);

    expect(fast).toEqual(upstream);
    expect(fast).not.toContain(17);
  });
});

describe("sampleNextTopK delegation", () => {
  it("hands greedy straight to upstream and never counts a fallback", () => {
    // Greedy already costs a fraction of a millisecond, and it is the mode
    // that makes a GPU run bit-comparable to the CPU oracle.
    const logits = logitsOf(500, 2);
    const stats = createSamplerStats();

    const id = sampleNextTopK(logits, [], { mode: "greedy" }, { stats });

    expect(id).toBe(sampleNext(logits, [], { mode: "greedy" }));
    expect(stats.calls).toBe(1);
    expect(stats.fallbacks).toBe(0);
  });

  it("hands a constrained draw to upstream wholesale", () => {
    // Masking makes every probability depend on the constraint set, so the
    // fast path would be code on the run path that nothing exercises.
    const logits = logitsOf(64, 4);
    const allowed = new Set([3, 9, 21]);
    const constraint = { nextAllowed: (): ReadonlySet<number> => allowed };
    const options = { mode: "top-p", temperature: 0.8, topP: 1, rng: () => 0.42 } as const;

    const id = sampleNextTopK(logits, [], options, { constraint });

    expect(id).toBe(sampleNext(logits, [], options, constraint));
    expect(allowed.has(id)).toBe(true);
  });
});

describe("sampleNextTopK validation", () => {
  const logits = logitsOf(32);
  const good = { mode: "top-p", temperature: 0.8, topP: 1, rng: () => 0.5 } as const;

  it("rejects a window size that is not a positive integer", () => {
    expect(() => sampleNextTopK(logits, [], good, { topK: 0 })).toThrow(/positive integer/);
    expect(() => sampleNextTopK(logits, [], good, { topK: 2.5 })).toThrow(/positive integer/);
  });

  it("repeats upstream's own message for a bad temperature", () => {
    // Verbatim, so a caller sees one behaviour whichever implementation ran.
    expect(() => sampleNextTopK(logits, [], { ...good, temperature: 0 })).toThrow(
      /sampleTopP: temperature must be > 0/,
    );
  });

  it("repeats upstream's own message for a bad topP", () => {
    expect(() => sampleNextTopK(logits, [], { ...good, topP: 0 })).toThrow(/topP must be in/);
    expect(() => sampleNextTopK(logits, [], { ...good, topP: 1.5 })).toThrow(/topP must be in/);
  });

  it("refuses logits with nothing finite to choose from", () => {
    const empty = new Float32Array(8).fill(-Infinity);

    expect(() => sampleNextTopK(empty, [], good)).toThrow(/no finite logit/);
  });

  it("caps the window at the vocabulary size", () => {
    // A window larger than the vocabulary is not an error; there is simply
    // less to keep.
    expect(() => sampleNextTopK(logits, [], good, { topK: 10_000 })).not.toThrow();
  });
});

describe("xorshift32", () => {
  it("produces the same stream from the same seed", () => {
    const a = xorshift32(42);
    const b = xorshift32(42);

    expect([a(), a(), a()]).toEqual([b(), b(), b()]);
  });

  it("produces a different stream from a different seed", () => {
    expect(xorshift32(1)()).not.toBe(xorshift32(2)());
  });

  it("stays inside [0, 1)", () => {
    const rng = xorshift32(9);
    for (let i = 0; i < 500; i += 1) {
      const value = rng();
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });

  it("substitutes a non-zero state for seed 0", () => {
    // xorshift is a fixed point at zero: without the substitution, seed 0
    // returns 0 forever and a caller's "reproducible" run is a constant.
    const rng = xorshift32(0);
    const first = rng();

    expect(first).not.toBe(0);
    expect(rng()).not.toBe(first);
  });
});

describe("topKMass", () => {
  it("measures the mass captured by each window size", () => {
    // Four tokens, one clearly dominant: at k=1 the top token's share, at k=4
    // everything.
    const logits = Float32Array.from([10, 0, 0, 0]);
    const [atOne, atFour] = topKMass(logits, 1, [1, 4]);

    const weights = [Math.exp(0), Math.exp(-10), Math.exp(-10), Math.exp(-10)];
    const total = weights.reduce((a, b) => a + b, 0);
    expect(atOne).toBeCloseTo(weights[0]! / total, 12);
    expect(atFour).toBeCloseTo(1, 12);
  });

  it("reports 1 for a window past the end of the vocabulary", () => {
    expect(topKMass(Float32Array.from([1, 2, 3]), 1, [99])).toEqual([1]);
  });

  it("returns the masses in the order the windows were asked for", () => {
    // The windows are sorted internally to walk the order once; the answers
    // still have to line up with the caller's list.
    const logits = logitsOf(50, 6);
    const [big, small] = topKMass(logits, 0.8, [40, 2]);

    expect(big).toBeGreaterThan(small!);
  });

  it("is non-decreasing in the window size", () => {
    const masses = topKMass(logitsOf(120, 8), 0.8, [1, 4, 16, 64, 120]);

    for (let i = 1; i < masses.length; i += 1) {
      expect(masses[i]!).toBeGreaterThanOrEqual(masses[i - 1]!);
    }
    expect(masses.at(-1)).toBeCloseTo(1, 12);
  });
});

describe("DEFAULT_TOP_K", () => {
  it("is the measured window size, not a round guess", () => {
    // 2048 is the flat part of the fallback-rate curve recorded in the module
    // doc. Pinned so a change has to come with a new measurement.
    expect(DEFAULT_TOP_K).toBe(2048);
  });
});
