import { sampleNext, type Constraint, type SamplerOptions } from "../../../web-xpu-ops/llm/sampler.js";

/**
 * A fast front end for web-xpu-ops' `llm/sampler.ts#sampleNext`, for the one
 * thing this spike does every decode step: draw one id out of 164,480 logits.
 *
 * ## Why (ISSUE #120)
 *
 * Greedy generation runs at ~209 tok/s on this machine's GPU; the same run in
 * sampled mode managed ~25. The GPU was not the problem. Measured here, on the
 * golden logits, one `sampleNext` call in `top-p` mode costs **38 ms** — a
 * ceiling of 26 tok/s before a single matmul happens — against 0.33 ms for
 * greedy. Upstream's top-p path allocates two 164,480-element Float64Arrays,
 * builds a 164,480-element `number[]` of indices, and **sorts it**, every
 * step. The sort is ~90% of the cost, and it is thrown away after the first
 * few hundred entries are read.
 *
 * ## What this does instead
 *
 * The draw only ever walks the descending-probability order until the
 * cumulative mass passes it, so only a prefix of that order is ever needed.
 * This module finds the top `topK` entries with a bounded min-heap — O(V·log k)
 * with no allocation proportional to V — and serves the draw from that window.
 *
 * ## It does not change the distribution. At all.
 *
 * The obvious version of this idea (truncate to the top k, renormalise, sample)
 * WOULD change it, and on this model badly. Measured over 60 consecutive real
 * decode steps of a sampled generation at temperature 0.8:
 *
 *     k     tail mass left outside the window (mean / median / worst)
 *     64    3.6e-1 / 3.7e-1 / 7.4e-1
 *     256   1.2e-1 / 8.8e-2 / 4.5e-1     <- the value ISSUE #120 guessed at
 *     1024  1.7e-2 / 7.1e-3 / 1.3e-1
 *     2048  4.1e-3 / 1.8e-3 / 4.2e-2
 *     4096  5.9e-4 / 3.4e-4 / 5.3e-3
 *     8192  4.4e-5 / 2.9e-5 / 2.1e-4
 *
 * MioTTS' next-token distribution over a 164,480-token vocabulary is flat:
 * the top token averaged 5% of the mass, and reaching 99.9% took a median of
 * ~2,900 tokens. Dropping the tail at any k cheap enough to be worth having
 * would be dropping real speech tokens, i.e. changing the voice.
 *
 * So the window is not a truncation — it is a **cache of the sort's prefix**.
 * When a draw lands beyond it (`topP` = 1 leaves the nucleus wider than any
 * window), this delegates to upstream's own `sampleNext`, replaying the SAME
 * rng value it already drew. One `rng()` call per invocation, exactly as
 * upstream, so the two consume a shared stream identically and a seeded run is
 * reproducible across the two implementations. `sampler.test.ts` pins that as
 * id-for-id agreement over hundreds of draws spanning the whole range; a
 * wider sweep over 60 real decode steps (14,400 draws) found 0 disagreements
 * and fell back on 0.39% of them.
 *
 * The one place the arithmetic is not bit-identical is `topP >= 1`, where
 * upstream normalises by `nucleusTotal` (the descending-order sum over the
 * nucleus) and this normalises by `total` (the index-order sum over the whole
 * vocabulary). Those are the same sum in different orders, so they differ only
 * by f64 accumulation rounding — ~1e-13 relative over 164,480 terms — and the
 * ids differ only if a draw lands inside that window of a bucket boundary. The
 * agreement sweep is what says it does not happen, not this comment.
 *
 * Greedy is delegated to upstream untouched: it is already 0.33 ms, and
 * `check-tts.mjs` asserts the GPU's greedy ids equal the CPU oracle's exactly.
 */

/**
 * The window size, chosen for FALLBACK RATE rather than for accuracy — the
 * fallback is exact, so k trades speed against how often a step pays for
 * upstream's sort.
 *
 * From the table above, at k=2048 the mean tail is 4.1e-3: roughly 1 step in
 * 240 falls back, costing 38 ms, so the amortised penalty is ~0.16 ms/step on
 * top of a ~3 ms fast path. k=1024 would make it ~0.65 ms and k=4096 ~0.02 ms
 * while making the heap work harder on every step; 2048 is the flat part of
 * that curve.
 */
export const DEFAULT_TOP_K = 2048;

/** How the sampler spent its steps, for a caller that wants to report it. */
export interface SamplerStats {
  /** Every `sampleNextTopK` call, greedy included. */
  calls: number;
  /** Calls whose draw fell outside the window and paid for upstream's full sort. */
  fallbacks: number;
}

export function createSamplerStats(): SamplerStats {
  return { calls: 0, fallbacks: 0 };
}

/** Deterministic xorshift32 in [0, 1) so sampled runs are reproducible from a seed. */
export function xorshift32(seed: number): () => number {
  let s = seed >>> 0 || 0x9e3779b9;
  return () => {
    s ^= s << 13;
    s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 4294967296;
  };
}

export interface TopKTuning {
  /** Window size. Defaults to `DEFAULT_TOP_K`. */
  topK?: number;
  /** Passed straight through to upstream — see the note in `sampleNextTopK`. */
  constraint?: Constraint;
  /** Incremented in place if given. */
  stats?: SamplerStats;
}

/**
 * The share of the temperature-scaled probability mass captured by the top
 * `k` tokens, for each `k` in `ks`.
 *
 * This is the measurement `DEFAULT_TOP_K` was chosen from, kept as code so the
 * number in the doc comment above can be re-derived rather than believed. It
 * does the full sort deliberately: it is a measuring instrument, not part of
 * the run path.
 */
export function topKMass(logits: ArrayLike<number>, temperature: number, ks: readonly number[]): number[] {
  const n = logits.length;
  let maxScaled = -Infinity;
  for (let i = 0; i < n; i += 1) {
    const v = logits[i]! / temperature;
    if (v > maxScaled) maxScaled = v;
  }
  const probs = new Float64Array(n);
  let total = 0;
  for (let i = 0; i < n; i += 1) {
    const p = Math.exp(logits[i]! / temperature - maxScaled);
    probs[i] = p;
    total += p;
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => probs[b]! - probs[a]!);
  const wanted = ks.map((k, index) => ({ k, index })).sort((a, b) => a.k - b.k);
  // A k past the vocabulary captures everything; the loop below overwrites the
  // ones it reaches.
  const out = new Array<number>(ks.length).fill(1);
  let cumulative = 0;
  let next = 0;
  for (let r = 0; r < order.length && next < wanted.length; r += 1) {
    cumulative += probs[order[r]!]!;
    while (next < wanted.length && r + 1 === wanted[next]!.k) {
      out[wanted[next]!.index] = cumulative / total;
      next += 1;
    }
  }
  return out;
}

/* -------------------------------------------------------------------------- *
 * The bounded selection
 * -------------------------------------------------------------------------- */

/**
 * A min-heap of the best `k` (probability, index) pairs seen so far, where
 * "best" is **higher probability, then lower index** — the exact order
 * upstream's `sort((a, b) => probs[b] - probs[a])` produces, because V8's sort
 * is stable and a comparator returning 0 leaves ties in ascending index order.
 *
 * Getting that tie-break wrong is invisible on real logits (exact ties are
 * rare) and changes the sampled id on any vector that has them, which is why
 * `sampler.test.ts` stresses it with a deliberately tie-heavy vector.
 *
 * The heap's root is the WORST of the kept entries, so a candidate is admitted
 * with one comparison against it in the overwhelmingly common reject case.
 */
class TopK {
  private readonly probs: Float64Array;
  private readonly indices: Int32Array;
  private size = 0;

  constructor(private readonly capacity: number) {
    this.probs = new Float64Array(capacity);
    this.indices = new Int32Array(capacity);
  }

  /** Scanned in ascending `index`, so an equal probability arriving later is always worse. */
  offer(index: number, prob: number): void {
    if (this.size < this.capacity) {
      this.probs[this.size] = prob;
      this.indices[this.size] = index;
      this.size += 1;
      this.siftUp(this.size - 1);
      return;
    }
    if (prob <= this.probs[0]!) return;
    this.probs[0] = prob;
    this.indices[0] = index;
    this.siftDown(0);
  }

  /** The kept entries in descending order: probability first, index as the tie-break. */
  drain(): { indices: Int32Array; probs: Float64Array } {
    const order = Array.from({ length: this.size }, (_, i) => i).sort((a, b) => {
      const d = this.probs[b]! - this.probs[a]!;
      return d !== 0 ? d : this.indices[a]! - this.indices[b]!;
    });
    const indices = new Int32Array(this.size);
    const probs = new Float64Array(this.size);
    for (let r = 0; r < order.length; r += 1) {
      indices[r] = this.indices[order[r]!]!;
      probs[r] = this.probs[order[r]!]!;
    }
    return { indices, probs };
  }

  /** True when `a` should be closer to the root, i.e. is the worse entry. */
  private worse(a: number, b: number): boolean {
    const pa = this.probs[a]!;
    const pb = this.probs[b]!;
    return pa !== pb ? pa < pb : this.indices[a]! > this.indices[b]!;
  }

  private swap(a: number, b: number): void {
    const p = this.probs[a]!;
    this.probs[a] = this.probs[b]!;
    this.probs[b] = p;
    const i = this.indices[a]!;
    this.indices[a] = this.indices[b]!;
    this.indices[b] = i;
  }

  private siftUp(start: number): void {
    let node = start;
    while (node > 0) {
      const parent = (node - 1) >> 1;
      if (!this.worse(node, parent)) break;
      this.swap(node, parent);
      node = parent;
    }
  }

  private siftDown(start: number): void {
    let node = start;
    for (;;) {
      const left = node * 2 + 1;
      if (left >= this.size) break;
      const right = left + 1;
      let worst = this.worse(left, node) ? left : node;
      if (right < this.size && this.worse(right, worst)) worst = right;
      if (worst === node) break;
      this.swap(node, worst);
      node = worst;
    }
  }
}

/* -------------------------------------------------------------------------- *
 * The sampler
 * -------------------------------------------------------------------------- */

/**
 * Picks the next token id — the same id `sampleNext` would pick from the same
 * logits and the same rng, in a fraction of the time.
 *
 * A `constraint` is delegated wholesale to upstream: masking makes the whole
 * vocabulary's probabilities depend on the constraint set, and nothing in this
 * spike passes one, so the fast path would be untested code on the run path.
 * Upstream consumes exactly one `rng()` call either way.
 */
export function sampleNextTopK(
  logits: ArrayLike<number>,
  prefixTokens: readonly number[],
  options: SamplerOptions,
  tuning: TopKTuning = {},
): number {
  const stats = tuning.stats;
  if (stats) stats.calls += 1;

  if (options.mode === "greedy" || tuning.constraint) {
    return sampleNext(logits, prefixTokens, options, tuning.constraint);
  }

  const topK = tuning.topK ?? DEFAULT_TOP_K;
  if (!Number.isInteger(topK) || topK < 1) {
    throw new Error(`sampleNextTopK: topK must be a positive integer, got ${topK}`);
  }
  // Upstream's validation, verbatim, so a caller sees one behaviour whichever
  // of the two ends up running. Its messages are asserted in the tests.
  const { temperature, topP } = options;
  if (!(temperature > 0)) {
    throw new Error(`sampleTopP: temperature must be > 0, got ${temperature}`);
  }
  if (!(topP > 0) || topP > 1) {
    throw new Error(`sampleTopP: topP must be in (0, 1], got ${topP}`);
  }

  const n = logits.length;
  // max(fl(l/T)) === fl(max(l)/T): correctly-rounded division is monotone, so
  // the maximum of the scaled logits is the scaled maximum. One pass, no array.
  let maxLogit = -Infinity;
  for (let i = 0; i < n; i += 1) {
    const v = logits[i]!;
    if (v > maxLogit) maxLogit = v;
  }
  const maxScaled = maxLogit / temperature;
  if (maxScaled === -Infinity) {
    throw new Error("sampleTopP: no finite logit to choose from");
  }

  // One pass for both the normaliser and the window. `total` accumulates in
  // index order over exactly the same f64 terms upstream sums, so it is the
  // same value bit for bit. Entries whose probability underflows to 0 are
  // skipped, matching upstream's `.filter((i) => probs[i] > 0)`.
  const heap = new TopK(Math.min(topK, n));
  let total = 0;
  for (let i = 0; i < n; i += 1) {
    const p = Math.exp(logits[i]! / temperature - maxScaled);
    total += p;
    if (p > 0) heap.offer(i, p);
  }
  const { indices, probs } = heap.drain();

  // Upstream's nucleus cutoff, run over the window. If it lands inside, the
  // window holds the entire nucleus and everything below is bit-identical to
  // upstream — same terms, same order, same sums.
  let cumulative = 0;
  let cutoff = -1;
  for (let r = 0; r < probs.length; r += 1) {
    cumulative += probs[r]!;
    if (cumulative / total >= topP) {
      cutoff = r + 1;
      break;
    }
  }

  const rng = options.rng ?? Math.random;
  // Drawn HERE, before any decision to fall back, and passed down if we do:
  // one rng() per call, whichever path runs. Anything else and a seeded run
  // would diverge from upstream after the first fallback.
  const u = rng();

  if (cutoff >= 0) {
    let nucleusTotal = 0;
    for (let r = 0; r < cutoff; r += 1) nucleusTotal += probs[r]!;
    const draw = u * nucleusTotal;
    let acc = 0;
    for (let r = 0; r < cutoff; r += 1) {
      acc += probs[r]!;
      if (draw < acc) return indices[r]!;
    }
    // Upstream's floating-point edge: the draw landed on (or an epsilon past)
    // the total, and the last nucleus member is the answer either way.
    return indices[cutoff - 1]!;
  }

  // The nucleus is wider than the window. For topP >= 1 it is the whole
  // vocabulary, whose mass is `total` up to summation order (see the module
  // doc), so the draw can still be served from the window whenever it lands
  // inside it — which is the common case, the window being ~99.6% of the mass.
  if (topP >= 1) {
    const draw = u * total;
    let acc = 0;
    for (let r = 0; r < probs.length; r += 1) {
      acc += probs[r]!;
      if (draw < acc) return indices[r]!;
    }
  }

  // Beyond the window (or a topP nucleus that does not fit in it): upstream,
  // replaying the value already drawn.
  if (stats) stats.fallbacks += 1;
  return sampleNext(logits, prefixTokens, { ...options, rng: () => u }, tuning.constraint);
}
