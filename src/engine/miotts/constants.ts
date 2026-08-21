/**
 * Generation limits shared by every runner of the MioTTS LM.
 *
 * browser.ts (the GPU page) and expected-tokens.ts (the CPU oracle) must cap
 * generation identically — the check script diffs their token streams id for
 * id, and a cap applied on one side only would surface as a fake divergence
 * on long prompts.
 */

/** The MioTTS reference server's own max_tokens. */
export const MAX_NEW_TOKENS = 700;

/** Engine sequence budget: prompt (~15) + 700 generated, with slack. */
export const MAX_SEQ_LEN = 768;

/**
 * The cap both runners apply: never generate past the sequence budget.
 *
 * `maxSeqLen` is a parameter because the engine's is a caller's choice — the
 * KV cache is sized from it, and an application sharing a GPU sets it low.
 * Defaulting to {@link MAX_SEQ_LEN} would clamp against a budget the engine
 * was not built with, and generation would run off the end of its own cache.
 */
export function maxNewFor(
  promptLen: number,
  maxNew: number = MAX_NEW_TOKENS,
  maxSeqLen: number = MAX_SEQ_LEN,
): number {
  return Math.min(maxNew, maxSeqLen - promptLen);
}
