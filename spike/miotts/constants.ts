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

/** The cap both runners apply: never generate past the sequence budget. */
export function maxNewFor(promptLen: number, maxNew: number = MAX_NEW_TOKENS): number {
  return Math.min(maxNew, MAX_SEQ_LEN - promptLen);
}
