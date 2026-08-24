/**
 * `normalize_text` from Irodori's `irodori_tts/text_normalization.py`, ported
 * rule for rule.
 *
 * Worth reading beside `src/engine/miotts/text.ts`, which is the same idea for a
 * different model and differs in every way that matters:
 *
 *   - **spaces survive here.** MioTTS deletes every ASCII space; this does not
 *     touch them, so "hello world" stays two words and tokenizes as two.
 *   - **bracket stripping is depth-aware.** MioTTS checks only the first and
 *     last character, so `「はじまり」おわり」` loses one from each end. This walks
 *     the string and strips only when the opening bracket's own depth returns
 *     to zero exactly at the end — so that string is left alone. It also loops,
 *     which is how `「(にじゅう)」` loses both pairs.
 *   - **NFKC does the width folding**, at the end, rather than the explicit
 *     56-character katakana table MioTTS carries.
 *   - `...` and `..` become `…`, *after* NFKC, so a full-width `．．．` folds
 *     first and collapses too.
 *
 * Two models by the same author, and none of the normalisation is shared. That
 * is a reason to port each one from its own source rather than reuse.
 */

/** `SIMPLE_REPLACE_MAP`, in the reference's insertion order (order is observable). */
const SIMPLE: [string, string][] = [
  ["\t", ""],
  ["[n]", ""],
  // The reference has `r"\[n\]"` as a second literal key. It is a regex string
  // used as a plain `str.replace` argument, so it removes the literal text
  // `\[n\]` — backslashes and all. Kept because it is what the reference does,
  // not because any input has it.
  ["\\[n\\]", ""],
  ["　", ""],
  ["？", "?"],
  ["！", "!"],
  ["♥", "♡"],
  ["●", "○"],
  ["◯", "○"],
  ["〇", "○"],
];

const REGEX: [RegExp, string][] = [
  [/[;▼♀♂《》≪≫①②③④⑤⑥]/g, ""],
  [/[˗‐-―⁃−⎯⏤─━⸺⸻]/g, ""],
  [/[～〜]/g, "ー"],
  [/…{3,}/g, "……"],
];

const BRACKET_PAIRS: Record<string, string> = {
  "「": "」",
  "『": "』",
  "（": "）",
  "【": "】",
  "(": ")",
};

/**
 * Strip a bracket pair only when it encloses the whole string, repeatedly.
 *
 * The depth walk is the point. A naive first/last check would strip
 * `「はじまり」おわり」` — the opening bracket closes at index 5, long before the
 * end, so its pair does not enclose everything and nothing is removed.
 */
export function stripOuterBrackets(text: string): string {
  for (;;) {
    if (text.length < 2) break;
    const open = text[0]!;
    const close = text[text.length - 1]!;
    if (BRACKET_PAIRS[open] !== close) break;

    let depth = 0;
    let enclosesAll = true;
    for (let i = 0; i < text.length; i += 1) {
      const char = text[i]!;
      if (char === open) depth += 1;
      else if (char === close) depth -= 1;
      if (depth === 0 && i < text.length - 1) {
        enclosesAll = false;
        break;
      }
    }
    if (!(enclosesAll && depth === 0)) break;
    text = text.slice(1, -1);
  }
  return text;
}

export function normalizeText(text: string): string {
  for (const [from, to] of SIMPLE) text = text.split(from).join(to);
  for (const [pattern, replacement] of REGEX) text = text.replace(pattern, replacement);

  text = stripOuterBrackets(text);
  text = text.normalize("NFKC");

  // Order matters and is the reference's: three dots first, then two, so
  // "...." becomes "…" + "…" rather than "…" + ".."
  text = text.split("...").join("…");
  text = text.split("..").join("…");

  return text;
}
