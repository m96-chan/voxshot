/**
 * `normalize_text` from the MioTTS reference server (`miotts_server/text.py`),
 * ported rule for rule — including its quirks, because the LM was served
 * behind exactly this normalization:
 *
 *   - the `" " -> ""` rule deletes EVERY ASCII space, not just doubles;
 *   - bracket stripping is startswith/endswith only, so an unmatched inner
 *     bracket does not stop it, and the five checks run sequentially (nested
 *     pairs strip in one call);
 *   - the trailing strip removes any run of `。`/`、`, but only when the text
 *     ends in one of them (which is also when a run exists — mirrored anyway);
 *   - the half-width katakana fold is an explicit 56-char table; dakuten
 *     marks (ﾞﾟ) are not in it and pass through, as in the reference.
 *
 * Pure function, no platform dependencies — testable against vectors captured
 * from the Python implementation (see text.test.ts).
 */

/** `REPLACE_MAP`, in the reference's insertion order (order is observable). */
const REPLACE: [RegExp, string][] = [
  [/\t/g, ""],
  [/\[n\]/g, ""],
  [/ /g, ""],
  [/　/g, ""], // 　 full-width space
  [/[;▼♀♂《》≪≫①②③④⑤⑥]/g, ""],
  // The reference's dash/bar class, escape for escape (˗ ‐‑‒–—― ⁃ − ⎯ ⏤ ─ ━ ⸺ ⸻).
  [/[˗‐-―⁃−⎯⏤─━⸺⸻]/g, ""],
  [/[～〜]/g, "ー"], // ～ and 〜 both become the long-vowel bar
  [/？/g, "?"],
  [/！/g, "!"],
  [/[●◯〇]/g, "○"],
  [/♥/g, "♡"],
];

// ＡＢＣ… (U+FF21..FF3A, U+FF41..FF5A) and １２３… (U+FF10..FF19) sit exactly
// 0xFEE0 above their ASCII counterparts — one offset covers both tables.
const FULLWIDTH_ALNUM = /[Ａ-Ｚａ-ｚ０-９]/g;

const HALFWIDTH_KATAKANA = "ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ";
const FULLWIDTH_KATAKANA = "ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン";

const KATAKANA_FOLD = new Map<string, string>();
for (let i = 0; i < HALFWIDTH_KATAKANA.length; i += 1) {
  KATAKANA_FOLD.set(HALFWIDTH_KATAKANA[i]!, FULLWIDTH_KATAKANA[i]!);
}

const BRACKET_PAIRS: [string, string][] = [
  ["「", "」"],
  ["『", "』"],
  ["（", "）"],
  ["【", "】"],
  ["(", ")"],
];

export function normalizeText(text: string): string {
  for (const [pattern, replacement] of REPLACE) {
    text = text.replace(pattern, replacement);
  }

  text = text.replace(FULLWIDTH_ALNUM, (ch) => String.fromCharCode(ch.charCodeAt(0) - 0xfee0));
  // `[ｦ-ﾝ]` (U+FF66..U+FF9D) is exactly the 56 characters KATAKANA_FOLD holds,
  // so the lookup never misses and there is no fallback branch to test. The
  // dakuten marks ﾞﾟ (U+FF9E/U+FF9F) sit past the class and pass through
  // untouched, which is what the reference does too.
  text = text.replace(/[ｦ-ﾝ]/g, (ch) => KATAKANA_FOLD.get(ch)!);

  text = text.replace(/…{3,}/g, "……");

  for (const [open, close] of BRACKET_PAIRS) {
    if (text.startsWith(open) && text.endsWith(close)) {
      text = text.slice(1, -1);
    }
  }

  if (text.endsWith("。") || text.endsWith("、")) {
    text = text.replace(/[。、]+$/, "");
  }

  return text;
}
