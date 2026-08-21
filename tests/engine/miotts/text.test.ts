import { describe, expect, it } from "vitest";
import { normalizeText } from "../../../src/engine/miotts/text.js";

/**
 * Vectors for the `normalize_text` port.
 *
 * Every {input, expected} pair below was produced by RUNNING the reference
 * implementation (MioTTS-Inference `miotts_server/text.py`, `normalize_text`)
 * via `python3 -c` on 2026-08-21 — not hand-derived. The port is measured
 * against the reference's actual behaviour, quirks included:
 *
 *   - `"「はじまり」おわり」"` strips one char off each end because Python only
 *     checks startswith/endswith, never that the pair actually matches.
 *   - `"「Ｈｅｌｌｏ　Ｗｏｒｌｄ！」。"` keeps its brackets: the bracket check
 *     runs while the text still ends in `。`, and only the `。` is stripped.
 *   - Half-width katakana folds through an explicit 56-char table (dakuten
 *     marks are NOT in it and would pass through — mirroring, not fixing).
 */
const VECTORS: { input: string; expected: string }[] = [
  // \t removal
  { input: "a\tb\tc", expected: "abc" },
  // literal [n] removal
  { input: "せりふ[n]つづき[n]", expected: "せりふつづき" },
  // ASCII space stripping (the r" " -> "" rule removes every space)
  { input: "hello world how are you", expected: "helloworldhowareyou" },
  // full-width space stripping
  { input: "こんにちは　世界　テスト", expected: "こんにちは世界テスト" },
  // symbol class ;▼♀♂《》≪≫①-⑥
  { input: "それは;▼♀♂《かっこ》≪にじゅう≫①②③④⑤⑥です", expected: "それはかっこにじゅうです" },
  // the dash/bar class ˗ ‐-― ⁃ − ⎯ ⏤ ─ ━ ⸺ ⸻
  { input: "ダッシュ˗‐‑‒–—―⁃−⎯⏤─━⸺⸻おわり", expected: "ダッシュおわり" },
  // ～ (U+FF5E) and 〜 (U+301C) both fold to ー
  { input: "そう～だね〜はい", expected: "そうーだねーはい" },
  // ？ -> ? and ！ -> !
  { input: "本当？すごい！ね？！", expected: "本当?すごい!ね?!" },
  // ●◯〇 -> ○ and ♥ -> ♡
  { input: "記号は●と◯と〇と♥です", expected: "記号は○と○と○と♡です" },
  // full-width alphabet -> half-width
  { input: "ＡＢＣＸＹＺａｂｃｘｙｚ", expected: "ABCXYZabcxyz" },
  // full-width digits -> half-width
  { input: "０１２３４５６７８９", expected: "0123456789" },
  // the whole half-width katakana table, in table order
  {
    input: "ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ",
    expected: "ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン",
  },
  // …{3,} collapses to ……; one and two survive
  { input: "え…だから……そして………ほら…………ね", expected: "え…だから……そして……ほら……ね" },
  // each bracket pair strips when it wraps the whole string
  { input: "「こんにちは」", expected: "こんにちは" },
  { input: "『ほん』", expected: "ほん" },
  { input: "（ちゅう）", expected: "ちゅう" },
  { input: "【だい】", expected: "だい" },
  { input: "(x)", expected: "x" },
  // the ifs run sequentially, so nested pairs strip in one call
  { input: "「（にじゅうかっこ）」", expected: "にじゅうかっこ" },
  // trailing 。/、 rstrip
  { input: "こんにちは。", expected: "こんにちは" },
  { input: "それで、、。。", expected: "それで" },
  // startswith/endswith only — an unmatched inner 」 does not stop the strip
  { input: "「はじまり」おわり」", expected: "はじまり」おわり" },
  // bracket check happens while the 。 is still there, so brackets survive
  { input: "「Ｈｅｌｌｏ　Ｗｏｒｌｄ！」。", expected: "「HelloWorld!」" },
  // the golden ja case is already normal — must pass through untouched
  { input: "こんにちは、今日はいい天気ですね", expected: "こんにちは、今日はいい天気ですね" },
  { input: "", expected: "" },
];

describe("normalizeText against the reference implementation's output", () => {
  for (const { input, expected } of VECTORS) {
    it(`normalizes ${JSON.stringify(input.length > 30 ? input.slice(0, 30) + "…" : input)}`, () => {
      expect(normalizeText(input)).toBe(expected);
    });
  }
});
