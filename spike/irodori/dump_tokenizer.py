"""Dump normalize_text + tokenizer vectors from the reference.

    cd spike/irodori && IRODORI_REPO=... ../dacvae/.venv/bin/python dump_tokenizer.py

Writes `golden/tokenizer_vectors.json`. Small enough to be cheap, but still not
in git — it is derived from a checkpoint, and regenerating it is one command.

## Why this is its own golden

The text path is the one part of Irodori that is not a tensor, so a
stage-by-stage tensor comparison cannot see it. It is also the part most likely
to be wrong in a way that still produces audio: a tokenizer that segments
differently gives the model different input and the model says something
slightly different, fluently.

## The two halves, and why they are separated

`normalize_text` is Irodori's own, and its rules differ from MioTTS's in ways
worth pinning rather than assuming:

  - it does **not** delete ASCII spaces (MioTTS deletes every one)
  - bracket stripping is depth-aware — it checks the pair actually encloses the
    whole string, where MioTTS only checks the first and last character
  - NFKC runs at the end, rather than explicit width tables
  - `...` and `..` collapse to `…` after that

The tokenizer is **Unigram with byte fallback**, not BPE: it scores every
segmentation the vocabulary admits and takes the best, rather than merging
greedily by rank. Metaspace maps spaces to U+2581 with `prepend_scheme: never`,
and `<s>` / `</s>` wrap the result.

Vectors cover both halves separately (`normalized`, then `ids`), so a
disagreement says which one is wrong.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
OUT = HERE / "golden"
REPO = "sbintuitions/modernbert-ja-310m"
REVISION = "77675fc96a7e445e982e2ba90246b816efc74ec6"

CASES = [
    # the lines this spike renders
    "……ん。来たんだ。",
    "全部追うの、無理だよ？",
    "こんにちは、今日はいい天気ですね。散歩でもしませんか。",
    "えっと、それはちょっと、わたしには難しいかもしれません。",
    # spaces survive here, unlike MioTTS
    "hello world how are you",
    "日本語 と English の 混在",
    # the replace map
    "a\tb\tc",
    "せりふ[n]つづき[n]",
    "全角　空白",
    "本当？すごい！ね？！",
    "記号は●と◯と〇と♥です",
    "ダッシュ˗‐‑‒–—―⁃−⎯⏤─━⸺⸻おわり",
    "そう～だね〜はい",
    "え…だから……そして………ほら…………ね",
    "dots... and.. more",
    # depth-aware bracket stripping — the interesting half
    "「こんにちは」",
    "「はじまり」おわり」",
    "「(にじゅう)」",
    "（片方だけ",
    "(a)(b)",
    # NFKC
    "ＡＢＣａｂｃ０１２",
    "ｦｧｨｩｪｫｬｭｮｯｰｱｲｳ",
    # byte fallback and rare characters
    "🙂 絵文字",
    "𠮷野家",
    "ẞ",
    "",
    " ",
]


def main() -> None:
    reference = Path(os.environ.get("IRODORI_REPO", "")).expanduser()
    if not reference.is_dir():
        raise SystemExit("Set IRODORI_REPO to a clone of Aratako/Irodori-TTS")
    sys.path.insert(0, str(reference))

    from transformers import AutoTokenizer  # noqa: E402

    from irodori_tts.text_normalization import normalize_text  # noqa: E402

    tokenizer = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
    OUT.mkdir(exist_ok=True)

    vectors = []
    for text in CASES:
        normalized = normalize_text(text).strip()
        ids = tokenizer(normalized)["input_ids"]
        vectors.append({"text": text, "normalized": normalized, "ids": ids})
        print(f"  {text[:26]!r:32s} -> {normalized[:24]!r:28s} {len(ids)} ids")

    payload = {
        "_rebuild": "cd spike/irodori && IRODORI_REPO=... ../dacvae/.venv/bin/python dump_tokenizer.py",
        "repo": REPO,
        "revision": REVISION,
        "bos_id": tokenizer.bos_token_id,
        "eos_id": tokenizer.eos_token_id,
        "unk_id": tokenizer.unk_token_id,
        "vocab_size": tokenizer.vocab_size,
        "vectors": vectors,
    }
    (OUT / "tokenizer_vectors.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"\nwrote {OUT}/tokenizer_vectors.json — {len(vectors)} vectors")


if __name__ == "__main__":
    main()
