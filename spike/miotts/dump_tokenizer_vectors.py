"""Dump tokenizer golden vectors from the reference HF tokenizer.

The output is **not in git** — golden/ is ignored and reproducible from here.
Regenerate with:

    python3 dump_tokenizer_vectors.py

Why vectors and not a re-derivation in the test: the TypeScript port re-implements
NFC + added-token splitting + the Qwen2 pre-tokenizer regex + byte-level BPE from
scratch, and every one of those stages has a way to be subtly wrong that still
produces plausible ids (a case-insensitive contraction group rewritten wrong, a
merge loop that ties differently, a byte map off by one). The only trustworthy
oracle is the reference tokenizer itself, run on inputs chosen to hit each stage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

SNAPSHOT = (
    "/home/m96-chan/.cache/huggingface/hub/models--Aratako--MioTTS-0.6B/"
    "snapshots/901ee12c50efd68ea3db6a6780cd29da197cd0da"
)
OUT = Path(__file__).parent / "golden" / "tokenizer_vectors.json"

# Each string exercises a specific stage of the pipeline; see the task list in
# the README of the port. Diversity is the point — a vector set of plain ASCII
# would let a broken \p{L} rewrite or a wrong byte map go unnoticed.
ENCODE_TEXTS = [
    # JA plain
    "こんにちは",
    "音声合成のテストです",
    # JA with punctuation and the long-vowel mark
    "こんにちは、世界。",
    "え、本当に?すごい!",
    "コーヒーをください。",
    # EN with spaces and contractions (case variants catch a wrong (?i:) rewrite)
    "Hello world",
    "I've been here before, and I don't think it's changed.",
    "I'VE SAID I DON'T KNOW, IT'S FINE.",
    "She'll say they're happy we'd met.",
    # contraction letters with a word continuing right after — the ONLY inputs
    # that distinguish the contraction group from the [^..]?\p{L}+ fallback
    # (for a plain "'VE" both branches produce the same split; for "'VEGOT"
    # the contraction branch stops at "'VE" and the fallback would not)
    "we'rex and I'VEGOT it, don'tx",
    "SHE'LLGO but it'sx THEY'DGO I'mx WE'VEX",
    # Unicode simple case folding inside the contraction group: the reference's
    # Rust (?i:...) folds U+017F (ſ) to 's', so "'ſ" is a contraction piece.
    # A rewrite as explicit [sS] classes misses it; these vectors pin it down.
    "it'ſ fine",
    "x'ſy",
    # mixed JA/EN
    "今日はmeetingが3件あります",
    "TTSモデルのlatencyを測る",
    # digits
    "2026年8月21日",
    "円周率は3.14です",
    "1234567890",
    # whitespace: newlines, multiple spaces, trailing space
    "line one\nline two\n\nline four",
    "a  b   c    d",
    "trailing space ",
    "tabs\tand\tmore",
    # emoji
    "おめでとう🎉",
    "🎉🎉 party 🎉🎉",
    # half/full-width forms (NFC does NOT fold these; the byte map must carry them)
    "ＡＢＣとABC",
    "７７個",
    "ｶﾀｶﾅとカタカナ",
    # added-token literals inside text
    "<|im_start|>user\nhi<|im_end|>",
    "<|s_0|><|s_12799|>",
    "<|s_1|>text<|s_2|>",
    "before<|endoftext|>after",
    "<think>reasoning</think>done",
    # near-miss: looks like a speech token but is out of range / malformed —
    # must go through BPE, not the added-token path
    "<|s_12800|>",
    "<|s_x|>and<|s_007|>",
    # long JA sentence (~100 chars) — also the perf benchmark input
    "本日は晴天なり、マイクのテスト中です。音声合成モデルの評価のために、"
    "少し長めの日本語の文章を読み上げています。句読点や、カタカナ、"
    "English、数字123なども含めて、全体でおよそ百文字程度になります。",
]

CHAT_CASES = [
    "こんにちは",
    "Hello! How are you today?",
]

# decode vectors are built from encode round-trips plus hand-picked id runs;
# resolved below once the tokenizer is loaded.


def main() -> None:
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)

    encode = []
    for text in ENCODE_TEXTS:
        ids = tok.encode(text, add_special_tokens=False)
        encode.append({"text": text, "ids": ids})

    chat = []
    for user in CHAT_CASES:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": user}],
            add_generation_prompt=True,
            tokenize=True,
        )
        # transformers 5.x wraps this in a BatchEncoding; unwrap to plain ints.
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        chat.append({"user": user, "ids": [int(i) for i in ids]})

    s = lambda n: 151669 + n  # noqa: E731 — speech token id
    decode_id_runs = [
        # pure speech-token run
        [s(0), s(1), s(2), s(12799)],
        # ChatML frame around speech tokens
        [151644, *tok.encode("assistant\n", add_special_tokens=False), s(5), s(6), 151645],
        # speech tokens interleaved with text
        [*tok.encode("音声:", add_special_tokens=False), s(100), s(200)],
        # eos / pad literals
        [151643, 151645, 151643],
        # plain text round-trips
        tok.encode("こんにちは、世界。", add_special_tokens=False),
        tok.encode("I've been here before.", add_special_tokens=False),
        tok.encode("ＡＢＣ ７７ ｶﾀｶﾅ 🎉", add_special_tokens=False),
        # a single multi-byte character split across BPE tokens still decodes
        tok.encode("𠮷野家で🍣", add_special_tokens=False),
    ]
    decode = []
    for ids in decode_id_runs:
        text = tok.decode(ids, skip_special_tokens=False)
        decode.append({"ids": ids, "text": text})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Pin the exact tokenizer.json these vectors were dumped from: every
    # consumer (tokenizer.test.ts, expected-tokens.ts, serve.mjs) resolves
    # tokenizer.json independently, and a silently different file would make
    # the vectors test the wrong thing.
    tokenizer_sha256 = hashlib.sha256(
        (Path(SNAPSHOT) / "tokenizer.json").read_bytes()
    ).hexdigest()
    payload = {
        "_rebuild": "cd spike/miotts && python3 dump_tokenizer_vectors.py",
        "snapshot": SNAPSHOT,
        "tokenizer_sha256": tokenizer_sha256,
        "encode": encode,
        "chat": chat,
        "decode": decode,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({len(encode)} encode, {len(chat)} chat, {len(decode)} decode)")


if __name__ == "__main__":
    main()
