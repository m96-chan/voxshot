# spike/irodori

Irodori-TTS v4.1 run against its own reference implementation, so there is
something to listen to before deciding whether to port it.

Phase 2 of ISSUE [#138](https://github.com/m96-chan/voxshot/issues/138) begins
here. The codec is already ported (`spike/dacvae`); the model above it — a
12-layer DiT, ModernBERT-ja 310M, an 8-layer speaker encoder and a duration
predictor — is not, and is a large piece of work. **Running the reference first
is the cheapest way to find out whether it is worth doing**, and it is the first
step of doing it either way: a port needs something to be wrong against.

## What is here

`samples/` holds four sentences rendered twice — once by Irodori v4.1, once by
MioTTS (voxshot's shipped engine) — **from the same reference clip, with the
same seed**. `reference-voice.wav` is the clip both cloned from.

| | text |
| --- | --- |
| 01 | ……ん。来たんだ。 |
| 02 | 全部追うの、無理だよ？ |
| 03 | こんにちは、今日はいい天気ですね。散歩でもしませんか。 |
| 04 | えっと、それはちょっと、わたしには難しいかもしれません。 |

They are not in git — regenerable, and audio. See "Reproducing" below.

**Nobody has judged these.** The measurements below are measurements; which one
sounds better is a listening question and belongs to whoever is choosing.

## Measured

Both on this machine's RTX 5090. **Neither is voxshot's port** — Irodori is its
reference implementation in torch, MioTTS is voxshot's engine on WebGPU. They
are not the same kind of number and are put side by side only for scale.

| | Irodori v4.1-Small | MioTTS 0.6B |
| --- | --- | --- |
| output | **48 kHz** | 24 kHz |
| generation | **non-autoregressive**, 32 flow steps | autoregressive, ~25 tokens/s of audio |
| `total_to_decode` | **0.76 – 0.88 s** | — |
| RTF | **≈ 0.12 – 0.20** (torch) | 0.49 – 0.53 (voxshot on WebGPU) |
| download | 3.06 GB + 1.26 GB (ModernBERT-ja) | 1.12 GB |

Per-stage, for one sentence:

```
tokenize_text        1.5 ms
prepare_reference  260.5 ms      ← encodes the reference clip
predict_duration   104.6 ms
sample_rf          405.4 ms      ← 32 flow-matching steps, the bulk of it
decode_latent       44.2 ms      ← the part spike/dacvae already ports
```

Three things follow.

**Non-autoregressive is structurally cheaper.** 32 steps regardless of length,
against one forward pass per generated token. That is where the speed
difference comes from, and it is the same observation #136 made about
OmniVoice — with the difference that this one has a usable licence.

**The part voxshot has already ported is 5% of the run.** `decode_latent` is
44 ms of 817. The expensive half is `sample_rf`, and porting it means the DiT.

**The download is 2.7x MioTTS's** — 3.06 GB against 1.12 GB, not the 4.3 GB an
earlier reading of this said. ModernBERT-ja's 1.26 GB is *inside* that
checkpoint: Irodori fine-tunes the backbone and ships all 152 of its tensors
under `pretrained_text_backbone.backbone.`. The separate 1.26 GB Hugging Face
download supplies `config.json` and `tokenizer.json` and nothing else — the
reference does not load those weights either
(`load_pretrained_backbone_weights=not use_pretrained_text_encoder`).

Still, a third of the model's parameters are a text encoder. There are
`-Quantized` checkpoints published alongside; they have not been looked at.

## What the samples show without listening

| | Irodori | MioTTS |
| --- | --- | --- |
| #01 | 3.88 s | 4.32 s |
| #02 | **3.84 s** | **2.68 s** |
| #03 | **6.92 s** | 5.68 s |
| #04 | **7.32 s** | 5.60 s |
| rms | 0.123 – 0.147 | 0.056 – 0.059 |

**Irodori speaks slower** — noticeably so on the longer sentences, 7.3 s against
5.6 s for the same text. Whether that is better pacing or dragging is a
listening question; it is a duration predictor making a choice, not a bug.

**Irodori is about 8 dB louder** at the same reference. Its peak touches full
scale on one sample out of 186,240, which was checked rather than assumed —
that is a loud transient, not clipping.

## Reproducing

```bash
# deps live in spike/dacvae/.venv (dacvae, transformers, peft, ...)
cd /tmp && git clone https://github.com/Aratako/Irodori-TTS
cd Irodori-TTS && /path/to/spike/dacvae/.venv/bin/python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small \
  --text "……ん。来たんだ。" \
  --ref-wav ../samples/reference-voice.wav \
  --output-wav out.wav --seed 42 --show-timings
```

`Aratako/Irodori-TTS-v4.1-Small` is newer than the v4-Small that #138's
research recorded, and newer still than the v3 the only existing browser port
(`ngc-shj/irodori-tts-webgpu`) targets.

## Ported so far

**The text path.** `normalize_text` and the tokenizer, checked against 27
vectors from the reference — both halves separately, because they fail
differently and a single number would hide the useful one.

```
all 27 vectors agree — normalization and Unigram both, against ModernBERT-ja @ 77675fc9
```

Two things were learned by getting them wrong first.

**Byte fallback is not a lattice candidate.** The `<0xNN>` pieces carry a score
of **0** in this vocabulary while every real piece is negative, so offering them
as candidates makes byte fallback win every position and the whole corpus
tokenizes one byte at a time — which is exactly what the first run did, 24
vectors out of 27. Byte fallback applies *after* the path is chosen, to
characters the vocabulary could not cover.

**Unigram is not BPE.** BPE merges greedily by rank; Unigram scores every
segmentation and takes the best. Replacing the Viterbi with greedy longest-match
fails only **3 of the 27 vectors** — most inputs agree — which is the argument
for porting the real algorithm rather than the one already to hand. A port that
was "mostly right" here would produce fluent output that says something slightly
else.

**ModernBERT-ja.** All 25 layers, on six web-xpu-ops operations — `gather`,
`layernorm`, `matmul`, `rope`, `attention`, `activation`. Every captured stage
agrees with Irodori's own fine-tuned backbone to ~1e-7 of peak, for two inputs.

```
18 stage comparisons agree within 5e-6 of peak, against Irodori's own ModernBERT-ja
```

**RoPE is the interesting part.** web-xpu-ops rotates adjacent lanes
`(x[2i], x[2i+1])`; HF's `rotate_half` rotates split halves `(x[i], x[i+d/2])`.
The two are the same rotation under a permutation of the head dimension, and
`q·k` is invariant when that permutation is applied to both — so the Q and K
rows of `Wqkv` are interleaved once at load and web-xpu-ops' own `rope` is used
unchanged. No second RoPE, and nothing downstream sees the permuted axis.

**Padding does not have to be computed.** Irodori pads every text to 256, but
key padding blocks column `j >= n` in both layer types, so rows `0..n-1` attend
only to each other at every layer. Truncating is exact, and for the 7-token
sample it is the difference between **2.9 s and 111.3 s** on the CPU reference —
both checked against the same goldens.

**A short input cannot check the sliding window.** With 7 real tokens,
`|i - j| <= 64` never binds: deleting the sliding pattern entirely, or widening
the window by one, both leave the check green. A second 78-token golden was
added for that reason, and it is the only one those two sabotages fail. The
tolerance was tightened for the same class of reason — at 2e-4 the check could
not distinguish exact `gelu` from its tanh approximation.

**The speaker encoder.** Eight `TextBlock`s over the reference clip's patched
DACVAE latents — this is the whole of the speaker identity, which is what makes
it zero-shot. It agrees to ~3e-6 of peak.

It runs on the *recorded input* rather than on anything ported: `dump_golden.py`
now captures each module's arguments as well as its output, so the speaker
encoder is checkable without the DACVAE encoder existing in TypeScript. That
inverted the porting order for everything left.

Irodori's own blocks share no convention with ModernBERT-ja — RMSNorm not
LayerNorm, SwiGLU not GeGLU, one RoPE base not two, a per-head Q/K norm, and a
sigmoid gate on the attention output that has no counterpart at all. The RoPE
lane convention is the opposite one too, which is why these blocks use
web-xpu-ops' `rope` directly where ModernBERT needed its weights permuted.

**One decision here is unverified.** The reference zeroes masked positions after
every block; a single reference clip at batch 1 has no padding, so removing that
changes nothing and the check stays green. It is ported because the reference
does it, not because anything has shown it matters.

Irodori's `normalize_text` also differs from MioTTS's at every point that
matters: spaces survive, bracket stripping walks the string for depth rather
than checking the first and last character, NFKC replaces the explicit width
tables. Two models by the same author, no shared normalisation.

## What this does not settle

- **Quality.** Nobody has listened. That is the point of the files.
- **Speed.** Everything ported so far runs on web-xpu-ops' CPU reference, which
  is the definition of correct and the slowest thing available. No GPU backend
  yet, so there is no RTF for the port.
- **The rest of the model.** The DiT, the speaker encoder and the duration
  predictor are not ported. `sample_rf` is 50% of the reference's run.
- **Browser viability.** The torch numbers above are a workstation GPU.

## Licences

Code MIT (`Aratako/Irodori-TTS`), weights MIT with ethical restrictions,
`facebookresearch/dacvae` Apache-2.0, ModernBERT-ja under its own terms
(`sbintuitions/modernbert-ja-310m`, not checked here).
