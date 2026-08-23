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

**The download is 3.8x MioTTS's.** 4.3 GB against 1.12 GB, and ModernBERT-ja is
1.26 GB of that — a third of the model's parameters are a text encoder. There
are `-Quantized` checkpoints published alongside; they have not been looked at.

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

Irodori's `normalize_text` also differs from MioTTS's at every point that
matters: spaces survive, bracket stripping walks the string for depth rather
than checking the first and last character, NFKC replaces the explicit width
tables. Two models by the same author, no shared normalisation.

## What this does not settle

- **Quality.** Nobody has listened. That is the point of the files.
- **Whether to port it.** `sample_rf` is 50% of the run and needs the DiT;
  ModernBERT-ja is a third of the weights and is a second model. Neither cost
  is known until the quality question is answered.
- **Browser viability.** These are torch numbers on a workstation GPU.

## Licences

Code MIT (`Aratako/Irodori-TTS`), weights MIT with ethical restrictions,
`facebookresearch/dacvae` Apache-2.0, ModernBERT-ja under its own terms
(`sbintuitions/modernbert-ja-310m`, not checked here).
