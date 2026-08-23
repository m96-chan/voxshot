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

**The DiT, and the flow loop around it.** Twelve `DiffusionBlock`s — joint
attention over the latent's own keys concatenated with the text, speaker and
caption states — and the 32-step rectified-flow sampler that runs them. The
whole loop, integrated from the reference's own initial noise, tracks the
reference's trajectory to **1.1e-6 of peak** at step 16 and again at step 31.

```
ok   x_t at step 16   max |diff| 4.05e-6 of peak 3.994  (1.01e-6 relative)
ok   x_t at step 31   max |diff| 6.79e-6 of peak 6.272  (1.08e-6 relative)
```

**Guidance is a batch, and only for the first half.** The conditional and two
unconditional variants go through one forward pass stacked on the batch axis
while `t >= 0.5`, then it drops to batch 1. That is readable straight off the
goldens — `[3, 97, 1280]` at step 0, `[1, 97, 1280]` at steps 16 and 31 — and
reading `x` as 291 tokens instead of three sequences of 97 was this port's first
mistake here. Six of nine block comparisons passed anyway, because the two later
ones really are batch 1.

**A prediction that turned out wrong, kept because it was worth testing.** The
timestep embedding evaluates `cos` near 999 radians, where float32 keeps about
four digits; torch computes it in float32 and this port in float64, and they
differ by 8.4e-5. That looked like it would compound over 32 integrated steps.
It does not — `cond_module`'s first `Linear` contracts it to 7.5e-8 before it
reaches a single AdaLN, and the loop ends no worse than a stage that never sees
it.

**The duration predictor decides the length before anything is sampled**, which
is what a non-autoregressive model needs and an autoregressive one does not. It
agrees to 7e-8 and predicts 97.07 frames — the 97 the pipeline used.

## End to end

```bash
cd spike/irodori && npm run say -- "こんにちは、今日はいい天気ですね。"
cd spike/irodori && npm run say -- "こんにちは。" --ref path/to/voice.wav
```

Text in, 48 kHz WAV out, and nothing calls torch: normalisation, the Unigram
tokenizer, ModernBERT-ja's 25 layers, the projector, the duration predictor, the
12-block DiT under 32 steps of rectified flow, and `spike/dacvae`'s codec at both
ends. With `--ref` the clip is loudness-normalised and encoded here, so the voice
is whatever was handed in; without it, the speaker condition is the latent
`dump_golden.py` recorded, which is exact and needs no GPU.

The two paths agree where they can be compared: the recorded latent asks for
97.07 latent frames and an encoded `reference-voice.wav` asks for 97.18.

**One gap, and it is speed.** There is no GPU backend for the Irodori half;
web-xpu-ops' CPU reference is the definition of correct and the slowest thing
available. The sixteen guided flow steps at batch 3 are most of it — 1446 s for
a four-second utterance. The codec at both ends uses `spike/dacvae`'s WebGPU
backend when a device is available, and says which one ran.

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
