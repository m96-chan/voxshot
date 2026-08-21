# spike/dacvae

DACVAE's decoder on web-xpu-ops, checked against Meta's reference — and the
first thing in voxshot to use `ops/snake`.

ISSUE [#138](https://github.com/m96-chan/voxshot/issues/138). Phase 1 of
implementing Irodori-TTS v4: its codec, on its own, before any of the model
that feeds it.

## Why the codec first, and why this codec

`snake` was one of fourteen operations in web-xpu-ops that no voxshot model had
ever called. Upstream's tests say each op computes what it claims; they cannot
say *we* are calling it correctly, and #124 produced two examples of that gap
being real — `gather`'s `rows` misread as an output count, and
`splitHeadsMajor`/`mergeHeadsMajor`'s argument order being the reverse of each
other. **An operation nobody has used against a real checkpoint is an
unverified contract, not an asset.**

DACVAE is the cheapest place to settle it. The codec is a standalone
repository, it decodes to a waveform on its own, and its whole main path is
three operations — `conv`, `conv_transpose`, `snake`. That makes it the
simplest graph voxshot has ported, and the fastest way to find out whether the
op inventory is worth what #137 claimed.

## Running

```bash
python3 -m venv .venv && .venv/bin/pip install torch torchaudio \
    "git+https://github.com/facebookresearch/dacvae" descript-audiotools
.venv/bin/python dump_weights.py   # ~250 MB, the decode path only
.venv/bin/python dump_golden.py    # ~75 MB, stage by stage
npm install && npm run check
```

Neither the weights nor the goldens are in git. Both are regenerable, and every
reader fails with the rebuild command rather than letting a run go green over
nothing.

## Result

```
48000 Hz, hop 1920, rates [12, 10, 8, 2]
latent [32, 30] -> waveform [1, 57600]

ok  after_out_proj     [1024, 30]     peak-relative 2.845e-7
ok  decoder_model_0    [1536, 30]     peak-relative 3.927e-7
ok  decoder_model_1    [768, 360]     peak-relative 8.690e-7
ok  decoder_model_2    [384, 3600]    peak-relative 1.859e-6
ok  decoder_model_3    [192, 28800]   peak-relative 1.269e-6
ok  decoder_model_4    [96, 57600]    peak-relative 3.003e-6
ok  tail_0_snake1d     [96, 57600]    peak-relative 3.702e-6
ok  tail_1_normconv1d  [1, 57600]     peak-relative 1.947e-6
ok  tail_2_tanh        [1, 57600]     peak-relative 1.951e-6
```

Stage by stage, not end to end: the decoder upsamples 1920x, so an error in the
first block is unrecognisable by the last and "the waveform is wrong" would be
true and useless.

**`snake` works, and our understanding of its contract was right.** DACVAE
computes `x + (alpha + 1e-9).reciprocal() * sin(alpha*x)²` and upstream's
`SNAKE_EPS` is `1e-9` — the same formula down to the epsilon, checked before a
line was written rather than discovered afterwards.

## What the green line is worth

It passed on the first run, so five decisions were sabotaged to find out
whether it was measuring anything:

| broken | result |
| --- | --- |
| `weight_norm` not folded (`g` used raw) | **red** from the first stage |
| `conv` padding without the dilation factor | **throws** — the residual shortcut cannot centre-crop |
| `snake`'s alpha made channel-uniform | **red** at `decoder_model_1` |
| the residual add dropped | **red** at `decoder_model_1` |
| `output_padding` forced to 0 | **green** — see below |

The last one is not observable here, and that is a fact about this checkpoint
rather than about the check. `output_padding` is `1 if stride % 2 else 0` and
the rates are [12, 10, 8, 2] — every one even. The rule is kept because it is
the reference's and an odd-rate rung would need it, but **nothing here has
verified it.**

## What is deliberately not implemented

Roughly 40% of the checkpoint.

- **The encoder** (27 M parameters). Decoding never touches it.
- **The watermark branches inside each `DecoderBlock`.** The block interleaves
  two paths in one `ModuleList` and `forward()` walks only the even chunks; the
  odd ones are reached solely by `upsample_group()`/`downsample_group()`, which
  only the real `watermark()` calls — and Irodori replaces that method.

What *is* implemented, and is easy to miss: the output tail lives inside
`wm_model.encoder_block.pre`. `Decoder.forward` always calls `watermark()`, the
checkpoint ships `alpha = 0.25`, and Irodori's override keeps
`forward_no_conv` — `Snake -> Conv(96->1) -> Tanh`. Setting alpha to zero and
stopping there returns 96 channels, which is not a waveform.

## What this does not show

- **Speed.** Everything runs on the reference implementations, on a CPU. This
  says the arithmetic is right, not that it is fast enough for a browser.
- **Round-trip quality.** The golden's input is a synthetic signal — chosen so
  the comparison is reproducible and unlicensed — and its round-trip
  correlation is only 0.70, because a codec trained on speech does not
  reconstruct broadband noise. Judging reconstruction needs real speech and is
  a separate exercise.
- **Anything above the codec.** The DiT, ModernBERT-ja, the speaker encoder and
  the duration predictor are phase 2.

## Licences

`facebookresearch/dacvae` is Apache-2.0. `Aratako/Semantic-DACVAE-Japanese-32dim`
and `Aratako/Irodori-TTS` are MIT.
