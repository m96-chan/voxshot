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
npm install
npm run check       # reference implementations, on the CPU
npm run check:gpu   # the same graph through the WGSL kernels
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

On WebGPU (RTX 5090), against the same goldens:

```
ok  after_out_proj     peak-relative 2.845e-7      ok  decoder_model_4    7.553e-6
ok  decoder_model_0    2.906e-6                    ok  tail_0_snake1d     8.065e-6
ok  decoder_model_1    3.318e-6                    ok  tail_1_normconv1d  4.372e-6
ok  decoder_model_2    5.218e-6                    ok  tail_2_tanh        4.241e-6
ok  decoder_model_3    3.299e-6

3 runs: 0.77 / 0.68 / 0.69 s — best RTF 0.57 for 1.20 s of audio
```

The GPU's errors are larger than the CPU's — 8.1e-6 against 3.7e-6 at the worst
stage — which is what a different summation order inside a kernel looks like,
and is still two orders inside the tolerance. **Both backends are held to the
reference, never to each other**: "the two agree" would be satisfied by two
identical mistakes.

Stage by stage, not end to end: the decoder upsamples 1920x, so an error in the
first block is unrecognisable by the last and "the waveform is wrong" would be
true and useless.

## RTF 0.57, and where the headroom is

Faster than real time on the first attempt, with nothing tuned. The obvious
cost is still being paid in full: **every stage uploads its input and reads its
output back**, about 61 MB per second of audio in each direction across roughly
a hundred dispatches. Keeping activations on the device between stages removes
almost all of that, and is deliberately not done yet so the number above is the
honest unoptimised one.

For scale, the reference implementations on the CPU are RTF 138 — they are the
definition of correct and the slowest thing here, exactly as intended.

## Dawn collects the instance out from under the device

Getting the GPU path to run at all took finding this, and it is the most
generally useful thing in the spike.

**Dawn's Node binding does not keep its `GPU` instance alive from the
`GPUDevice`.** Once the instance becomes unreachable the collector takes it,
and dispatches after that crash — as a futex abort
(`std::system_error: Invalid argument`), a segfault, or a hang, whichever the
race lands on. Measured on 200 trivial dispatches, three runs each:

| how the device was made | result |
| --- | --- |
| inside an async function, instance and adapter dropped | hang / hang / abort |
| **inside the same async function, instance and adapter held** | **ok / ok / ok** |
| at module top level | ok / ok / ok |

The only difference between the first two rows is a reference. Fifty dispatches
pass; a hundred start failing — so it is not the count but how much opportunity
the collector has had.

Two other explanations were tested first and **both were wrong**: heavy CPU work
between dispatches (64 MB of allocate-and-scan per stage) does not reproduce it,
and nor do file reads between them. Guessing would have produced a confident
wrong answer twice.

`Gpu` therefore takes a `retain` argument and holds whatever produced the
device. That is not defensiveness, it is the fix.

Sent upstream as a hypothesis rather than a diagnosis — Dawn's own source has
not been read here — and **upstream confirmed it**. `harness/wgsl.ts`'s
`createRunner` was exactly this shape, and an A-B-A on `ops/gqa/wgsl.test.ts`
(the file web-xpu-ops #68 recorded as 0/5) went **0/5 with main, 5/5 holding
the instance in module scope, 0/4 after reverting**, with nothing else changed.
Four issues there — #38, #49, #68, #107 — had been accumulating descriptions of
the symptom for three weeks: "a test that takes more than a few milliseconds
before its first dispatch", "a file that holds too many dispatches". Both are
descriptions of *more opportunity for the collector to run*, and vitest calls a
test body as a function, so an instance created there is unreachable the moment
the body returns — which is why no pool configuration ever helped.

**How much of the story it is remains open.** Upstream's
`llm/engine-q8-resident.wgsl.test.ts` measured 2/3 during the experiment and
3/3 on a retest, which is too few runs to call either way. That file builds ~45
pipelines, and a separate limit on total GPU objects may still be at work there,
so upstream's PR states the claim as "at least three of the four issues" rather
than all of them.

Two independent causes producing one symptom would be a good explanation for
why this read as flake for so long: fixing either alone leaves failures
standing, so neither looks like a cause.

## The round trip, on real speech

    port vs reference : peak-relative 1.827e-6   correlation 1.000000
    round trip vs input: correlation 0.9859     rms 0.1245 against 0.1232
    4.00 s of audio in 2.04 s — RTF 0.51

Both are reported because they answer different questions. The first is about
this code and sits at the f32 noise floor. The second is about the **codec**,
which no amount of correct porting can improve — a disappointing reconstruction
would be Meta's and Aratako's, not ours. The CPU and GPU backends give the same
0.9859.

`roundtrip.ts` writes `input.wav`, `reference.wav` and `port.wav`. The numbers
are a summary; the files are the answer, and someone has to listen to them.

**`snake` needed chunking to get here.** It flattens `C * L` into one dispatch
dimension, so past about 3.5 seconds of 48 kHz audio it asks for 72,000
workgroups against a limit of 65,535 — which a browser hits too, that being the
spec's guaranteed minimum. It is chunked by channel: a run of whole channels is
itself a valid snake with a sliced alpha, so this stays inside the op's contract
rather than reaching for a dimension the kernel does not read. The convolutions
are unaffected; they dispatch `[ceil(Lout/256), Cout]`.

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

- **Speed in a browser.** RTF 0.57 is Dawn in Node on an RTX 5090. A browser's
  WebGPU is the same API over the same driver, but it is not the same
  measurement, and neither is a laptop's GPU.
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
