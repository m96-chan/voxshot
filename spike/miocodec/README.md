# spike/miocodec

Step 1 of [#106](https://github.com/m96-chan/voxshot/issues/106): get MioCodec's
decoder producing a waveform in the browser, and find out what it costs.

**A spike, not shippable code.** Nothing here is exported from `src/`, nothing
here runs under `npm test`, and none of it is on the package's install path.

## What is and is not in git

| | in git | why |
| --- | --- | --- |
| `dump_golden.py` | **yes** | a golden nobody can regenerate is a golden nobody can trust |
| `golden/` | no | ~7 MB of raw floats; says nothing in a diff, and this script rebuilds it |
| `.venv/` | no | 2.9 GB |
| the checkpoint | no | 523 MB, lives in `~/.cache/huggingface` |

`golden/index.json` records the checkpoint's sha256, so a golden and the weights
it was taken from cannot drift apart unnoticed.

## Running the port's tests

```bash
npm install     # web-xpu-ops arrives as a file: dependency
npm test
```

**This assumes [web-xpu-ops](https://github.com/m96-chan/web-xpu-ops) is checked
out beside this repository** — `package.json` names it as
`file:../../../web-xpu-ops`. It is unpublished and ships raw TypeScript, so
there is nothing to install from a registry; `vitest.config.ts` inlines it,
because a `file:` symlink is still `node_modules` as far as Vitest's
"don't transform dependencies" rule is concerned.

`npm test` at the repository root does **not** run these. Its `include` is
`tests/**`, and these tests need goldens that are not in git — a suite that
skipped itself when its fixtures were absent would report success for having
checked nothing.

### The demo

`examples/mio-tts.html` decodes a committed token fixture in the browser and
plays the result. Build and check it with:

```bash
npm run build        # esbuild browser.ts -> examples/mio-tts.js
npm run check:demo   # drives the page in headless Chromium and reads the numbers back
```

`check:demo` runs the page twice, once with `?backend=cpu` and once on `auto`,
and compares **both** against the torch decode of the same tokens — not against
each other, since two implementations agreeing says nothing if they agree on the
wrong answer. Measured on this machine:

| backend | decode | RTF | vs torch |
| --- | ---: | ---: | ---: |
| reference (CPU) | 36.3 s | 5.98 | 1.64e-4 |
| WebGPU kernels | 28.0 s | 4.67 | 2.63e-4 |

**Neither number is a hardware measurement, and the second one especially is
not.** Chromium here reports its WebGPU adapter as `google swiftshader` — a
software rasteriser — whatever combination of `--enable-gpu`,
`--ignore-gpu-blocklist`, Vulkan feature flags, sandbox flags and headed-versus-
headless it is given, on a machine with an RTX 5090 and a working Vulkan ICD.
So the run checks that the **kernels compute the right thing**, and says nothing
about speed. `check:demo` prints a warning when the adapter is software, because
an RTF printed next to "WebGPU" looks exactly like a hardware number.

The WebGPU path is also a **hybrid**: `matmul`, `conv1d` and the inverse
transform are dispatched as kernels, and every other stage — layernorm, RoPE,
attention, group norm, the activations — stays on the reference
implementations. Those three are where the arithmetic is; the rest is linear in
its data. A full-kernel port would be a different measurement.

`npm run bench` runs the CPU decode in Node — RTF 4.90 there against 5.98 in the
browser.

### Why the GPU check runs in a browser rather than in Vitest

Dawn's native module works in plain Node — verified with a standalone dispatch —
but a `matmul` of `25x768x5` core dumps it, and under Vitest 4's fork pool the
worker dies in 2.6 seconds, before any GPU work starts. Upstream hit the same
wall on Vitest 2 and lists four pool configurations that did not help
(`scripts/test.mjs`), which are what `vitest.config.ts` here started from.
Chromium is the target for this code anyway, so that is where it is checked.

`check:demo` routes the checkpoint request to the local Hugging Face cache
rather than pulling 523 MB per run. The network path is checked separately and
more cheaply — the CDN answers 200 with `access-control-allow-origin: *` and a
content-length of 523,087,956, matching the golden's manifest.

### What is checked so far

`decoder.test.ts` — the **whole** decoder, stage by stage in graph order, so the
first failure names the culprit. Agreement is 2.1e-6 to 2.2e-5 relative to each
stage's peak. On the demo's real speech clip the whole 6-second waveform matches
the reference decode to **1.2e-4** relative.

`istft.test.ts` — the decoder's last stage on its own, taken first because it is
the one that had no expression in the op library until
[web-xpu-ops#92](https://github.com/m96-chan/web-xpu-ops/issues/92), and because
it needs no weights.

All three cases reproduce the reference waveform to **2.4–2.9e-7 relative**
against the peak, which is f32 round-off between a float64 JavaScript reference
and a float32 torch tensor. The other two padding modes are wrong by a whole hop
(`center`) or refuse outright (`none` fails NOLA at sample 0, where a periodic
Hann is zero), and the test asserts both — so it is passing for the reason it
claims.

## Regenerating the golden

```bash
uv venv --python 3.14 --system-site-packages .venv
VIRTUAL_ENV=.venv uv pip install --no-deps "miocodec @ git+https://github.com/Aratako/MioCodec.git"
VIRTUAL_ENV=.venv uv pip install einops julius safetensors ruamel.yaml json5 'jsonargparse[signatures]' huggingface_hub soundfile
VIRTUAL_ENV=.venv uv pip uninstall torch triton   # see below
.venv/bin/python dump_golden.py
```

Two things about that sequence are deliberate.

**`--system-site-packages`, then uninstalling torch from the venv.** `julius`
pulls torch transitively and would install 2.13 into the venv. Every numerical
convention in [web-xpu-ops](https://github.com/m96-chan/web-xpu-ops) was verified
against **torch 2.10**, so the venv is made to fall through to the system one
instead. The manifest records the version that actually ran.

**CPU, not CUDA.** `miocodec.model._get_autocast_context` enables bfloat16
autocast on CUDA — `torch.complex` has no bf16 support on CPU, so they only turn
it on where it works. A golden taken on the GPU would carry bf16 rounding into
every comparison and the port would be measured against a blurred target.

## The three cases, and what each one observes

A single decode does not exercise the whole graph, which is why there are three.

| case | tokens | audio length | STFT frames | observes |
| --- | ---: | ---: | ---: | --- |
| `aligned` | 25 | 24120 (derived) | 50 | the natural path — **interpolate is the identity here** |
| `resampled` | 25 | 20000 (forced) | 41 | the interpolate genuinely resamples, 50 → 41 |
| `windowed` | 80 | 76920 (derived) | 160 | 160 frames through a 65-wide attention window |

`aligned` is what `decode()` does on its own, and in it the conv upsample's 50
frames and the STFT length of 50 agree — so `F.interpolate` resamples 50 to 50
and changes nothing. **A port that dropped the interpolate entirely would pass
that case.** `manifest.json` records `interpolate_is_identity` per case rather
than leaving it to be worked out from the shapes.

Likewise both 25-token cases sit under the wave decoder's 65-wide window, so
neither can catch a port that attends to the whole sequence. `windowed` is the
one that can.

## Layout

```
golden/
  index.json              config, checkpoint sha256, torch version, all three cases
  <case>/
    manifest.json         per-tensor shape, byte count and sha256
    tokens.f32            input: content token indices (as f32)
    global_embedding.f32  input: the AdaLN-Zero conditioning vector, 128-dim
    content_embedding.f32 FSQ decode output
    after_prenet.f32      wave_prenet          Transformer 768, 6L x 12H, window 65 -> 512
    after_conv_upsample.f32  wave_conv_upsample  ConvTranspose1d k=2 s=2 (25 Hz -> 50 Hz)
    after_interpolate.f32    F.interpolate(mode="linear") to the STFT length
    after_prior_net.f32   wave_prior_net       ResNet x2: GroupNorm(32) -> SiLU -> Conv1d
    after_decoder.f32     wave_decoder         Transformer 512, 8L x 8H, window 65, AdaLN-Zero
    after_post_net.f32    wave_post_net        ResNet x2
    istft_linear.f32      istft_head.out       Linear [512 -> 1922], before the mag/phase split
    spec_real.f32 / spec_imag.f32   the complex spectrogram entering the inverse transform
    waveform.f32          24 kHz output
```

Every file is raw little-endian float32 in C order; shapes are in the manifest.

## Why per stage rather than only the waveform

A port checked at the output alone learns that something is wrong and nothing
about where — every stage's error arrives there together. The transformer stages
are the ones most likely to be subtly wrong (RoPE pairing, AdaLN ordering, the
window mask) in ways that still sound like speech.

## Known gap, before any of this is ported

MioCodec's inverse transform is **`padding="same"`** (adapted from X-Codec-2.0),
not `torch.istft`'s `center=True`. It overlap-adds to `(T-1)·hop + n_fft` and
then crops `(n_fft - hop)/2 = 720` from **both** ends, giving `T · hop` samples —
which is why every waveform above is exactly `stft_length × 480`.

`web-xpu-ops`'s `istft` has `center` true or false and a `length` that truncates
the tail. Neither reproduces that crop, and `center: false` throws before it gets
the chance: sample 0 is covered only by frame 0 at `n = 0`, where a periodic Hann
is exactly 0, so the `w²` envelope is 0 and the NOLA guard fires. `length` cannot
help — it trims the end, and the problem is the start.

Tracked upstream; the rest of the graph needs no op that is not already there.
