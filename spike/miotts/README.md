# spike/miotts

Step 2 of [#106](https://github.com/m96-chan/voxshot/issues/106), tracked as
[#117](https://github.com/m96-chan/voxshot/issues/117): the text → content
tokens half of MioTTS, so that `examples/mio-tts.html` is a real end-to-end
TTS page rather than a codec demo. The decoder half lives in
[`spike/miocodec`](../miocodec/README.md) and this spike imports it as-is.

**A spike, not shippable code.** Nothing here is exported from `src/`, nothing
here runs under the repository root's `npm test`, and none of it is on the
package's install path.

## What runs where

```
text ─ normalize (text.ts, = reference server's normalize_text)
     ─ ChatML prompt ids (tokenizer.ts, Qwen2 byte-level BPE)
     ─ MioTTS-0.6B, int8 on WebGPU (gpu-engine.ts)     → <|s_n|> ids
     ─ n = id − 151669                                  → codec indices
     ─ MioCodec decoder (../miocodec, WebGPU)           → 24 kHz waveform

ref audio ─ decodeAudioData → mono → 24 kHz (WebAudio)
          ─ MioCodec encoder (../miocodec, WebGPU)      → 128-dim embedding
```

Speaker identity enters **only** through MioCodec's 128-dim global embedding —
the LM prompt carries just the text. That is how the reference server
([Aratako/MioTTS-Inference](https://github.com/Aratako/MioTTS-Inference))
works too: it feeds a preset embedding to the codec and never shows the LM any
reference audio. The default voice is the embedding from
`examples/mio-codec-fixture.json`; the page's Voice section can instead make
one from any audio file — actual zero-shot cloning, below.

## Voice cloning

Pick "reference audio…" under 話者 and choose a clip. The page decodes it
(WebAudio, resampled to 24 kHz mono), runs `../miocodec/encoder.ts` —
`encodeGlobal`, the WavLM→ConvNeXt global path — through the codec's **GPU
backend** on the same shared device, and conditions every later synthesis on
the resulting 128-dim embedding. Embeddings are cached per file (name+size),
so a clip is encoded once; the encoder's 117 MB weight file
(`export_encoder_weights.py`'s artifact, served by serve.mjs) is fetched
lazily on the first encode, so the default voice never pays for it.

Measured on this machine: **1.17 s** to encode the 5.08 s `jp_ref1.wav` on the
GPU — the reference implementation needs 40–55 s for the same clip, which is
why the run path rides the WGSL kernels (the CPU reference stays what it is
everywhere else in this repo: the test oracle).

**The resample caveat.** The page resamples with Chrome's WebAudio resampler;
the encoder golden's 24 kHz input came through torchaudio's polyphase. The two
waveforms disagree by a measured **4.4e-2** of peak — yet the embeddings land
**5.97e-4** apart, because the encoder's pooled statistics attenuate input
skew by ~two orders of magnitude. `check:tts` therefore holds the page's
embedding to the golden at 3e-3 (end-to-end, resampler included) *and* to a
reference-CPU encode of the **same browser-resampled samples** at 5e-6
(measured 8.05e-7 — pure kernel arithmetic, the authoritative comparison for
the port; a Chrome resampler change can move the first bound but not this
one).

## Measured on this machine

RTX 5090, driver 610.57.04, headed Chromium with `--use-angle=vulkan`
(see `../miocodec/check-demo.mjs` for why those exact flags):

| stage | greedy, ja golden text |
| --- | ---: |
| LM prefill, 15 prompt tokens (no logits) | 35 ms |
| LM decode, 87 generated tokens | ~0.48 s, **~210 tok/s** |
| MioCodec decode, 87 tokens | 1.2–1.3 s |
| **text → 3.48 s of audio, total** | **~1.75 s (RTF 0.50)** |
| voice encode, 5.08 s reference clip (once per file) | 1.17 s |

Processing time is ~0.5× the audio's length, and the LM is a rounding error
next to the codec. Cold start — 583 MB of q8 weights plus the 523 MB codec
checkpoint over localhost, packing, GPU upload — is ~4 s on top.

Sampled mode (temperature 0.8, the reference server's default) used to run at
**26 tok/s** against greedy's 210, and the GPU had nothing to do with it:
upstream `llm/sampler.ts` sorts all 164,480 logits on the CPU every step,
38 ms a step, a 26 tok/s ceiling before a single matmul. `sampler.ts` now
serves each draw from a top-2048 window found with a bounded heap and falls
back to upstream only when the draw lands past it — same id, measured, not
assumed — which puts sampled mode at **139 tok/s** (ISSUE #120).

## The int8 decision, and what it costs

The checkpoint is bf16 (1.2 GB). web-xpu-ops speaks exactly one quantized
format — per-row absmax symmetric int8 with a fused W8A32 GEMV (`matvecQ8`) —
so that is the format: 583 MiB on disk and in VRAM, roughly half the download
and no f16 kernels needed (none exist upstream).

Measured cost, against the f32 per-stage golden:

- element error at the theoretical bound: ~scale/2 ≈ 0.0039 of each row's peak
- prompt logits: worst 4.8e-2 peak-relative (ja), 1.2e-2 (en) — argmax
  unaffected at the prompt boundary in both cases
- greedy trajectories **diverge from f32 at step 6** on the ja golden text
  (an adjacent-codebook near-tie flips), then walk a different but perfectly
  well-formed path: 87 speech tokens and a clean eos against f32's 83.

That divergence is why the GPU engine is validated by **bit-exact id equality
against the CPU int8 oracle** (`model-q8.ts`), not by matching the f32 token
sequence. Both halves of that comparison are themselves pinned: the f32 port
against torch per stage (≤2.3e-6), and the int8 graph against the f32 graph
with quantization removed (<1e-4, the "graph parity" test — which is also what
proves the RoPE channel permutation below).

## RoPE: one graph, two channel orders

HF Qwen3 rotates channel pairs `(j, j+64)`; web-xpu-ops' `ops/rope` rotates
`(2i, 2i+1)`. The f32 oracle (`model.ts`) implements HF's pairing directly so
its stage tensors compare 1:1 with the torch golden. The q8 artifacts instead
**pre-permute** `wq`/`wk` rows and the q/k-norm gammas at conversion time
(`convert_weights.py`, `ropePermuted: true` in the manifest), so the GPU path
uses the stock kernel unchanged. `model-q8.test.ts` proves the two graphs are
the same function before quantization enters, and `weights-q8.test.ts` proved
the manifest flag observes reality by running the converter once with the
permutation disabled and watching exactly the permutation tests fail.

## Checks, in dependency order

Everything below `check:tts` runs in Vitest (`npm test`, 199 tests; no GPU —
Dawn kills Vitest workers, same story as the decoder spike).

| check | oracle | agreement |
| --- | --- | --- |
| `tokenizer.test.ts` | HF tokenizer vectors (sha-pinned) | exact ids |
| `text.test.ts` | reference `normalize_text` outputs | exact strings |
| `model.test.ts` | torch per-stage golden | ≤2.3e-6 rel; greedy ids exact |
| `model-q8.test.ts` | f32 golden + graph parity | see bounds above |
| `sampler.test.ts` | upstream `llm/sampler.ts` | same id, 700 draws/vector; ≥4× faster |
| `weights-q8.test.ts` | ops/quantize bit-parity, sha256, permutation | exact |
| `../miocodec/encoder.test.ts` | torch per-stage encoder golden | ≤1e-4/5e-4 rel per stage |
| `check:tts` (browser) | CPU q8 oracle + Node codec decode | ids exact; WAV 9.9e-5 rel |
| `check:tts` voice clone | encoder golden + same-input CPU encode | 5.97e-4 / **8.05e-7** rel |

`check:tts` (`DISPLAY=:1 npm run check:tts`) drives the real page: it spawns
the CPU oracle for the golden text, clicks the page's run button, and asserts
the GPU ids equal the oracle's exactly, the WAV is `960 × tokens` samples, the
audio matches an independent CPU decode of the same ids to 16-bit fidelity,
and the adapter is real hardware (it refuses to print SwiftShader numbers as
if they were measurements). It then clones a voice from `jp_ref1.wav` (the
clip the encoder golden was dumped from), holds the page's embedding to both
bounds in the Voice-cloning section above, and asserts the cloned-voice
greedy ids **equal** the default voice's — the LM never sees the voice, so
that equality is free and sharp — with a well-formed `960 × tokens` WAV.
~5 min, nearly all of it the ~2.5 s/token CPU oracle.

## Running it yourself

```bash
npm install
python3 dump_golden.py            # per-stage LM golden into golden/  (~35 s)
python3 dump_tokenizer_vectors.py # tokenizer vectors into golden/
python3 convert_weights.py        # bf16 -> q8 artifacts into q8/     (~6 s)
(cd ../miocodec && .venv/bin/python dump_encoder_golden.py \
                && .venv/bin/python export_encoder_weights.py)
                                  # encoder golden + weights — serve.mjs
                                  # refuses to start without the weights
npm test                          # 199 tests, ~6 min (the q8 greedy is slow)
npm run build                     # browser.ts -> ../../examples/mio-tts.js
npm run serve                     # port 8082
# open http://localhost:8082/mio-tts.html in Chromium with the WebGPU flags
DISPLAY=:1 npm run check:tts      # the end-to-end check
```

Requires [web-xpu-ops](https://github.com/m96-chan/web-xpu-ops) checked out
beside this repository (`file:` dependency, raw TypeScript, inlined by
Vitest), the MioTTS-0.6B and MioCodec-25Hz-24kHz checkpoints in
`~/.cache/huggingface` (the golden/convert scripts fetch them), and system
torch 2.10 + transformers 5.3 for the Python scripts. `golden/` and `q8/` are
gitignored and regenerable; every manifest records the checkpoint sha256 it
was derived from, so artifacts and weights cannot drift apart unnoticed.

## Known gaps

- **Voice-clone audio is shape-checked, not listened to**: the embedding is
  held to the golden and the ids to the default voice's, but whether the
  cloned voice *sounds like* the reference is for a human and the page.
- **Sampled mode is still ~35% slower than greedy** (139 vs 210 tok/s). What
  is left is `sampler.ts`' own two passes over the 164,480 logits — the
  exponentials and the heap scan, ~2.6 ms a step against greedy's 0.33 ms
  argmax. Taking the top-k on the GPU (option 3 in ISSUE #120, which would
  also cut the 657 KB logits readback) is the next move if it matters.
- **A freshly seeded run's first sampled token is effectively greedy.**
  `xorshift32(seed)` returns `seed × 6.3e-5` on its first call, so for any
  small seed the first draw lands on the argmax. Pre-existing, unrelated to
  #120, and it costs one token of variety per run — but a seed sweep over
  first tokens would measure almost nothing until the generator is warmed.
- **Sampled audio is shape-checked, not listened to** — the check asserts
  well-formedness; judging how it sounds needs a human and the page.
- The E2E oracle covers the golden ja text; other inputs exercise the same
  code paths but have no committed expectation (`expected-tokens.ts` computes
  one for any text if wanted).
- `mio-tts.html` serves weights from localhost; there is no hosted-CDN story
  for the q8 artifacts yet.
