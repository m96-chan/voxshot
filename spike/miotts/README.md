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
```

Speaker identity enters **only** through MioCodec's 128-dim global embedding —
the LM prompt carries just the text. That is how the reference server
([Aratako/MioTTS-Inference](https://github.com/Aratako/MioTTS-Inference))
works too: it feeds a preset embedding to the codec and never shows the LM any
reference audio. The demo uses the embedding from
`examples/mio-codec-fixture.json`; the encoder half (reference audio → a new
embedding, i.e. actual zero-shot cloning) is out of this spike's scope.

## Measured on this machine

RTX 5090, driver 610.57.04, headed Chromium with `--use-angle=vulkan`
(see `../miocodec/check-demo.mjs` for why those exact flags):

| stage | greedy, ja golden text |
| --- | ---: |
| LM prefill, 15 prompt tokens (no logits) | 35 ms |
| LM decode, 87 generated tokens | ~0.48 s, **~210 tok/s** |
| MioCodec decode, 87 tokens | 1.2–1.3 s |
| **text → 3.48 s of audio, total** | **~1.75 s (RTF 0.50)** |

Processing time is ~0.5× the audio's length, and the LM is a rounding error
next to the codec. Cold start — 583 MB of q8 weights plus the 523 MB codec
checkpoint over localhost, packing, GPU upload — is ~4 s on top. Sampled mode
(temperature 0.8, the reference server's default) runs at only ~25 tok/s:
upstream `llm/sampler.ts` does a full-vocab (164,480) CPU softmax per step.
Greedy is the validation path, so that cost is left where it is, noted.

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

Everything below `check:tts` runs in Vitest (`npm test`, 182 tests; no GPU —
Dawn kills Vitest workers, same story as the decoder spike).

| check | oracle | agreement |
| --- | --- | --- |
| `tokenizer.test.ts` | HF tokenizer vectors (sha-pinned) | exact ids |
| `text.test.ts` | reference `normalize_text` outputs | exact strings |
| `model.test.ts` | torch per-stage golden | ≤2.3e-6 rel; greedy ids exact |
| `model-q8.test.ts` | f32 golden + graph parity | see bounds above |
| `weights-q8.test.ts` | ops/quantize bit-parity, sha256, permutation | exact |
| `check:tts` (browser) | CPU q8 oracle + Node codec decode | ids exact; WAV 9.9e-5 rel |

`check:tts` (`DISPLAY=:1 npm run check:tts`) drives the real page: it spawns
the CPU oracle for the golden text, clicks the page's run button, and asserts
the GPU ids equal the oracle's exactly, the WAV is `960 × tokens` samples, the
audio matches an independent CPU decode of the same ids to 16-bit fidelity,
and the adapter is real hardware (it refuses to print SwiftShader numbers as
if they were measurements). ~4.5 min, nearly all of it the ~2.5 s/token CPU
oracle.

## Running it yourself

```bash
npm install
python3 dump_golden.py            # per-stage LM golden into golden/  (~35 s)
python3 dump_tokenizer_vectors.py # tokenizer vectors into golden/
python3 convert_weights.py        # bf16 -> q8 artifacts into q8/     (~6 s)
npm test                          # 182 tests, ~6 min (the q8 greedy is slow)
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

- **No encoder half**: voices are limited to saved global embeddings; actual
  zero-shot cloning from reference audio is a follow-up issue.
- **Sampled-mode throughput** (~25 tok/s) is CPU-sampler-bound, not GPU-bound.
- **Sampled audio is shape-checked, not listened to** — the check asserts
  well-formedness; judging how it sounds needs a human and the page.
- The E2E oracle covers the golden ja text; other inputs exercise the same
  code paths but have no committed expectation (`expected-tokens.ts` computes
  one for any text if wanted).
- `mio-tts.html` serves weights from localhost; there is no hosted-CDN story
  for the q8 artifacts yet.
