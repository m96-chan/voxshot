# VoxShot Browser Demo

A minimal page to verify text-to-speech end to end in a real browser:
type text, press **Speak**, and VoxShot synthesizes and plays it.

## Run

```bash
# 1. Build the library (the example consumes the built package via file:../..)
cd ../..
npm install
npm run build

# 2. Start the demo
cd examples/browser
npm install
npm run dev
```

Open the printed URL (default <http://localhost:5173>).

## Engines

- **Placeholder** (default) — no model download. Renders speech-shaped tones,
  so you can verify the whole pipeline (clone → synthesize → play) instantly.
  The reference voice is generated in code; no file needed.
- **Chatterbox (English)** — the real zero-shot TTS model
  (`onnx-community/chatterbox-ONNX` via `@huggingface/transformers`).
  The first run downloads the model into the browser cache. Upload a few
  seconds of reference audio, then Speak clones that voice and synthesizes
  with it. WebGPU is used when available, WASM otherwise.
- **Chatterbox Multilingual (experimental, currently broken)** — the
  multilingual weights are not loadable straight from the Hub (the repo is
  missing the configs Transformers.js needs), so this variant serves a
  locally assembled copy. Download it once (~3.7 GB):

  ```bash
  bash scripts/download-multilingual.sh
  ```

  **Known blocker:** the multilingual checkpoint requires classifier-free
  guidance during generation, which Transformers.js has not shipped yet
  (open PR: [huggingface/transformers.js#1705](https://github.com/huggingface/transformers.js/pull/1705)).
  Until that lands, this engine loads and runs but produces only short
  unintelligible vocalizations — for any language, on any backend (verified
  against native CPU inference as well). The English Chatterbox engine is
  unaffected. Once the PR is released, updating `@huggingface/transformers`
  and passing `guidance_scale` should make this variant work.
