# ZeroVox Browser Demo

A minimal page to verify text-to-speech end to end in a real browser:
type text, press **Speak**, and ZeroVox synthesizes and plays it.

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
- **Chatterbox** — the real zero-shot TTS model
  (`onnx-community/chatterbox-multilingual-ONNX` via `@huggingface/transformers`).
  The first run downloads the model into the browser cache. Upload a few
  seconds of reference audio, then Speak clones that voice and synthesizes
  with it. WebGPU is used when available, WASM otherwise.
