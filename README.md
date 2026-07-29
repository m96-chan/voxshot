# ZeroVox

> Browser-first Zero-Shot Text-to-Speech for JavaScript and TypeScript.

ZeroVox is a lightweight JavaScript/TypeScript library that enables **zero-shot voice cloning** and **high-quality text-to-speech** directly in modern web browsers.

No Python.
No backend.
No API keys.

Powered by WebGPU, ONNX Runtime Web, and modern open-source speech models.

> **Status:** 🚧 Early development (Proof of Concept)

> **What works today:** the full browser pipeline — reference audio decoding,
> voice extraction, voice persistence, text chunking, streaming synthesis and
> playback — plus a real zero-shot engine: **Chatterbox ONNX via
> Transformers.js v4** (English; the multilingual repo is not yet loadable by
> Transformers.js), running on WebGPU with a WASM fallback, optionally
> inside a Web Worker.
>
> A dependency-free `PlaceholderEngine` (speech-shaped audio, not speech) is
> still bundled as the default, so the library works with nothing installed.
> The Chatterbox engine has **not yet been verified against the real model in a
> browser** — see [#2](https://github.com/m96-chan/zerovox/issues/2).

---

## Features

* 🎙️ Zero-shot voice cloning
* 🌐 Browser-first architecture
* ⚡ WebGPU acceleration
* 🖥️ WASM fallback
* 📦 Simple npm package
* 🧩 Framework agnostic
* 💾 Voice embedding cache
* 🔊 Audio streaming support (planned)
* 🇯🇵 Japanese text normalization (planned)
* 🌍 Multilingual support (planned)

---

## Why ZeroVox?

Most voice cloning projects require Python, PyTorch, or a backend server.

ZeroVox focuses on a different goal:

> **Make zero-shot TTS as easy as installing an npm package.**

```bash
npm install zerovox
```

No Docker.

No CUDA.

No Python environment.

Just JavaScript.

---

## Quick Start

```ts
import { ZeroVox } from "zerovox";

const tts = await ZeroVox.create();

await tts.cloneVoice(referenceAudioFile);

const audio = await tts.speak(
  "Hello! This voice was cloned directly inside your browser."
);

await audio.play();
```

---

## Browser Support

| Browser | Status |
| ------- | ------ |
| Chrome  | ✅      |
| Edge    | ✅      |
| Brave   | ✅      |
| Firefox | 🚧     |
| Safari  | 🚧     |

WebGPU is recommended for the best performance.

A WASM fallback will be available for unsupported environments.

---

## API

```ts
const tts = await ZeroVox.create({
  device: "auto",   // "auto" | "webgpu" | "wasm"
  model: "default"
});

tts.device;          // the backend that was actually selected
tts.sampleRate;      // sample rate of the audio this instance produces

await tts.cloneVoice(file);   // ArrayBuffer | Blob | File | typed array | { samples, sampleRate }
await tts.speak(text);        // -> SynthesizedAudio
await tts.speak(text, { speed: 1.2 });

for await (const chunk of tts.stream(text)) {
  await chunk.play();         // play sentence by sentence, no need to wait for the rest
}

await tts.saveVoice("alice");
await tts.useVoice("alice");
await tts.listVoices();       // ["alice"]
await tts.deleteVoice("alice");

await tts.dispose();
```

`speak()` and `stream()` return `SynthesizedAudio`:

```ts
audio.samples;     // Float32Array, mono
audio.sampleRate;
audio.duration;    // seconds
audio.toWav();     // ArrayBuffer (16 bit PCM RIFF)
audio.toBlob();    // Blob, type "audio/wav"
await audio.play();
```

### Real voice cloning with Chatterbox

```bash
npm install zerovox @huggingface/transformers
```

```ts
import { ChatterboxEngine, ZeroVox } from "zerovox";

const engine = new ChatterboxEngine({
  // "onnx-community/chatterbox-ONNX" (English) by default — the multilingual
  // repo currently lacks the config files Transformers.js needs to load it
  onProgress: (p) => console.log(p.status, p.file, p.progress),
});

const tts = await ZeroVox.create({
  engine,
  minChunkLength: 20,   // very short prompts destabilise the model
});

await tts.cloneVoice(referenceAudioFile);   // 5-15s of clean speech
await (await tts.speak("こんにちは、世界。")).play();
```

* `@huggingface/transformers` is an **optional peer dependency**, imported
  lazily. Nothing is downloaded unless you actually construct the engine.
* Model weights are cached by Transformers.js in the browser's Cache Storage
  (`env.useBrowserCache`), so only the first load pays the download.
* Device and quantization are chosen for you and degrade automatically:
  WebGPU `q4f16` → WebGPU `q4` → WASM `q4`. Override per session with `dtype`.
* Output is 24 kHz — the S3Gen vocoder's rate.
* `speed` is applied by resampling the rendered waveform, so it shifts pitch
  like a playback-rate change. Chatterbox exposes no duration control.

### Keeping inference off the UI thread

```ts
// tts.worker.ts
import { ChatterboxEngine, exposeEngine, type RpcEndpoint } from "zerovox";

const engine = new ChatterboxEngine({ onProgress: (p) => serve.emitProgress(p) });
const serve = exposeEngine(engine, self as unknown as RpcEndpoint);
```

```ts
// main thread
import { WorkerSynthesisEngine, ZeroVox } from "zerovox";

const worker = new Worker(new URL("./tts.worker.ts", import.meta.url), { type: "module" });
const engine = new WorkerSynthesisEngine(worker, {
  onProgress: (p) => updateProgressBar(p),
});

const tts = await ZeroVox.create({ engine });
```

Audio crosses the boundary as a transferable buffer, always as a copy, so the
caller's `Float32Array` is never detached. The transport is a small typed
`postMessage` protocol — no Comlink dependency required, though Comlink works
equally well if you prefer it: `exposeEngine` only needs an object with
`postMessage` / `addEventListener`.

### Bring your own model

Every part of the pipeline is injectable, so a real model only has to
implement `SynthesisEngine`:

```ts
import { ZeroVox, type SynthesisEngine } from "zerovox";

class MyOnnxEngine implements SynthesisEngine {
  readonly name = "my-model";
  readonly sampleRate = 24_000;

  async load(device) { /* ... */ }
  async embed(audio) { /* -> Float32Array speaker embedding */ }
  async synthesize({ text, voice, speed }) { /* -> Float32Array samples */ }
  async dispose() { /* ... */ }
}

const tts = await ZeroVox.create({ engine: new MyOnnxEngine() });
```

The voice store (`VoiceStore`) and the browser bindings (`Platform`:
decoder / player / GPU probe) are injectable in the same way.

---

## Development

```bash
npm install
npm test          # vitest + coverage (90% threshold, enforced)
npm run typecheck
npm run build
```

A runnable browser demo (text box → synthesize → play) lives in
[`examples/browser`](./examples/browser). See its README for setup.

Contribution rules — TDD, coverage, and ticket-driven development — are in
[CLAUDE.md](./CLAUDE.md).

---

## Design Goals

* Browser-first
* Zero dependencies on Python
* Clean TypeScript API
* Easy integration
* Privacy-friendly (everything runs locally)
* Pluggable model architecture

---

## Roadmap

### v0.1

* Browser-only inference
* Voice cloning
* Basic TTS
* TypeScript SDK

### v0.2

* Streaming synthesis
* Voice management
* IndexedDB cache

### v0.3

* Japanese text normalization
* Better sentence segmentation
* Audio queue

### v0.4

* Multiple model support
* Emotion control
* Speech-to-Speech

---

## Vision

ZeroVox aims to become the browser-native voice toolkit for modern web applications.

Possible use cases include:

* AI assistants
* Virtual avatars
* News readers
* Accessibility tools
* Games
* Interactive storytelling
* Voice-enabled web apps

---

## License

MIT

---

## Contributing

Contributions, bug reports, and feature requests are welcome.

If you have ideas for improving browser-based TTS or voice cloning, feel free to open an issue or submit a pull request.

---

## Acknowledgements

ZeroVox builds upon the incredible work of the open-source speech AI community, including projects such as:

* ONNX Runtime Web
* Transformers.js
* Chatterbox
* WebGPU
* The broader open-source TTS ecosystem

Thank you to everyone pushing browser AI forward. ❤️
