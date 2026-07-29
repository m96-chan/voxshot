# ZeroVox

> Browser-first Zero-Shot Text-to-Speech for JavaScript and TypeScript.

ZeroVox is a lightweight JavaScript/TypeScript library that enables **zero-shot voice cloning** and **high-quality text-to-speech** directly in modern web browsers.

No Python.
No backend.
No API keys.

Powered by WebGPU, ONNX Runtime Web, and modern open-source speech models.

> **Status:** 🚧 Early development (Proof of Concept)

> **What works today (v0.1):** the full browser pipeline — reference audio
> decoding, voice extraction, voice persistence, text chunking, streaming
> synthesis and playback — behind a stable TypeScript API.
> The bundled engine is a **placeholder** that renders speech-shaped audio, not
> speech: it exists so the pipeline can ship and be tested while the ONNX
> Runtime Web backend is built ([#2](https://github.com/m96-chan/zerovox/issues/2)).
> Bring your own model today by implementing the `SynthesisEngine` interface.

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
