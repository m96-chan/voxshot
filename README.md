# ZeroVox

> Browser-first Zero-Shot Text-to-Speech for JavaScript and TypeScript.

ZeroVox is a lightweight JavaScript/TypeScript library that enables **zero-shot voice cloning** and **high-quality text-to-speech** directly in modern web browsers.

No Python.
No backend.
No API keys.

Powered by WebGPU, ONNX Runtime Web, and modern open-source speech models.

> **Status:** 🚧 Early development (Proof of Concept)

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

## Planned API

```ts
const tts = await ZeroVox.create({
  device: "auto",
  model: "default"
});

await tts.cloneVoice(file);

await tts.speak(text);

await tts.stream(text);

await tts.saveVoice("alice");

await tts.useVoice("alice");

await tts.deleteVoice("alice");
```

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
