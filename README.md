# ZeroVox

> Browser-first Zero-Shot Text-to-Speech for JavaScript and TypeScript.

ZeroVox is a lightweight JavaScript/TypeScript library that enables **zero-shot voice cloning** and **high-quality text-to-speech** directly in modern web browsers.

No Python.
No backend.
No API keys.

Powered by WebGPU, ONNX Runtime Web, and modern open-source speech models.

[![npm](https://img.shields.io/npm/v/zerovox)](https://www.npmjs.com/package/zerovox)
[![CI](https://github.com/m96-chan/zerovox/actions/workflows/ci.yml/badge.svg)](https://github.com/m96-chan/zerovox/actions/workflows/ci.yml)

> **Status:** 🌱 v0.1.0 on npm — usable, API not frozen yet.

> **Verified end to end in a browser:** reference audio decoding, voice
> cloning, voice persistence, text chunking, streaming synthesis and gapless
> playback, driven by a real zero-shot engine — **Chatterbox ONNX via
> Transformers.js v4** on WebGPU (WASM fallback), optionally inside a Web
> Worker. Measured on an RTX 5090: 5.3 s of speech rendered in 2.7 s.
>
> **English only for now.** The multilingual Chatterbox checkpoint needs
> classifier-free guidance during generation, which Transformers.js has not
> shipped yet — tracked in
> [#25](https://github.com/m96-chan/zerovox/issues/25).
>
> A dependency-free `PlaceholderEngine` (speech-*shaped* audio, not speech) is
> the default, so the library runs with nothing else installed.

---

## Features

* 🎙️ Zero-shot voice cloning
* 🌐 Browser-first architecture
* ⚡ WebGPU acceleration
* 🖥️ WASM fallback
* 📦 Simple npm package
* 🧩 Framework agnostic
* 💾 Voice embedding cache (IndexedDB) + rendered-audio cache
* 🔊 Gapless streaming playback (AudioWorklet) with one-chunk prefetch
* 🇯🇵 Japanese reading conversion (`toJapaneseReading`)
* 🌍 Multilingual speech — blocked upstream, see [#25](https://github.com/m96-chan/zerovox/issues/25)

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

WebGPU is recommended for the best performance; environments without it fall
back to WASM automatically (`device: "auto"`).

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

### Gapless streaming playback

`play()` streams chunks into an AudioWorklet ring buffer, so sentences play
back to back with no scheduling gaps. The next chunk is synthesized while the
current one plays (one chunk of lookahead), and rendered audio is cached per
voice + text + speed, so repeating a phrase is instant.

```ts
const speech = tts.play("Long text. It starts playing before it is fully rendered.", {
  speed: 1.0,
  volume: 0.8,
});

speech.setVolume(0.5);  // live volume control
await speech.skip();    // jump past the chunk currently playing
await speech.stop();    // stop and discard everything
await speech.done;      // resolves when playback finished or was stopped
```

Like everything else, the output device is injectable: `play()` uses
`platform.streamingPlayer`, and the default browser implementation
(`BrowserStreamingAudioPlayer`) loads its worklet from an inline blob — no
extra asset to serve. Tune or disable the cache with
`ZeroVox.create({ synthesisCache: new SynthesisCache({ maxEntriesPerVoice: 8 }) })`
or `synthesisCache: null`.

### Japanese reading conversion

```ts
import { toJapaneseReading } from "zerovox";

toJapaneseReading("1,000円");             // "せんえん"
toJapaneseReading("会議は3月4日の14:00"); // "会議はさんがつよっかのじゅうよじ"
toJapaneseReading("AIが50%");             // "エーアイがごじゅうパーセント"
```

Numbers, dates, clock times, units, numeric symbols and upper-case acronyms
become kana readings. It is opt-in — run it before `speak()`/`play()` for
Japanese text; other languages should skip it.

Note that this normalizes *text*. Speaking the result needs a model whose
tokenizer covers Japanese, which the bundled English Chatterbox checkpoint
does not ([#25](https://github.com/m96-chan/zerovox/issues/25)).

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
await (await tts.speak("Cloned from a few seconds of reference audio.")).play();
```

* **English only.** This checkpoint's tokenizer has no kana or CJK tokens, so
  non-Latin text maps to unknown tokens and comes out near-silent. Japanese
  speech needs the multilingual checkpoint —
  [#25](https://github.com/m96-chan/zerovox/issues/25).
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

### Releasing

CI runs typecheck, tests (90% coverage enforced) and the build on every push
and pull request. Publishing is driven by GitHub Releases:

1. Bump the version and land it on `main`: `npm version <patch|minor|major>`
2. Create a GitHub Release whose tag is `v<version>` (matching `package.json`;
   the workflow fails the publish if they disagree)
3. The `Publish` workflow re-runs the checks and publishes to npm with
   [provenance](https://docs.npmjs.com/generating-provenance-statements),
   using the repository's `NPM_TOKEN` secret

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

## Performance

Measured on an RTX 5090 / Linux Chrome, English Chatterbox on WebGPU:

| | |
| --- | --- |
| Synthesis | ~0.5× real time (5.3 s of speech in 2.7 s) |
| Model load, warm cache | ~56 s — dominated by ONNX session creation, not download |
| Model download, first run | ~1.5 GB |

Notes:

* Linux Chrome does not expose `shader-f16`, so the engine automatically
  degrades from the `q4f16` language model to `q4` (bigger, slower). On
  Windows / macOS the f16 path is selected and is faster.
* Loading is the bottleneck, not synthesis. Start `ZeroVox.create()` early —
  the [demo](./examples/browser) begins loading as soon as an engine is
  picked. Tuning work is tracked in
  [#31](https://github.com/m96-chan/zerovox/issues/31).
* Keep inference off the UI thread with `WorkerSynthesisEngine` (below); model
  loading blocks whichever thread it runs on.

---

## Roadmap

Shipped:

* ✅ Browser-only inference, voice cloning, TypeScript SDK
* ✅ Streaming synthesis (`stream()`) and gapless playback (`play()`)
* ✅ Voice management + IndexedDB persistence, rendered-audio cache
* ✅ Japanese reading conversion, bracket-aware sentence segmentation
* ✅ Off-thread inference (`WorkerSynthesisEngine`)

Next:

* Multilingual speech — blocked on upstream CFG support
  ([#25](https://github.com/m96-chan/zerovox/issues/25))
* Faster model load ([#31](https://github.com/m96-chan/zerovox/issues/31))
* Multiple model support, emotion control, speech-to-speech

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
