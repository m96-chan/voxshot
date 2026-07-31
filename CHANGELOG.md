# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version stays below `1.0.0`, breaking changes ship in minor releases.

## [Unreleased]

### Added

- `requiresGpu` on `SynthesisEngine` and on `ChatterboxEngineOptions`. An
  engine that says so is refused before `load()` on a machine without WebGPU,
  and drops the WASM plan from its fallback chain. The Chatterbox pipeline
  measures ~5.8x slower than real time on CPU — not a degraded experience but a
  broken one, arriving with no error after a ~1.5 GB download. Left off by
  default, because whether CPU is usable depends entirely on the model.
  ([#107])
- `requiresGpu` on `WorkerSynthesisEngineOptions`. The engine that needs the
  GPU lives in the worker and the main thread only sees the proxy, so without
  this the requirement could not be expressed in the deployment shape the
  library is built around. ([#107])
- `loadedDevice` on `SynthesisEngine`, reported by `VoxShot.device`. The getter
  is documented as "the backend that was actually selected" and returned the
  resolved device, so an engine that degraded to WASM internally was still
  reported as running on WebGPU. A degrade is observable now instead of
  silent. ([#107])

### Changed

- `device: "webgpu"` no longer documents itself as "never silently slow". It
  picks what to try; an engine may still degrade internally. Use
  `VoxShot.device` to see where it landed, or `requiresGpu` to refuse. No
  behaviour changed — the promise was never kept. ([#107])


## [0.3.0] - 2026-08-01

### Added

- `signal` on `SpeakOptions`, so `speak()`, `stream()` and `play()` can be
  stopped. Only `play()` could before, which left the two ways of rendering a
  long text with no way out: a paper-sized input is hundreds of chunks and tens
  of minutes of work, and dropping the promise stopped none of it. Since the
  worker runs one call at a time, that abandoned work also blocked everything
  queued behind it. ([#92])
- `signal` on `SpeechPlaybackInput`, which is how `play()` honours the option.
  The subscription belongs to the playback so it can be dropped the moment the
  utterance ends — a page-lifetime signal would otherwise collect a listener per
  utterance, each holding a finished playback. ([#92])
- `expressiveness` on `SpeakOptions` and `SynthesisRequest`, setting how
  animated a single utterance is without rebuilding the engine — which
  previously meant reloading the model. `ChatterboxEngine` maps it onto its
  `exaggeration` control and falls back to the value it was constructed with.
  It also takes part in the synthesis cache key, so the same line at two
  settings is rendered twice rather than served from the first. ([#69])

- `SynthesisRequest.signal`, so a render in progress can be abandoned.
  `SpeechPlayback.stop()` now uses it to cancel the lookahead chunk it walks
  away from, instead of leaving it running. ([#67])

### Fixed
- Loads no longer probe a language model that is never fetched. Transformers.js
  reads `dtype` by **session key** when it builds the list of files a load will
  touch, and Chatterbox maps `model` to the file `language_model` — so a record
  keyed only by file name missed, and the expected-file list fell through to the
  device default: `onnx/language_model.onnx` (fp32, 2.08 GB) on WebGPU and
  `onnx/language_model_quantized.onnx` (a 404) on WASM. Both keys are carried
  now. The fp32 file was never downloaded but was seeded into `progress_total`,
  which is what made the download denominator roughly twice the real transfer.
  ([#62])
- The published package now contains its license. `package.json` declared MIT
  while no license text was shipped or even present in the repository, so what
  reached npm granted nothing on its own terms. ([#98])
- Source maps resolve. `src` is now part of the package, so the `../../src/…`
  paths every `.js.map` and `.d.ts.map` already carried point at real files:
  stack traces land on TypeScript and go-to-definition reaches the source
  rather than stopping at the declaration. Previously 125 kB of the 331 kB
  tarball was maps that resolved to nothing. ([#98])
- `ChatterboxEngine` rejects `stallTimeoutMs` and `maxNewTokens` values that
  used to do the opposite of what they read like. `stallTimeoutMs: Infinity`
  spelled "wait forever" and failed the load in about a millisecond, because
  `setTimeout` coerces the delay to 1; it now means the same as the documented
  `0`. A value between 0 and 1 is rejected rather than rounded up into a guard
  that fires before anything can happen. `maxNewTokens: 0` reached the model as
  a cap of nothing, producing no audio and reporting every call as truncated.
  ([#81])
- A budget the engine derives for itself is now capped. `synthesize` is public
  and bounds nothing, so a long chunk could ask for minutes of audio in one
  call — and the upstream KV-cache growth scales with the budget. A cap the
  caller names is still honoured as written. ([#81])
- `AutoConfig` is optional on `TransformersModule`, like the two criteria
  classes beside it. `loadModule` is public API, so a custom module written
  before that member existed no longer dies with an unwrapped `TypeError`. A
  config that is absent, unreadable or hung now degrades to the load layout
  used before it existed rather than failing the load. ([#81])
- The config fetch runs inside the stall guard. It is the load's first network
  request and used to run before the guard armed, so a hung `config.json` left
  `load()` pending with no timer to end it. ([#81])
- A load declared stalled stops emitting. The abandoned transfer kept arming
  timers nothing would clear, and kept forwarding download progress to a
  consumer that had already been handed its `LoadStalledError`. ([#81])
- `synthesize` releases its abort listener after every render, not only a
  cancelled one. One signal covers a whole utterance, so a 100-chunk render
  left 100 listeners on it, each holding a stopping criterion. ([#81])
- An already-aborted `synthesize` renders nothing. The check ran only after
  generation, so an abandoned request still paid for preprocessing, a forward
  pass and the full vocoder pass. Empty text is still answered first. ([#81])

- A load that is abandoned, duplicated or disposed underneath no longer leaves
  the engine half-loaded or leaves ONNX sessions with nothing to release them.
  The lifetime is one state value rather than four fields assigned at different
  moments, so a load lands whole or not at all — and every failure path,
  including a processor that fails after the model is built, releases what it
  already holds. ([#86])

- Truncation is detected by counting what generation produced rather than by
  inferring it from the waveform's length. The old arithmetic assumed a fixed
  relationship that does not exist — the waveform also depends on the reference
  voice — so it was calibrated to one voice and wrong for any other. ([#90])
- Long chunks are no longer truncated mid-sentence. The chunk budget is
  measured in characters and the generation cap in tokens, and nothing related
  the two: at roughly 2.4 tokens per character the old fixed cap of 256 ran out
  at about 89 characters, inside the 120-character chunks the splitter
  produces. The cap is now sized to each chunk, and generation that stops
  because it ran out of budget reports `synthesize-truncated` instead of simply
  ending the audio. ([#65])

- The engine no longer stops answering after an utterance is cut mid-render.
  Requests to the worker are serialized, so a second call cannot re-enter an
  ONNX session that is not re-entrant, and `cancel` / `dispose` bypass the
  queue so teardown stays reachable behind a render that never finishes.
  ([#67])

### Note

- The worker protocol gained a `cancel` method, but `PROTOCOL_VERSION` is
  unchanged: old worker bundles still serve every other method. A stale worker
  paired with a new main bundle therefore degrades rather than breaks — the
  cancel is answered with an unknown-method error nobody is waiting for, and
  the abandoned render runs to completion. **Rebuild the worker alongside the
  main bundle** for cancellation to take effect.

## [0.2.0] - 2026-07-30

Model loading is the theme of this release: it no longer hangs forever, it
reports what it is doing, and it stops making two doomed requests on every cold
start.

### Breaking

- `TransformersModule` gained a required `AutoConfig` member. The type is
  exported, so anyone supplying their own `loadModule` must add `AutoConfig` to
  their module double. Users of the default loader are unaffected. ([#45])
- `load()` now rejects after five minutes of silence instead of staying pending
  forever. This is a fix, but it is observable: code that previously waited
  indefinitely will now see a `LoadStalledError`. Set `stallTimeoutMs: 0` to
  restore the old behaviour. ([#43])

### Added

- `stallTimeoutMs` on `ChatterboxEngineOptions`, rejecting a load that goes
  quiet. The clock measures *silence* rather than total elapsed time, because
  session creation is legitimately quiet for tens of seconds and a total cap
  would abandon healthy loads. Defaults to five minutes; `0` disables it. ([#43])
- `LoadStalledError` and the `LOAD_STALLED` error code. Branch on `code` rather
  than `instanceof`: across a Web Worker boundary the error is rebuilt on the
  main thread. ([#43])
- Load milestones delivered through `onProgress` alongside the usual file
  progress: `load-start`, `load-fallback` (with the reason) and `load-ready`.
  The engine degrades `q4f16` → `q4` → WASM, and these events are the only way
  to tell which plan won, or that a fallback happened at all. ([#44])
- `ChatterboxLifecycleEvent` and `ChatterboxLoadEvent` types for those
  milestones. ([#44])

### Fixed

- Two 404s and a fallback warning on every cold load. Transformers.js registers
  Chatterbox under the class name `ChatterboxModel`, but the model repo's
  `config.json` declares only `model_type: "chatterbox"` and no `architectures`,
  so the lookup missed and fell back to a single-file layout that does not
  exist. The engine now names the architecture. This also repairs the download
  denominator, which previously held `config.json` alone — the reason aggregate
  progress reached 97% while whole files were still unstarted. ([#45])

### Demo and docs

- The browser demo degrades gracefully when `CacheStorage` is unavailable
  instead of failing the entire load. ([#40])
- The deployed demo artifact carries a `CNAME`. ([#46])
- Commit messages are written in English. ([#51])

## [0.1.0] - 2026-07-29

Initial release: the core library plus a real Chatterbox ONNX engine.

### Added

- `VoxShot` facade with voice cloning and speech synthesis
- `ChatterboxEngine` backed by Chatterbox ONNX through Transformers.js v4,
  with per-session dtype plans and WebGPU/WASM device resolution
- `WorkerSynthesisEngine` and `exposeEngine`, so inference can run off the main
  thread
- Gapless streaming playback, one-chunk prefetch and a synthesis cache
- Japanese reading conversion and bracket-aware sentence splitting
- Browser demo under `examples/browser`
- CI and npm publish workflows

[Unreleased]: https://github.com/m96-chan/voxshot/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/m96-chan/voxshot/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/m96-chan/voxshot/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/m96-chan/voxshot/releases/tag/v0.1.0
[#40]: https://github.com/m96-chan/voxshot/issues/40
[#43]: https://github.com/m96-chan/voxshot/issues/43
[#44]: https://github.com/m96-chan/voxshot/issues/44
[#45]: https://github.com/m96-chan/voxshot/issues/45
[#46]: https://github.com/m96-chan/voxshot/issues/46
[#51]: https://github.com/m96-chan/voxshot/issues/51
[#67]: https://github.com/m96-chan/voxshot/issues/67
[#62]: https://github.com/m96-chan/voxshot/issues/62
[#65]: https://github.com/m96-chan/voxshot/issues/65
[#69]: https://github.com/m96-chan/voxshot/issues/69
[#81]: https://github.com/m96-chan/voxshot/issues/81
[#86]: https://github.com/m96-chan/voxshot/issues/86
[#90]: https://github.com/m96-chan/voxshot/issues/90
[#98]: https://github.com/m96-chan/voxshot/issues/98
[#107]: https://github.com/m96-chan/voxshot/issues/107
[#92]: https://github.com/m96-chan/voxshot/issues/92
