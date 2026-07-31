# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version stays below `1.0.0`, breaking changes ship in minor releases.

## [Unreleased]

### Added

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

[Unreleased]: https://github.com/m96-chan/voxshot/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/m96-chan/voxshot/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/m96-chan/voxshot/releases/tag/v0.1.0
[#40]: https://github.com/m96-chan/voxshot/issues/40
[#43]: https://github.com/m96-chan/voxshot/issues/43
[#44]: https://github.com/m96-chan/voxshot/issues/44
[#45]: https://github.com/m96-chan/voxshot/issues/45
[#46]: https://github.com/m96-chan/voxshot/issues/46
[#51]: https://github.com/m96-chan/voxshot/issues/51
[#67]: https://github.com/m96-chan/voxshot/issues/67
[#69]: https://github.com/m96-chan/voxshot/issues/69
