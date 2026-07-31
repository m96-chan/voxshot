# Patches

Two patches that make the **Chatterbox multilingual** checkpoint produce
intelligible speech today, instead of waiting for an upstream release.

Take them if you need them. Nothing in this repository applies them for you.

| patch | what it does |
| --- | --- |
| `@huggingface+transformers+4.2.0.patch` | Backports [huggingface/transformers.js#1705](https://github.com/huggingface/transformers.js/pull/1705) — classifier-free guidance for Chatterbox, plus `MinPLogitsWarper` |
| `voxshot+0.3.0.patch` | Adds a `guidanceScale` option to `ChatterboxEngine` and forwards it to `generate` |

## Why

The multilingual checkpoint is generated with classifier-free guidance in the
reference implementation (`resemble-ai/chatterbox`, `T3.inference`, always
`cfg_weight=0.5`). Transformers.js has not shipped it. Without it the model
loads and runs and produces sound, so nothing looks broken — it simply rambles.

Measured on the assembled checkpoint with one Japanese sentence,
`こんにちは。今日はいい天気ですね。`:

| | `guidance_scale` unset | `guidance_scale: 1.5` |
| --- | --- | --- |
| Audio length | **16.08 s** | **2.52 s** |
| `dims` | `[1, 385920]` | `[1, 60480]` |
| RMS | 0.0401 | 0.0912 |
| NaN | 0 | 0 |

Sixteen seconds of output for a sentence that takes two and a half. `dims[0]`
stays 1, so the two-row guidance batch is recombined rather than leaking into
the result.

Numbers cannot tell you the speech is *intelligible*. They can tell you the
failure mode is gone. Listen before you trust it.

## Applying them

```bash
npm install --save-dev patch-package
mkdir -p patches && cp path/to/*.patch patches/
npx patch-package
```

Add `"postinstall": "patch-package"` to `package.json` if you want them
reapplied after every install.

Then ask for guidance:

```ts
const engine = new ChatterboxEngine({
  modelId: "chatterbox-multilingual",
  guidanceScale: 1.5, // = 1 + cfg_weight, matching the reference default
});
```

`guidanceScale` defaults to 1, which forwards nothing — an unpatched
Transformers.js keeps behaving exactly as before rather than tripping over an
option it does not know.

## Which patch applies where

The transformers patch applies anywhere `@huggingface/transformers@4.2.0` is a
real dependency, **including this example**.

The voxshot patch targets `voxshot@0.3.0` **as installed from npm**. It does
*not* apply inside this example, which links the library with `file:../..` —
there is no `node_modules/voxshot` for patch-package to find. To try the
multilingual engine from this example, edit the library source directly.

## How the transformers patch was built

Not by hand-editing bundles. Upstream was cloned at the `4.2.0` tag and rebuilt
**unmodified** first — the result matches the published `dist/` byte for byte.
Applying #1705 to the source and rebuilding therefore yields a diff containing
only that PR's changes, with no porting step to get wrong.

```bash
git clone --depth 1 --branch 4.2.0 https://github.com/huggingface/transformers.js
cd transformers.js && pnpm install --frozen-lockfile --filter ./packages/transformers...
cd packages/transformers && node scripts/build.mjs      # matches published dist
gh api repos/huggingface/transformers.js/pulls/1705 \
  -H "Accept: application/vnd.github.v3.diff" | git apply -p3 --directory=packages/transformers
node scripts/build.mjs && pnpm typegen
```

It covers the three bundles `exports` actually resolves —
`transformers.web.js`, `transformers.node.mjs`, `transformers.node.cjs` — plus
the `src/` and `types/` files the PR touches. The minified bundles are
deliberately excluded: they are unreachable through `exports`, and including
them took the patch from 40 kB to 4.2 MB, because their lines run to 175,000
characters.

The PR's own tests pass (6), but note that its three new ones cover
`MinPLogitsWarper` only. The guidance path itself is covered by the measurement
above and nothing else.

## Delete these when

`@huggingface/transformers` ships a release containing #1705, and `voxshot`
ships `guidanceScale` as a real option. Both patches pin an exact version, so
they will refuse to apply to anything else rather than silently rotting —
`patch-package` fails loudly on a version mismatch.

Tracked in
[#103](https://github.com/m96-chan/voxshot/issues/103) and
[#25](https://github.com/m96-chan/voxshot/issues/25).
