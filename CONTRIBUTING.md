# Contributing to VoxShot

Thanks for your interest. This project has a few rules that are stricter than
average, and they are easier to follow if you know about them before you write
code rather than after review. This document is the short version;
[`CLAUDE.md`](CLAUDE.md) is the full set and is the authority if the two ever
disagree.

## Workflow

**An issue comes first.** Every change starts from an issue describing the
background, what is in scope, what is deliberately out of scope, and what
"done" means. This is not ceremony — writing the scope down is usually where a
change turns out to be bigger, smaller or unnecessary.

One issue maps to one branch and one PR. Put `Closes #<n>` in the PR body so the
merge closes the issue; closing issues by hand tends to close things that were
never actually implemented.

If the work changes shape as you go, update the issue or open a new one. An
issue that no longer describes the change is worse than no issue.

## Tests come first

Write the failing test, then the implementation. Do not commit code that has no
test.

Coverage must stay at or above **90%** for statements, branches, functions and
lines. The thresholds live in `vitest.config.ts` and fail the build, so this is
enforced rather than aspirational.

Coverage is a floor, not a goal. Tests that exist only to reach a number are
worse than the gap they fill — assert behaviour a consumer could observe, and if
a branch is genuinely unreachable, say so in the code rather than writing a test
that pretends otherwise.

**Watch each new test fail before you make it pass**, individually and for the
reason it is meant to catch. A failure count is not enough: if you add five
tests and three fail, the two that passed are the interesting ones — a test that
never reproduces its condition is green against the unfixed code too.

**When fixing a bug, disable the fix and confirm the new test goes red**, then
restore it. Without that step you can only claim a test was added, not that it
guards anything. And if a test still passes with the code under test removed,
the observable is wrong rather than the code.

Coverage will not catch any of this. The lines run either way.

## Read before you write

Do not infer an API's behaviour and build on the guess. Read the existing
implementation, the type definitions, the library's own source, or run it and
look at the output.

This matters more than usual here because the library sits on
`@huggingface/transformers` and `onnxruntime-web`, where the documented surface
and the actual behaviour diverge in places that only show up at runtime with a
1.5 GB model. Several fixes in this repository came from reading the vendored
Transformers.js source; if you find yourself writing "this should…", go and
check.

## Commit messages

English, imperative present-tense subject (`Add …`, `Fix …`).

Explain **why** in the body. What changed is already in the diff; what a future
reader cannot recover is the reason it was worth changing, and what was
considered and rejected.

Commits already merged into `main` are never rewritten.

## Changelog

If your change is observable to a consumer — new API, breaking change,
behaviour change, bug fix — add an entry under `Unreleased` in
[`CHANGELOG.md`](CHANGELOG.md) in the same PR.

Internal refactors and test-only or docs-only changes are exempt.

## Commands

```bash
npm install          # install dependencies
npm test             # run tests with the 90% coverage thresholds
npm run test:watch   # watch mode
npm run typecheck    # type check
npm run build        # build to dist/
```

The browser demo is a separate workspace with its own suite:

```bash
cd examples/browser
npm install
npm run dev          # serve the demo
npm run typecheck
npm test
```

CI runs `typecheck`, `test` and `build` on Node 20 and 22, then the
`examples/browser` checks. All of it must pass.

## Architecture constraints

**No direct dependency on browser APIs.** `AudioContext`, `navigator.gpu` and
`indexedDB` are reached through interfaces in [`src/platform.ts`](src/platform.ts)
and injected, so tests can substitute them. Reaching for a global directly will
be sent back in review.

**Inference engines sit behind `SynthesisEngine`.** Nothing above that interface
should know which model is running. Swapping in a different backend means
implementing the interface and nothing else.

**Side-effect-free logic is extracted.** Text processing and audio conversion
are pure functions with their own unit tests, separate from anything that
touches the platform.

## Releases

1. Bump `version` in `package.json` and move `Unreleased` in `CHANGELOG.md` to
   the new version heading.
2. Merge that to `main`.
3. Create a GitHub Release tagged `vX.Y.Z`.

Publishing is automated: the Release triggers
[`.github/workflows/publish.yml`](.github/workflows/publish.yml), which verifies
that the tag matches `package.json` before publishing to npm with provenance. A
mismatch fails the run rather than shipping a package whose version disagrees
with its tag.

While the version is below `1.0.0`, breaking changes go in minor releases.

## Questions

If something here is unclear or seems wrong, open an issue. A rule nobody can
follow is a bug in the rule.
