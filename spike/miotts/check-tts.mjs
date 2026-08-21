/**
 * Drive `examples/mio-tts.html` — text in, WAV out — and check every stage of
 * what came out against an oracle that shares no code with the page's run path.
 *
 * The page is the deliverable, so "it should work" is not a claim this repo
 * accepts (rule 2). Same philosophy as ../miocodec/check-demo.mjs: the page
 * and the reference are never compared against each other's own output —
 * each side of the pipeline is held against an independent implementation:
 *
 *   - the GPU engine's greedy token ids must EXACTLY equal the CPU q8
 *     oracle's (expected-tokens.ts / model-q8.ts — same quantized weights, no
 *     WGSL). The two do NOT compute the same numbers: the CPU oracle sums
 *     sequentially (f64 accumulator), the WGSL kernels tree-reduce in f32,
 *     and web-xpu-ops documents agreement-not-equality for exactly this
 *     reason. What IS pinned here is the empirical fact that on this model,
 *     this machine and these weights, greedy argmax never lands close enough
 *     to a tie for the summation orders to pick different ids — both runs
 *     are deterministic, so the id streams either match exactly or something
 *     changed. On divergence the dump below must distinguish a near-tie
 *     argmax flip (tiny relGap at the diverging step — summation-order noise,
 *     re-examine this assertion) from a real bug (large relGap — the GPU
 *     computed different math).
 *   - the WAV blob the page hands a listener must match the MioCodec CPU
 *     backend's decode of the same ids, run here in Node. The comparison is
 *     16-bit tolerant (the WAV is int16; ~3e-5 of quantisation on its own),
 *     bound 5e-3 relative to the reference's peak — the same convention
 *     check-demo.mjs uses against torch.
 *
 * Needs a display (DISPLAY=:1) — headless Chromium only reaches SwiftShader
 * on this machine, and a software adapter is an explicit failure here because
 * the printed tok/s and RTF would look exactly like hardware numbers.
 *
 *     cd spike/miotts && npm run check:tts        # = npx tsx check-tts.mjs
 *
 * (tsx, not node: the codec reference is imported straight from
 * ../miocodec/*.ts, the same sources the page's bundle was built from —
 * but note the page runs the codec's GPU backend, so the Node cpuBackend
 * decode below is still an independent implementation of the codec.)
 *
 * The q8 CPU oracle takes ~4 minutes for the golden text; it is spawned once,
 * first, and runs concurrently with everything else.
 */

import { execFile, spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

import { chromium } from "playwright";

import { cpuBackend, decode, MIOCODEC_24K } from "../miocodec/decoder.js";
import { encodeGlobal } from "../miocodec/encoder.js";
import { GoldenCase, worstDifference } from "../miocodec/golden.js";
import { loadEncoderWeights, loadWeights } from "../miocodec/weights-cache.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const BASE = "http://localhost:8082";
const CHECKPOINT_HOST = "https://huggingface.co/Aratako/MioCodec-25Hz-24kHz/**";
// The golden ja text — also the page textarea's prefill (examples/mio-tts.html).
const JA_TEXT = "こんにちは、今日はいい天気ですね";
const EOS_IDS = [151645, 151643];
const SAMPLES_PER_TOKEN = 960; // 24 kHz / 25 tokens per second
const SOFTWARE = /swiftshader|llvmpipe|software|lavapipe/i;

const failures = [];
function check(ok, message) {
  if (!ok) failures.push(message);
  return ok;
}

/* -------------------------------------------------------------------------- *
 * The CPU q8 oracle, spawned once and awaited late (it dominates wall time)
 * -------------------------------------------------------------------------- */

const tsx = join(HERE, "node_modules", ".bin", "tsx");
const oracleExec = promisify(execFile)(tsx, ["expected-tokens.ts", JA_TEXT], {
  cwd: HERE,
  maxBuffer: 64 * 1024 * 1024,
});
const oraclePromise = oracleExec.then(({ stdout, stderr }) => {
  if (stderr.trim()) console.log(`oracle stderr   ${stderr.trim().split("\n").at(-1)}`);
  // stdout is exactly one JSON line: {promptIds, ids, speech, eosReached}.
  return JSON.parse(stdout);
});
oraclePromise.catch(() => {}); // awaited later; don't let an early failure crash unhandled
process.on("exit", () => oracleExec.child.kill()); // don't orphan a 4-minute child on early exit
console.log(`oracle          npx tsx expected-tokens.ts "${JA_TEXT}" (running, ~4 min)`);

/* -------------------------------------------------------------------------- *
 * The server — serve.mjs, reused as-is (spawned unless one is already up)
 * -------------------------------------------------------------------------- */

async function serverUp() {
  try {
    const response = await fetch(`${BASE}/mio-tts.html`, { method: "HEAD" });
    return response.ok;
  } catch {
    return false;
  }
}

let serverChild = null;
if (await serverUp()) {
  console.log(`server          already listening on ${BASE}, reusing it`);
} else {
  serverChild = spawn(process.execPath, [join(HERE, "serve.mjs")], { stdio: "ignore" });
  const deadline = Date.now() + 10_000;
  while (!(await serverUp())) {
    if (Date.now() > deadline) throw new Error("serve.mjs did not come up on :8082 within 10 s");
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  console.log(`server          spawned serve.mjs on ${BASE}`);
}
process.on("exit", () => serverChild?.kill());

/* -------------------------------------------------------------------------- *
 * The browser — headed Chromium with the flags that reach the hardware
 * (measured in ../miocodec/check-demo.mjs; headless only finds SwiftShader)
 * -------------------------------------------------------------------------- */

const browser = await chromium.launch({
  headless: !process.env.DISPLAY,
  args: [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan",
    "--use-angle=vulkan",
    "--ignore-gpu-blocklist",
    "--enable-gpu",
    "--disable-gpu-sandbox",
    "--no-sandbox",
  ],
});
const page = await browser.newPage();

const problems = [];
page.on("console", (message) => {
  if (message.type() === "error") problems.push(`console: ${message.text()}`);
});
page.on("pageerror", (error) => problems.push(`pageerror: ${error.message}`));

// The MioCodec checkpoint request starts at the canonical huggingface.co URL;
// route it to serve.mjs's HF-cache copy (check-demo.mjs's 302 trick — half a
// gigabyte per run from the CDN would make this something nobody runs).
await page.route(CHECKPOINT_HOST, (route) =>
  route.fulfill({ status: 302, headers: { location: `${BASE}/model.safetensors` } }),
);

/** Click #run, wait for the metrics, return the page's E2E hook result. */
async function runOnce(timeoutMs) {
  await page.click("#run");
  await page.waitForSelector("#metrics:not(.hidden)", { timeout: timeoutMs });
  return await page.evaluate(() => window.__result);
}

/** The 16-bit WAV the page hands a listener, read back off the blob URL. */
async function readWav() {
  return await page.evaluate(async () => {
    const src = document.getElementById("player").src;
    const response = await fetch(src);
    const bytes = new Uint8Array(await response.arrayBuffer());
    return {
      src: src.slice(0, 5),
      bytes: bytes.length,
      riff: String.fromCharCode(...bytes.slice(0, 4)),
      wave: String.fromCharCode(...bytes.slice(8, 12)),
      pcm: Array.from(new Int16Array(bytes.buffer.slice(44))),
    };
  });
}

await page.goto(`${BASE}/mio-tts.html`);

// --- Greedy run: fill the golden ja text explicitly (the page prefills it,
// but this check must not depend on the HTML's prefill staying in sync);
// greedy is the select's default. Cold path (downloads + q8 packing + GPU
// upload) ≈ 4 s locally; the timeout is slack, not an expectation.
await page.fill("#text", JA_TEXT);
const greedy = await runOnce(900_000);
if (!check(greedy?.status === "done", `greedy run: ${JSON.stringify(greedy)}`)) {
  finish();
}
const wav = await readWav();

console.log(`adapter         ${greedy.adapter}`);
check(
  !SOFTWARE.test(greedy.adapter),
  `the WebGPU adapter is a software rasteriser (${greedy.adapter}) — every number below ` +
    "would be meaningless. Run headed: DISPLAY=:1 npm run check:tts",
);

// --- Sampled sanity (reference server's T=0.8, top-p=1.0; fixed seed). Run
// while the oracle still grinds; asserts only shape — listenability is human.
await page.selectOption("#mode", "sample");
await page.fill("#seed", "42");
const sampled = await runOnce(300_000);

/* -------------------------------------------------------------------------- *
 * Voice clone, in the browser: choose jp_ref1.wav as the reference, wait for
 * the page's GPU encode, then synthesize greedy with the cloned voice. The
 * embedding and the browser-resampled input are read back here; the numeric
 * checks against the golden run below, after the browser closes.
 * -------------------------------------------------------------------------- */

// The same clip the encoder golden was dumped from, out of the HF snapshot
// (serve.mjs's resolve, inlined — it is three lines).
const refWavPath = (() => {
  const root = join(homedir(), ".cache", "huggingface", "hub", "models--Aratako--MioTTS-0.6B");
  const revision = readFileSync(join(root, "refs", "main"), "utf8").trim();
  return join(root, "snapshots", revision, "samples", "jp_ref1.wav");
})();

// That path goes through a MUTABLE `refs/main`, so the file it lands on can
// change without anything here saying so — and every embedding comparison
// below would then fail as a phantom kernel or resampler bug. The golden
// records the sha256 of the wav it was actually built from; hash the one being
// fed to the page and stop right here if they differ, with the cause named.
const GOLDEN_ENCODER = join(HERE, "../miocodec/golden-encoder");
const encoderIndex = JSON.parse(readFileSync(join(GOLDEN_ENCODER, "index.json"), "utf8"));
{
  const recorded = encoderIndex.cases?.jp_ref1?.source_sha256;
  const actual = createHash("sha256").update(readFileSync(refWavPath)).digest("hex");
  if (!recorded) {
    throw new Error(
      `golden-encoder/index.json records no source_sha256 for jp_ref1 — the golden predates the ` +
        `pin. Regenerate it:\n  cd spike/miocodec && .venv/bin/python dump_encoder_golden.py`,
    );
  }
  if (actual !== recorded) {
    throw new Error(
      `THE REFERENCE WAV DRIFTED — not a kernel or resampler problem.\n` +
        `  ${refWavPath}\n` +
        `  hashes to ${actual.slice(0, 12)}, but golden-encoder/ was dumped from ` +
        `${recorded.slice(0, 12)}.\n` +
        `refs/main moved under us, so the embedding comparisons below would be measuring two ` +
        `different clips against each other. Regenerate the golden from the new file:\n` +
        `  cd spike/miocodec && .venv/bin/python dump_encoder_golden.py`,
    );
  }
  console.log(`ref wav         jp_ref1.wav sha256 ${actual.slice(0, 12)} matches the golden's`);
}

await page.selectOption("#voice", "reference");
await page.setInputFiles("#refaudio", refWavPath);
await page.waitForFunction(() => window.__voice !== null && window.__voice.status !== "encoding", {
  timeout: 300_000,
});
const voice = await page.evaluate(() => window.__voice);
let cloned = null;
let clonedWav = null;
let voiceWave = null;
if (check(voice?.status === "ready", `voice encode: ${JSON.stringify(voice)}`)) {
  // What the encoder actually consumed — Chrome's resample of the 44.1 kHz
  // clip — for the same-input oracle below.
  voiceWave = await page.evaluate(() => window.__voiceWave());

  // Greedy again, now with the cloned voice (mode was left on "sample" above).
  await page.selectOption("#mode", "greedy");
  cloned = await runOnce(300_000);
  if (check(cloned?.status === "done", `cloned-voice run: ${JSON.stringify(cloned)}`)) {
    clonedWav = await readWav();
  }
}

await browser.close();

/* -------------------------------------------------------------------------- *
 * Check 1 — GREEDY IDS: exactly the CPU q8 oracle's
 * -------------------------------------------------------------------------- */

const oracle = await oraclePromise;

const idsEqual = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

console.log(`prompt ids      ${greedy.promptIds.length} (oracle ${oracle.promptIds.length})`);
console.log(`generated ids   ${greedy.generatedIds.length} (oracle ${oracle.ids.length}, ${greedy.speechIndices.length} speech)`);
check(idsEqual(greedy.promptIds, oracle.promptIds), "prompt ids differ from the oracle's");
if (!check(idsEqual(greedy.generatedIds, oracle.ids), "generated ids differ from the CPU q8 oracle's")) {
  const n = Math.min(greedy.generatedIds.length, oracle.ids.length);
  let first = -1;
  for (let i = 0; i < n && first < 0; i += 1) if (greedy.generatedIds[i] !== oracle.ids[i]) first = i;
  console.log(`  first divergence at step ${first}: gpu ${greedy.generatedIds[first]} vs oracle ${oracle.ids[first]}`);
  console.log(`  gpu    ${JSON.stringify(greedy.generatedIds)}`);
  console.log(`  oracle ${JSON.stringify(oracle.ids)}`);
  if (first >= 0) {
    // Near-tie or real bug? Rerun the oracle with --dump-margins (failure
    // path only — the success path never pays this second ~4 min run) and
    // show the CPU argmax margin at the diverging step: a relGap near f32
    // rounding noise means the two summation orders picked different sides
    // of a tie; a large relGap means the GPU computed genuinely different math.
    console.log("  rerunning the oracle with --dump-margins for the divergence step (~4 min)...");
    try {
      const { stdout } = await promisify(execFile)(
        tsx,
        ["expected-tokens.ts", JA_TEXT, "--dump-margins"],
        { cwd: HERE, maxBuffer: 64 * 1024 * 1024 },
      );
      const margin = JSON.parse(stdout).margins?.[first];
      if (margin) {
        console.log(
          `  cpu oracle margin at step ${first}: top1 ${margin.top1}, top2 ${margin.top2}, ` +
            `relGap ${margin.relGap.toExponential(2)} ` +
            `(tiny relGap = near-tie argmax flip; large = real bug)`,
        );
      }
    } catch (error) {
      console.log(`  (margin rerun failed: ${error?.message ?? error})`);
    }
  }
}
check(oracle.eosReached, "the oracle itself did not reach eos — the comparison is against a truncated run");
check(EOS_IDS.includes(greedy.generatedIds.at(-1)), "greedy: last generated id is not an eos");
check(greedy.nonSpeechIds.length === 0, `greedy: non-speech ids ${JSON.stringify(greedy.nonSpeechIds)}`);

/* -------------------------------------------------------------------------- *
 * Check 2 — the WAV: real RIFF, right length, and equal to the codec CPU
 * backend's decode of the same ids (run right here, in Node)
 * -------------------------------------------------------------------------- */

check(wav.src === "blob:", `player src is ${wav.src}, not a blob`);
check(wav.riff === "RIFF" && wav.wave === "WAVE", `not a WAV (${wav.riff}/${wav.wave})`);
const expectedSamples = SAMPLES_PER_TOKEN * greedy.speechIndices.length;
check(
  wav.pcm.length === expectedSamples,
  `WAV has ${wav.pcm.length} samples, expected ${SAMPLES_PER_TOKEN} * ${greedy.speechIndices.length} = ${expectedSamples}`,
);

console.log("codec reference decoding on the Node cpuBackend...");
const fixture = JSON.parse(readFileSync(join(HERE, "../../examples/mio-codec-fixture.json"), "utf8"));
const codecWeights = loadWeights("Aratako/MioCodec-25Hz-24kHz");
const { waveform: reference } = await decode(
  Float32Array.from(greedy.speechIndices),
  Float32Array.from(fixture.global_embedding),
  2 * greedy.speechIndices.length, // the aligned path: 2 STFT frames per 25 Hz token
  MIOCODEC_24K,
  codecWeights,
  cpuBackend,
);

let peak = 0;
for (const v of reference) peak = Math.max(peak, Math.abs(v));
let worstAbs = 0;
for (let i = 0; i < Math.min(wav.pcm.length, reference.length); i += 1) {
  worstAbs = Math.max(worstAbs, Math.abs(wav.pcm[i] / 0x8000 - reference[i]));
}
const worstRel = worstAbs / peak;
console.log(`wav             ${wav.riff}/${wav.wave}, ${wav.bytes.toLocaleString()} B, ${wav.pcm.length.toLocaleString()} samples, peak ${peak.toFixed(2)}`);
console.log(`vs cpu codec    abs ${worstAbs.toExponential(2)}  rel ${worstRel.toExponential(2)}  (bound 5e-3)`);
check(reference.length === wav.pcm.length, `cpu codec produced ${reference.length} samples vs the page's ${wav.pcm.length}`);
check(worstRel < 5e-3, `page WAV vs cpu codec decode: worst rel ${worstRel.toExponential(2)} >= 5e-3`);

/* -------------------------------------------------------------------------- *
 * Check 3 — sampled mode: right shape, length recorded (nothing more; the
 * ids are seeded-random and their sound is for a human to judge)
 * -------------------------------------------------------------------------- */

if (check(sampled?.status === "done", `sampled run: ${JSON.stringify(sampled)}`)) {
  check(EOS_IDS.includes(sampled.generatedIds.at(-1)), "sampled: last generated id is not an eos");
  check(sampled.nonSpeechIds.length === 0, `sampled: non-speech ids ${JSON.stringify(sampled.nonSpeechIds)}`);
  check(sampled.speechIndices.length > 0, "sampled: no speech tokens");
  console.log(
    `sampled (s=42)  ${sampled.generatedIds.length} ids, ${sampled.speechIndices.length} speech, ` +
      `${sampled.audioSeconds.toFixed(2)} s audio, ${sampled.lmTokensPerSec.toFixed(1)} tok/s`,
  );
  // The number ISSUE #120 turned on: sampler.ts serves each draw from a
  // top-2048 window and only falls back to upstream's full-vocabulary sort
  // (~38 ms) when the draw lands past it. Before the fix EVERY step paid that,
  // which is why sampled mode ran at ~25 tok/s while greedy ran at ~209. If
  // this ratio ever climbs back towards 1, the tok/s above will follow it down.
  console.log(
    `sampler         ${sampled.samplerFallbacks}/${sampled.generatedIds.length} steps fell back to the full-vocabulary sort`,
  );
}

/* -------------------------------------------------------------------------- *
 * Check 4 — voice clone: the page's GPU embedding against the encoder golden,
 * then the cloned-voice synthesis
 * -------------------------------------------------------------------------- */

// Bounds are ~5x over what this machine measures (printed alongside):
//
//   page GPU vs golden           5.97e-4 measured -> bound 3e-3
//   page GPU vs CPU same-input   8.05e-7 measured -> bound 5e-6
//
// The two comparisons answer different questions, and the SAME-INPUT one is
// authoritative for the port: it runs the reference encoder on the very
// samples the page encoded, so the resampler cancels and only the WGSL
// conv/matmul arithmetic is on trial — 8e-7 is f32 summation-order noise.
// The golden comparison additionally swallows the input skew (Chrome's
// WebAudio resampler vs torchaudio's polyphase disagree by a measured
// **4.4e-2** of the waveform's peak), which the encoder's pooled statistics
// attenuate by ~two orders of magnitude. Its looser bound guards the whole
// path end to end; if it ever fails alone, suspect a resampler change
// (Chrome update), not the kernels.
const EMBED_GOLDEN_BOUND = 3e-3;
const EMBED_SAME_INPUT_BOUND = 5e-6;

/**
 * Worst relative disagreement, via ../miocodec/golden.js's `worstDifference`
 * (one metric, one definition of "relative to the signal's own peak", shared
 * with the encoder suite). That one throws on a length mismatch; here a
 * mismatch is a reportable outcome rather than a crash, so it becomes Infinity
 * and the caller prints the two lengths.
 */
const worstRelOf = (actual, expected) => {
  if (actual.length !== expected.length) return Infinity;
  return worstDifference(actual, expected).rel;
};

// The golden's tensors through `GoldenCase`, so every read is sha256-checked
// against the manifest exactly as the encoder suite's are — a truncated or
// half-written dump reads back as plausible floats otherwise, and the port is
// the first thing anyone would blame. The per-case manifest lives in
// index.json (read above for the wav pin) and the layout is the decoder
// golden's, so no adaptation beyond pointing it at golden-encoder/jp_ref1.
const jpRef1 = new GoldenCase(encoderIndex.cases.jp_ref1, join(GOLDEN_ENCODER, "jp_ref1"));

// The embedding accuracy checks depend ONLY on the captured voice and
// voiceWave, so they run whenever the encode reached "ready" — a failed
// cloned-voice SYNTHESIS still fails the script (checked at the run above)
// but must not hide the encoder's numbers, which are what these bounds exist
// to watch.
if (voice?.status === "ready") {
  console.log(`\nvoice encode    ${voice.encodeMs.toFixed(0)} ms (GPU, jp_ref1.wav — paid once per file)`);

  check(voice.embedding.length === 128, `embedding has ${voice.embedding.length} dims, expected 128`);

  // The input skew, named before the embedding is judged: the page resampled
  // 44.1 kHz -> 24 kHz with Chrome's WebAudio resampler, the golden's input
  // came through torchaudio's polyphase. Same length, different arithmetic.
  const goldenWave = jpRef1.tensor("waveform_24k").data;
  const waveRel = worstRelOf(voiceWave, goldenWave);
  console.log(
    `resample skew   browser wave vs golden wave: worst rel ${waveRel === Infinity ? `length ${voiceWave.length} vs ${goldenWave.length}` : waveRel.toExponential(2)}`,
  );

  // End to end: page GPU embedding vs the torch golden — kernels AND
  // resampler together, at the looser bound (see the block above for why,
  // and which of the two comparisons is authoritative).
  const goldenEmbedding = jpRef1.tensor("global_embedding").data;
  const embedRelGolden = worstRelOf(voice.embedding, goldenEmbedding);
  console.log(`embedding       page GPU vs golden: worst rel ${embedRelGolden.toExponential(2)}  (bound ${EMBED_GOLDEN_BOUND.toExponential(0)})`);
  check(
    embedRelGolden < EMBED_GOLDEN_BOUND,
    `page embedding vs golden: worst rel ${embedRelGolden.toExponential(2)} >= ${EMBED_GOLDEN_BOUND}`,
  );

  // The authoritative, same-input oracle: the reference (CPU) encoder on the
  // SAME browser-resampled samples. The resampler cancels; only the WGSL
  // kernels are on trial, at the tight bound. ~40 s of reference conv, the
  // check's second-largest cost — paid because it is the one comparison a
  // Chrome resampler change cannot move.
  console.log("encoder reference (CPU) on the browser-resampled wave...");
  const encoderWeights = loadEncoderWeights();
  const sameInput = await encodeGlobal(Float32Array.from(voiceWave), encoderWeights);
  const embedRelSame = worstRelOf(voice.embedding, sameInput);
  console.log(`embedding       page GPU vs CPU same-input: worst rel ${embedRelSame.toExponential(2)}  (bound ${EMBED_SAME_INPUT_BOUND.toExponential(0)})`);
  check(
    embedRelSame < EMBED_SAME_INPUT_BOUND,
    `page embedding vs same-input reference: worst rel ${embedRelSame.toExponential(2)} >= ${EMBED_SAME_INPUT_BOUND}`,
  );
}

if (voice?.status === "ready" && cloned?.status === "done" && clonedWav) {
  // The cloned synthesis. The LM never sees the voice — identity enters only
  // at the decoder — so greedy ids must EQUAL the default-voice greedy ids.
  check(cloned.voice === "reference", `cloned run reports voice "${cloned.voice}", expected "reference"`);
  check(
    Array.isArray(cloned.embedding) && idsEqual(cloned.embedding, voice.embedding),
    "the cloned run's embedding is not the one the encode produced",
  );
  check(
    idsEqual(cloned.generatedIds, greedy.generatedIds),
    "cloned-voice greedy ids differ from the default voice's — the LM must not see the voice",
  );
  check(
    clonedWav.riff === "RIFF" && clonedWav.wave === "WAVE",
    `cloned WAV is not a WAV (${clonedWav.riff}/${clonedWav.wave})`,
  );
  const clonedExpected = SAMPLES_PER_TOKEN * cloned.speechIndices.length;
  check(
    clonedWav.pcm.length === clonedExpected,
    `cloned WAV has ${clonedWav.pcm.length} samples, expected ${clonedExpected}`,
  );
  console.log(
    `cloned voice    ${cloned.speechIndices.length} speech tokens (ids = default greedy), ` +
      `${clonedWav.pcm.length.toLocaleString()} samples, decode ${(cloned.decodeMs / 1000).toFixed(2)} s`,
  );
}

/* -------------------------------------------------------------------------- *
 * The numbers (greedy run, hardware adapter — asserted above)
 * -------------------------------------------------------------------------- */

console.log(`\nprefill         ${greedy.prefillMs.toFixed(0)} ms / ${greedy.promptIds.length} prompt tokens (KV-only steps skip the lm_head)`);
console.log(`lm              ${greedy.lmMs.toFixed(0)} ms / ${greedy.lmSteps} steps = ${greedy.lmTokensPerSec.toFixed(1)} tok/s`);
console.log(`codec decode    ${(greedy.decodeMs / 1000).toFixed(2)} s`);
console.log(`total e2e       ${(greedy.totalMs / 1000).toFixed(2)} s for ${greedy.audioSeconds.toFixed(2)} s of audio`);
console.log(`rtf             ${greedy.rtf.toFixed(2)}  (processing / audio, same convention as ../miocodec)`);
console.log(`engine          ${JSON.stringify(greedy.engineStats)}`);

finish();

function finish() {
  for (const problem of problems) failures.push(problem);
  if (failures.length) {
    console.error("\nFAILED:");
    for (const failure of failures) console.error(`  ${failure}`);
    process.exit(1);
  }
  console.log("\nok");
  process.exit(0);
}
