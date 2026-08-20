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
 *     oracle's (expected-tokens.ts / model-q8.ts — same quantized weights,
 *     scalar f32/f64 arithmetic, no WGSL). Exact, not tolerant: both engines
 *     are deterministic and argmax over the same numbers must agree.
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
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

import { chromium } from "playwright";

import { cpuBackend, decode, MIOCODEC_24K } from "../miocodec/decoder.js";
import { loadWeights } from "../miocodec/weights-cache.js";

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

// --- Greedy run: the textarea prefill IS the golden ja text; greedy is the
// select's default. Cold path (downloads + q8 packing + GPU upload) ≈ 4 s
// locally; the timeout is slack, not an expectation.
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
}

/* -------------------------------------------------------------------------- *
 * The numbers (greedy run, hardware adapter — asserted above)
 * -------------------------------------------------------------------------- */

console.log(`\nlm              ${greedy.lmMs.toFixed(0)} ms / ${greedy.lmSteps} steps = ${greedy.lmTokensPerSec.toFixed(1)} tok/s`);
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
