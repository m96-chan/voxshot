/**
 * Drive `examples/mio-tts.html` in a real browser and report what it produced.
 *
 * The page is the deliverable, so "it should work" is not a claim this repo
 * accepts (rule 2). This opens it in headless Chromium, clicks the button,
 * waits for the decode and reads the numbers back out of the DOM.
 *
 *     node check-demo.mjs
 *
 * The checkpoint request is **routed to the local Hugging Face cache** rather
 * than fetched. Half a gigabyte per run would make this something nobody runs,
 * and the network path is verified separately and more cheaply: the CDN answers
 * 200 with `access-control-allow-origin: *` and a content-length of
 * 523,087,956, which is the size in the golden's manifest. What routing cannot
 * check is that URL; what it does check is everything after it — streaming,
 * progress, safetensors parsing, the decode, and the WAV that comes out.
 */

import { createReadStream, readFileSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { homedir } from "node:os";
import { extname, join, normalize } from "node:path";
import { chromium } from "playwright";

const ROOT = new URL("../../examples/", import.meta.url).pathname;
const HUB = join(homedir(), ".cache", "huggingface", "hub", "models--Aratako--MioCodec-25Hz-24kHz");
const CHECKPOINT_HOST = "https://huggingface.co/Aratako/MioCodec-25Hz-24kHz/**";

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
};

function checkpointPath() {
  const revision = readFileSync(join(HUB, "refs", "main"), "utf8").trim();
  return join(HUB, "snapshots", revision, "model.safetensors");
}

const server = createServer((request, response) => {
  const name = normalize(decodeURIComponent(new URL(request.url, "http://x").pathname)).replace(/^\/+/, "");
  // The checkpoint is streamed from the HF cache rather than read into memory:
  // `route.fulfill` cannot carry half a gigabyte (Playwright stringifies the
  // body and Node caps a string at 0x1fffffe8), so the route below answers with
  // a redirect here instead.
  const file = name === "model.safetensors" ? checkpointPath() : join(ROOT, name || "mio-tts.html");
  try {
    statSync(file);
  } catch {
    response.writeHead(404).end("not found");
    return;
  }
  response.writeHead(200, {
    "content-type": TYPES[extname(file)] ?? "application/octet-stream",
    // Needed even though this is the page's own origin: the checkpoint request
    // *starts* at huggingface.co, so the redirect lands here still tainted as
    // cross-origin and the browser demands the header on the final response.
    // Measured — without it Chromium refuses with "No 'Access-Control-Allow-
    // Origin' header is present" and the page reports "Failed to fetch".
    "access-control-allow-origin": "*",
    "content-length": String(statSync(file).size),
  });
  createReadStream(file).pipe(response);
});

await new Promise((resolve) => server.listen(8081, resolve));

// Headless Chromium ships WebGPU behind flags and with no GPU process by
// default; without these the page's `navigator.gpu` is absent and `auto` would
// quietly pick the CPU path — which is exactly the silent downgrade this check
// exists to catch.
// Headed, when there is a display to be headed on.
//
// Measured on this machine: headless Chromium reports its WebGPU adapter as
// "google swiftshader" whatever combination of --enable-gpu,
// --ignore-gpu-blocklist and Vulkan feature flags it is given. SwiftShader
// answers every call correctly and says nothing whatsoever about speed, so a
// run against it verifies the kernels and must not be quoted as a GPU number.
// With a display, Dawn reaches the real adapter.
const headless = !process.env.DISPLAY;
const browser = await chromium.launch({
  headless,
  args: [
    "--enable-unsafe-webgpu",
    // Dawn talks to Vulkan directly; ANGLE is for GL and gets in the way here.
    "--enable-features=Vulkan,VulkanFromANGLE",
    "--ignore-gpu-blocklist",
    "--enable-gpu",
    // Headless Chromium disables the GPU process outright unless told not to,
    // and then reports a WebGPU adapter backed by SwiftShader — a software
    // rasteriser that answers every call correctly and says nothing useful
    // about speed. Measured: without these the adapter came back as
    // "google swiftshader".
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

// The checkpoint, from disk, with the headers the page's streaming reader needs.
const local = checkpointPath();
const size = statSync(local).size;
await page.route(CHECKPOINT_HOST, async (route) => {
  await route.fulfill({ status: 302, headers: { location: "http://localhost:8081/model.safetensors" } });
});

/**
 * The reference decode of the same tokens, from `dump_demo_fixture.py`.
 *
 * This is what makes the run a check rather than a demonstration: both browser
 * backends are compared against torch's own output for the fixture, not against
 * each other. Two implementations agreeing says nothing if they agree on the
 * wrong answer.
 */
function referenceWaveform() {
  const path = new URL("./golden/demo_reference.f32", import.meta.url).pathname;
  try {
    const bytes = readFileSync(path);
    return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  } catch {
    throw new Error(
      `${path} is missing. It is written alongside the fixture:\n` +
        `  cd spike/miocodec && .venv/bin/python dump_demo_fixture.py`,
    );
  }
}

/** Worst disagreement, relative to the reference's own peak. */
function worstRelative(actual, expected) {
  let peak = 0;
  for (let i = 0; i < expected.length; i += 1) peak = Math.max(peak, Math.abs(expected[i]));
  let worst = 0;
  for (let i = 0; i < Math.min(actual.length, expected.length); i += 1) {
    worst = Math.max(worst, Math.abs(actual[i] - expected[i]));
  }
  return { abs: worst, rel: worst / peak, lengths: [actual.length, expected.length] };
}

async function decodeWith(backend) {
  await page.goto(`http://localhost:8081/mio-tts.html?backend=${backend}`);
  await page.click("#run");
  // The reference implementations take about half a minute for six seconds of
  // audio; five is slack for a slower machine, not an expectation.
  await page.waitForSelector("#metrics:not(.hidden)", { timeout: 300_000 });

  const metrics = await page.evaluate(() => ({
    status: document.getElementById("status").textContent,
    backend: document.getElementById("m-backend").textContent,
    audio: document.getElementById("m-audio").textContent,
    elapsed: document.getElementById("m-elapsed").textContent,
    rtf: document.getElementById("m-rtf").textContent,
    tokens: document.getElementById("m-tokens").textContent,
    src: document.getElementById("player").src.slice(0, 5),
  }));

  // The 16-bit WAV the page hands a listener, read back and unpacked. Checking
  // the blob rather than an internal buffer is deliberate: it is the difference
  // between "the numbers rendered" and "audio actually came out".
  const wav = await page.evaluate(async () => {
    const response = await fetch(document.getElementById("player").src);
    const bytes = new Uint8Array(await response.arrayBuffer());
    return {
      bytes: bytes.length,
      riff: String.fromCharCode(...bytes.slice(0, 4)),
      wave: String.fromCharCode(...bytes.slice(8, 12)),
      pcm: Array.from(new Int16Array(bytes.buffer.slice(44))),
    };
  });

  return { metrics, wav };
}

const reference = referenceWaveform();
const runs = {};
for (const backend of ["cpu", "auto"]) {
  runs[backend] = await decodeWith(backend);
}

await browser.close();
server.close();

console.log(`checkpoint    ${local} (${size.toLocaleString()} B, routed from disk)`);
console.log(`reference     ${reference.length.toLocaleString()} samples from the torch decode\n`);

const failures = [...problems];
for (const [requested, { metrics, wav }] of Object.entries(runs)) {
  // The WAV is 16-bit, so the comparison is against what a listener gets rather
  // than against the f32 the decoder produced. That costs about 3e-5 of
  // quantisation on its own, which is why the bound below is not tighter.
  const pcm = Float32Array.from(wav.pcm, (v) => v / 0x8000);
  const worst = worstRelative(pcm, reference);
  console.log(`[?backend=${requested}] -> ${metrics.backend}`);
  console.log(`  audio       ${metrics.audio}`);
  console.log(`  decode      ${metrics.elapsed}   RTF ${metrics.rtf}`);
  console.log(`  wav         ${wav.riff}/${wav.wave}, ${wav.bytes.toLocaleString()} B`);
  console.log(`  vs torch    abs ${worst.abs.toExponential(2)}  rel ${worst.rel.toExponential(2)}`);

  if (metrics.src !== "blob:") failures.push(`${requested}: player src is ${metrics.src}`);
  if (wav.riff !== "RIFF" || wav.wave !== "WAVE") failures.push(`${requested}: not a WAV`);
  if (pcm.length !== reference.length) {
    failures.push(`${requested}: ${pcm.length} samples against the reference's ${reference.length}`);
  }
  if (!(worst.rel < 5e-3)) failures.push(`${requested}: worst ${worst.rel.toExponential(2)} vs torch`);
}

if (runs.auto.metrics.backend === runs.cpu.metrics.backend) {
  failures.push(`auto chose ${runs.auto.metrics.backend}; WebGPU did not run, so nothing was checked`);
}

// A software adapter runs the same WGSL and gets the same answers, so it checks
// the kernels — and says nothing at all about speed. Called out loudly because
// the RTF printed above looks exactly like a hardware number and is not one.
const SOFTWARE = /swiftshader|llvmpipe|software|lavapipe/i;
if (SOFTWARE.test(runs.auto.metrics.backend)) {
  console.log(
    `\nNOTE: the WebGPU adapter is a software rasteriser (${runs.auto.metrics.backend}).\n` +
      "      Correctness is checked; the RTF above is not a hardware measurement.",
  );
}

if (failures.length) {
  console.error("\nFAILED:");
  for (const failure of failures) console.error(`  ${failure}`);
  process.exit(1);
}
console.log("\nok");
