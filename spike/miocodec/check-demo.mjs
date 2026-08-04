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

const browser = await chromium.launch();
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

await page.goto("http://localhost:8081/mio-tts.html");
await page.click("#run");
// The reference implementations decode six seconds of audio in about thirty;
// five minutes is slack for a slower machine, not an expectation.
await page.waitForSelector("#metrics:not(.hidden)", { timeout: 300_000 });

const result = await page.evaluate(() => ({
  status: document.getElementById("status").textContent,
  audio: document.getElementById("m-audio").textContent,
  elapsed: document.getElementById("m-elapsed").textContent,
  rtf: document.getElementById("m-rtf").textContent,
  tokens: document.getElementById("m-tokens").textContent,
  src: document.getElementById("player").src.slice(0, 5),
}));

// The player having a blob: URL is the difference between "the numbers rendered"
// and "audio actually came out".
const played = await page.evaluate(async () => {
  const response = await fetch(document.getElementById("player").src);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const riff = String.fromCharCode(...bytes.slice(0, 4));
  const wave = String.fromCharCode(...bytes.slice(8, 12));
  return { bytes: bytes.length, riff, wave };
});

await browser.close();
server.close();

console.log(`checkpoint    ${local} (${size.toLocaleString()} B, routed from disk)`);
console.log(`status        ${result.status}`);
console.log(`audio         ${result.audio}`);
console.log(`decode        ${result.elapsed}   RTF ${result.rtf}`);
console.log(`tokens        ${result.tokens}`);
console.log(`player src    ${result.src}:  WAV ${played.riff}/${played.wave}, ${played.bytes.toLocaleString()} B`);

const failures = [
  ...problems,
  result.src === "blob:" ? null : `player src is ${result.src}, expected a blob`,
  played.riff === "RIFF" && played.wave === "WAVE" ? null : `not a WAV: ${played.riff}/${played.wave}`,
  played.bytes > 100_000 ? null : `WAV is ${played.bytes} bytes, too short to be six seconds`,
].filter(Boolean);

if (failures.length) {
  console.error("\nFAILED:");
  for (const failure of failures) console.error(`  ${failure}`);
  process.exit(1);
}
console.log("\nok");
