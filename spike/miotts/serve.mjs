/**
 * Static server for the MioTTS browser demo.
 *
 *     cd spike/miotts && npm run build   # bundle browser.ts -> examples/mio-tts.js
 *     node serve.mjs                     # then open http://localhost:8082/mio-tts.html
 *
 * Serves `examples/` and maps the model assets the page fetches:
 *
 *   /miotts/q8/*            -> spike/miotts/q8/*            (convert_weights.py's artifacts)
 *   /miotts/tokenizer.json  -> the MioTTS-0.6B HF cache snapshot's tokenizer.json
 *   /model.safetensors      -> the MioCodec HF cache checkpoint
 *
 * The page requests the MioCodec checkpoint from its canonical huggingface.co
 * URL (same as mio-codec.html); a driver script (or a human with devtools
 * routing) redirects that request to /model.safetensors here — the same
 * route/302 trick spike/miocodec/check-demo.mjs uses, and the reason this
 * server answers with `access-control-allow-origin: *`: the request starts
 * cross-origin and stays tainted through the redirect. Without the redirect
 * the page simply downloads the 523 MB from the CDN, which also works.
 *
 * `content-length` is set on every response because the page's streaming
 * progress bars need a total.
 */

import { createReadStream, readFileSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { homedir } from "node:os";
import { extname, join, normalize } from "node:path";

const EXAMPLES = new URL("../../examples/", import.meta.url).pathname;
const Q8_DIR = new URL("./q8/", import.meta.url).pathname;
const HUB = join(homedir(), ".cache", "huggingface", "hub");
const PORT = 8082;

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".bin": "application/octet-stream",
  ".safetensors": "application/octet-stream",
};

/** `refs/main` -> snapshot file, the layout huggingface_hub writes (same resolve as check-demo.mjs). */
function hubFile(repoDir, name) {
  const root = join(HUB, repoDir);
  const revision = readFileSync(join(root, "refs", "main"), "utf8").trim();
  return join(root, "snapshots", revision, name);
}

function resolve(pathname) {
  const name = normalize(decodeURIComponent(pathname)).replace(/^\/+/, "");
  if (name === "model.safetensors") {
    return hubFile("models--Aratako--MioCodec-25Hz-24kHz", "model.safetensors");
  }
  if (name === "miotts/tokenizer.json") {
    return hubFile("models--Aratako--MioTTS-0.6B", "tokenizer.json");
  }
  if (name.startsWith("miotts/q8/")) {
    return join(Q8_DIR, name.slice("miotts/q8/".length));
  }
  return join(EXAMPLES, name || "mio-tts.html");
}

const server = createServer((request, response) => {
  const pathname = new URL(request.url, "http://x").pathname;
  let file;
  let size;
  try {
    file = resolve(pathname);
    size = statSync(file).size;
  } catch {
    response.writeHead(404).end("not found");
    return;
  }
  response.writeHead(200, {
    "content-type": TYPES[extname(file)] ?? "application/octet-stream",
    "content-length": String(size),
    // Needed for the checkpoint redirect case — see the module doc.
    "access-control-allow-origin": "*",
  });
  if (request.method === "HEAD") {
    response.end();
    return;
  }
  createReadStream(file).pipe(response);
});

server.listen(PORT, () => {
  console.log(`serving http://localhost:${PORT}/mio-tts.html`);
  console.log(`  examples/           ${EXAMPLES}`);
  console.log(`  /miotts/q8/*        ${Q8_DIR}`);
  console.log(`  /miotts/tokenizer.json + /model.safetensors from ${HUB}`);
});
