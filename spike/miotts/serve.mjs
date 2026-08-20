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
 *   /miotts/encoder-weights.safetensors
 *                           -> ../miocodec/golden-encoder/encoder-weights.safetensors
 *                              (export_encoder_weights.py's artifact — the voice-clone encoder)
 *   /samples/*.wav          -> the MioTTS-0.6B HF cache snapshot's samples/
 *                              (reference clips for the voice-clone UI and its check)
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

import { createHash } from "node:crypto";
import { createReadStream, readFileSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { homedir } from "node:os";
import { extname, join, normalize } from "node:path";

const EXAMPLES = new URL("../../examples/", import.meta.url).pathname;
const Q8_DIR = new URL("./q8/", import.meta.url).pathname;
const ENCODER_WEIGHTS = new URL(
  "../miocodec/golden-encoder/encoder-weights.safetensors",
  import.meta.url,
).pathname;
const HUB = join(homedir(), ".cache", "huggingface", "hub");
const PORT = 8082;

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".bin": "application/octet-stream",
  ".safetensors": "application/octet-stream",
  ".wav": "audio/wav",
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
  if (name === "miotts/encoder-weights.safetensors") {
    return ENCODER_WEIGHTS;
  }
  if (name.startsWith("samples/")) {
    return hubFile("models--Aratako--MioTTS-0.6B", name);
  }
  if (name.startsWith("miotts/q8/")) {
    return join(Q8_DIR, name.slice("miotts/q8/".length));
  }
  return join(EXAMPLES, name || "mio-tts.html");
}

// Startup gate: the tokenizer.json this server hands the page (resolved via
// refs/main, which can silently move) must be byte-identical to the one the
// golden vectors — and therefore the TS tokenizer port — were verified
// against. Serving a drifted file would break the page in ways no check
// attributes to the tokenizer.
{
  let pinned;
  try {
    pinned = JSON.parse(
      readFileSync(new URL("./golden/tokenizer_vectors.json", import.meta.url), "utf8"),
    ).tokenizer_sha256;
  } catch {
    /* handled below */
  }
  if (!pinned) {
    console.error(
      "golden/tokenizer_vectors.json (with tokenizer_sha256) is missing — rebuild it with\n" +
        "  cd spike/miotts && python3 dump_tokenizer_vectors.py",
    );
    process.exit(1);
  }
  const served = resolve("/miotts/tokenizer.json");
  const actual = createHash("sha256").update(readFileSync(served)).digest("hex");
  if (actual !== pinned) {
    console.error(
      `tokenizer.json drift: ${served}\n` +
        `hashes to ${actual}, but the golden vectors were dumped from ${pinned}.\n` +
        "refs/main moved under us — re-verify the tokenizer port against the new file and rerun\n" +
        "  cd spike/miotts && python3 dump_tokenizer_vectors.py",
    );
    process.exit(1);
  }
}

// Same policy for the encoder weights: fail at startup with the command,
// rather than letting the page's first voice-clone attempt 404 into a message
// that names neither the file nor how to build it.
try {
  statSync(ENCODER_WEIGHTS);
} catch {
  console.error(
    `${ENCODER_WEIGHTS} is missing — the voice-clone encoder cannot be served. It is exported\n` +
      "from the two source checkpoints, not in git:\n" +
      "  cd spike/miocodec && .venv/bin/python export_encoder_weights.py\n" +
      "(dump the golden first if golden-encoder/ is empty: .venv/bin/python dump_encoder_golden.py)",
  );
  process.exit(1);
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
