import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { loadWeightsQ8, type Qwen3WeightsQ8, type Sha256Fn, type WeightsQ8Manifest } from "../../src/engine/miotts/lm/weights-q8.js";

/**
 * Node-only feeding of `loadWeightsQ8`: file reads and the `node:crypto`
 * hasher live here so `weights-q8.ts` stays importable from a browser bundle.
 */

export const sha256Hex: Sha256Fn = (bytes) => createHash("sha256").update(bytes).digest("hex");

function readBin(dir: string, name: string): ArrayBuffer {
  let raw: Buffer;
  try {
    raw = readFileSync(join(dir, name));
  } catch {
    throw new Error(
      `${join(dir, name)} is missing. The q8 artifacts are deliberately not in git; rebuild with\n` +
        `  cd spike/miotts && python3 convert_weights.py`,
    );
  }
  // Copy to a tight ArrayBuffer: a Buffer is a view into a pooled allocation,
  // and loadWeightsQ8 indexes the buffer from byte 0. (An explicit copy rather
  // than buffer.slice(), whose type is ArrayBuffer | SharedArrayBuffer.)
  const out = new ArrayBuffer(raw.byteLength);
  new Uint8Array(out).set(raw);
  return out;
}

export function loadWeightsQ8FromDir(dir: string): Qwen3WeightsQ8 {
  let manifest: WeightsQ8Manifest;
  try {
    manifest = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8")) as WeightsQ8Manifest;
  } catch {
    throw new Error(
      `${join(dir, "manifest.json")} is missing or unreadable. Rebuild the q8 artifacts with\n` +
        `  cd spike/miotts && python3 convert_weights.py`,
    );
  }
  return loadWeightsQ8({
    manifest,
    codes: readBin(dir, manifest.files.codes.name),
    scales: readBin(dir, manifest.files.scales.name),
    norms: readBin(dir, manifest.files.norms.name),
    sha256: sha256Hex,
  });
}
