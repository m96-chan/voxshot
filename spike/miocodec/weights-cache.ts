import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { Weights } from "./decoder.js";
import { Safetensors } from "./safetensors.js";

/**
 * The checkpoint, from wherever `huggingface_hub` put it.
 *
 * Node-only, and only for the tests — the browser fetches from the CDN instead.
 * Shared between the CPU and GPU suites so both read the same file rather than
 * each having its own idea of where it is.
 */
export function loadWeights(repoId: string): Weights {
  const hub = join(homedir(), ".cache", "huggingface", "hub");
  const repo = `models--${repoId.replace("/", "--")}`;
  let file: string;
  try {
    const revision = readFileSync(join(hub, repo, "refs", "main"), "utf8").trim();
    file = join(hub, repo, "snapshots", revision, "model.safetensors");
  } catch {
    throw new Error(
      `${repoId}'s checkpoint is not in the HF cache. It arrives with the golden:\n` +
        `  cd spike/miocodec && .venv/bin/python dump_golden.py`,
    );
  }
  const bytes = readFileSync(file);
  return new Weights(
    Safetensors.parse(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer,
    ),
  );
}
