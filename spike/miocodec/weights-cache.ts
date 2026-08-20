import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { Weights } from "./decoder.js";
import { Safetensors } from "./safetensors.js";

/**
 * The checkpoint, from wherever `huggingface_hub` put it.
 *
 * Node-only, and only for the tests — the browser fetches from the CDN instead.
 * Shared between the CPU and GPU suites so both read the same file rather than
 * each having its own idea of where it is.
 */
/**
 * The encoder's weights, from the file `export_encoder_weights.py` writes.
 *
 * One local file rather than two checkpoints: the export folds pos_conv's
 * weight-norm and precomputes the resample kernel, neither of which this port
 * should reimplement just to read a checkpoint. It sits beside the golden and
 * is gitignored with it, so the missing-file message names the command.
 */
export function loadEncoderWeights(): Weights {
  const path = join(
    dirname(fileURLToPath(import.meta.url)),
    "golden-encoder",
    "encoder-weights.safetensors",
  );
  let bytes: Buffer;
  try {
    bytes = readFileSync(path);
  } catch {
    throw new Error(
      `${path} is missing. It is exported from the two source checkpoints, not in git:\n` +
        `  cd spike/miocodec && .venv/bin/python export_encoder_weights.py\n` +
        `(dump the golden first if golden-encoder/ is empty: .venv/bin/python dump_encoder_golden.py)`,
    );
  }
  return new Weights(
    Safetensors.parse(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer,
    ),
  );
}

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
