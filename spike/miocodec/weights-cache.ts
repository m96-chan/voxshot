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
/** The two source-checkpoint digests `export_encoder_weights.py` records. */
interface EncoderWeightsManifest {
  sources?: Record<string, { sha256?: string } | undefined>;
}

/** The corresponding digests `dump_encoder_golden.py` records. */
interface EncoderGoldenIndex {
  checkpoint?: { sha256?: string };
  ssl_checkpoint?: { sha256?: string };
}

/**
 * The consume-time half of the export's sha cross-check: the export diffs the
 * source checkpoints against the golden when it RUNS, but a weights file
 * exported before either checkpoint moved stays stale on disk — and stale
 * weights make every stage comparison fail as a phantom kernel bug. Throws,
 * with the fix, when the recorded shas disagree.
 */
export function checkEncoderProvenance(
  manifest: EncoderWeightsManifest,
  index: EncoderGoldenIndex,
): void {
  const pairs = [
    ["miocodec", index.checkpoint],
    ["wavlm", index.ssl_checkpoint],
  ] as const;
  for (const [name, recorded] of pairs) {
    const source = manifest.sources?.[name];
    if (!source?.sha256 || !recorded?.sha256 || source.sha256 !== recorded.sha256) {
      throw new Error(
        `encoder weights are stale: the ${name} checkpoint sha256 recorded in ` +
          `encoder-weights.json (${source?.sha256?.slice(0, 12) ?? "absent"}) does not match ` +
          `golden-encoder/index.json's (${recorded?.sha256?.slice(0, 12) ?? "absent"}) — ` +
          `weights and golden come from different checkpoints, and every stage comparison ` +
          `would chase a phantom. Re-export:\n` +
          `  cd spike/miocodec && .venv/bin/python export_encoder_weights.py`,
      );
    }
  }
}

/**
 * The encoder's weights, from the file `export_encoder_weights.py` writes.
 *
 * One local file rather than two checkpoints: the export folds pos_conv's
 * weight-norm and precomputes the resample kernel, neither of which this port
 * should reimplement just to read a checkpoint. It sits beside the golden and
 * is gitignored with it, so the missing-file message names the command.
 *
 * The recorded source-checkpoint shas are cross-checked against the golden's
 * index before the file is trusted — see {@link checkEncoderProvenance}.
 */
export function loadEncoderWeights(): Weights {
  const root = join(dirname(fileURLToPath(import.meta.url)), "golden-encoder");
  const path = join(root, "encoder-weights.safetensors");
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

  // Cross-check the export's recorded shas against the golden's, and skip
  // SILENTLY only when the golden index itself is absent: with no golden
  // there is nothing to compare against, the weights are still internally
  // consistent, and every consumer that needs the golden already fails loudly
  // with the dump command — a second error here would just shadow that one.
  let index: EncoderGoldenIndex | null = null;
  try {
    index = JSON.parse(readFileSync(join(root, "index.json"), "utf8")) as EncoderGoldenIndex;
  } catch {
    index = null;
  }
  if (index) {
    let manifest: EncoderWeightsManifest;
    try {
      manifest = JSON.parse(
        readFileSync(join(root, "encoder-weights.json"), "utf8"),
      ) as EncoderWeightsManifest;
    } catch {
      throw new Error(
        `${join(root, "encoder-weights.json")} is missing beside the weights — a partial export ` +
          `cannot be provenance-checked. Re-export:\n` +
          `  cd spike/miocodec && .venv/bin/python export_encoder_weights.py`,
      );
    }
    checkEncoderProvenance(manifest, index);
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
