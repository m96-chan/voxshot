import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Reader for the goldens `dump_golden.py` writes.
 *
 * They are not in git — raw floats say nothing in a diff and they are
 * reproducible — so everything here has to fail loudly when they are absent
 * rather than letting a suite go green over nothing. Forked from
 * `spike/miocodec/golden.ts`; only the index/manifest types differ, because
 * the LM dump records a different config and per-case greedy metadata.
 */

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "golden");

export interface TensorMeta {
  shape: number[];
  dtype: "float32";
  bytes: number;
  sha256: string;
}

export interface CaseManifest {
  name: string;
  content: string;
  prompt_tokens: number;
  tensors: Record<string, TensorMeta>;
  greedy_generate_matches_manual?: boolean;
  greedy_generate_len?: number;
  greedy_full_len?: number;
  speech_token_count?: number;
  eos_reached?: boolean;
}

export interface GoldenIndex {
  repo_id: string;
  seed: number;
  torch: string;
  transformers: string;
  checkpoint: { file: string; bytes: number; sha256: string; header_bytes: number };
  config: {
    num_layers: number;
    hidden: number;
    heads: number;
    kv_heads: number;
    head_dim: number;
    ffn: number;
    vocab: number;
    rope_theta: number;
    rms_eps: number;
    tie_word_embeddings: boolean;
    eos_ids: number[];
    speech_token_base: number;
  };
  cases: Record<string, CaseManifest>;
}

/**
 * The message a missing golden produces.
 *
 * Spelled out because the alternative — tests that skip themselves when the
 * fixtures are absent — is how a suite ends up reporting success for having
 * checked nothing. Whoever hits this needs the command, not a diagnosis.
 */
function missing(what: string): never {
  throw new Error(
    `${what} is missing. The goldens are deliberately not in git; rebuild them with\n` +
      `  cd spike/miotts && python3 dump_golden.py`,
  );
}

export function loadIndex(): GoldenIndex {
  try {
    return JSON.parse(readFileSync(join(ROOT, "index.json"), "utf8")) as GoldenIndex;
  } catch {
    missing("golden/index.json");
  }
}

export class GoldenCase {
  constructor(
    readonly manifest: CaseManifest,
    private readonly directory: string,
  ) {}

  /**
   * One tensor, as raw f32 with its shape.
   *
   * The sha256 in the manifest is checked rather than trusted. A truncated or
   * half-written dump reads back as a shorter array of perfectly plausible
   * floats, and the first thing anyone would blame is the port.
   */
  tensor(name: string): { data: Float32Array; shape: number[] } {
    const meta = this.manifest.tensors[name];
    if (!meta) {
      throw new Error(
        `no tensor "${name}" in case "${this.manifest.name}"; have ` +
          Object.keys(this.manifest.tensors).join(", "),
      );
    }
    let bytes: Buffer;
    try {
      bytes = readFileSync(join(this.directory, `${name}.f32`));
    } catch {
      missing(`golden/${this.manifest.name}/${name}.f32`);
    }
    const digest = createHash("sha256").update(bytes).digest("hex");
    if (digest !== meta.sha256) {
      throw new Error(
        `golden/${this.manifest.name}/${name}.f32 does not match its manifest ` +
          `(sha256 ${digest.slice(0, 12)} vs ${meta.sha256.slice(0, 12)}). ` +
          `Regenerate rather than guessing which one is stale.`,
      );
    }
    const data = new Float32Array(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    );
    return { data, shape: meta.shape };
  }
}

export function loadCase(name: string): GoldenCase {
  const index = loadIndex();
  const manifest = index.cases[name];
  if (!manifest) {
    throw new Error(`no case "${name}"; have ${Object.keys(index.cases).join(", ")}`);
  }
  return new GoldenCase(manifest, join(ROOT, name));
}

/** Worst absolute and relative disagreement, and where. */
export function worstDifference(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
): { index: number; actual: number; expected: number; abs: number; rel: number } {
  if (actual.length !== expected.length) {
    throw new Error(`length ${actual.length} against ${expected.length}`);
  }
  let worst = { index: -1, actual: 0, expected: 0, abs: 0, rel: 0 };
  let scale = 0;
  for (let i = 0; i < expected.length; i += 1) scale = Math.max(scale, Math.abs(expected[i]!));
  for (let i = 0; i < actual.length; i += 1) {
    const abs = Math.abs(actual[i]! - expected[i]!);
    if (abs > worst.abs) {
      // Relative to the signal's own scale rather than to the element: hidden
      // states cross zero constantly, and dividing by an element that happens
      // to sit near zero reports a vast relative error for a negligible
      // difference.
      worst = { index: i, actual: actual[i]!, expected: expected[i]!, abs, rel: abs / scale };
    }
  }
  return worst;
}
