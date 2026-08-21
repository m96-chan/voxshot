import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Reader for the goldens `dump_golden.py` writes.
 *
 * They are not in git — see README.md — so everything here has to fail loudly
 * when they are absent rather than letting a suite go green over nothing.
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
  num_tokens: number;
  audio_length: number;
  stft_length: number;
  interpolate_is_identity: boolean;
  tensors: Record<string, TensorMeta>;
}

export interface GoldenIndex {
  repo_id: string;
  seed: number;
  torch: string;
  codebook_size: number;
  config: {
    sample_rate: number;
    n_fft: number;
    hop_length: number;
    istft_padding: string;
    wave_decoder_dim: number;
    wave_resnet_num_groups: number;
    wave_interpolation_mode: string;
    wave_upsampler_factors: number[] | null;
  };
  checkpoint: { file: string; bytes: number; sha256: string };
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
      `  cd spike/miocodec && .venv/bin/python dump_golden.py\n` +
      `See spike/miocodec/README.md for how the venv is put together.`,
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

/**
 * `[bins, frames]` (torch's layout, and the golden's) to `[frames, bins]`.
 *
 * `ops/stft` is frame-major by a documented decision — a vocoder head emits one
 * row per frame — and the reference implementation this golden comes from is
 * bin-major. Neither is wrong and the transpose has to happen somewhere; doing
 * it here keeps it out of every test.
 */
export function toFrameMajor(data: Float32Array, bins: number, frames: number): Float32Array {
  const out = new Float32Array(frames * bins);
  for (let frame = 0; frame < frames; frame += 1) {
    for (let bin = 0; bin < bins; bin += 1) {
      out[frame * bins + bin] = data[bin * frames + frame]!;
    }
  }
  return out;
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
      // Relative to the signal's own scale rather than to the element: a
      // waveform crosses zero constantly, and dividing by a sample that happens
      // to sit near one reports a vast relative error for an inaudible
      // difference.
      worst = { index: i, actual: actual[i]!, expected: expected[i]!, abs, rel: abs / scale };
    }
  }
  return worst;
}
