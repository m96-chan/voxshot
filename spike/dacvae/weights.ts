/**
 * The DACVAE decoder's weights, read out of the reference checkpoint.
 *
 * `weights.pth` is a torch pickle, which nothing here is going to parse — the
 * dump script writes a flat `.f32` per tensor plus an index, the same shape as
 * `spike/miocodec`'s goldens, and this reads that.
 *
 * ## weight_norm is folded here, not at run time
 *
 * Every convolution in the checkpoint stores `weight_g` and `weight_v` rather
 * than `weight`: `torch.nn.utils.weight_norm` reparameterises a filter as a
 * direction and a magnitude, `w = g * v / ||v||`, where the norm is taken over
 * every axis but the output channel. That is a *training* device. At inference
 * it is a constant, so materialising it once at load costs nothing per call and
 * keeps the run path down to the three ops this port actually needs.
 *
 * The axis is worth stating plainly, because it is easy to encode the wrong
 * rule. `weight_norm` normalises over **all dims except dim 0** of the stored
 * tensor, whatever that tensor means — so one loop is correct for both layouts.
 * What differs is only the interpretation: dim 0 is `Cout` for a Conv1d
 * (`[Cout, Cin, K]`) and `Cin` for a ConvTranspose1d (`[Cin, Cout, K]`). A port
 * that "helpfully" normalised per output channel in both cases would be wrong
 * for every upsampling stage, and wrong in a way that still produces numbers.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "weights");

export interface TensorMeta {
  shape: number[];
  dtype: "float32";
  bytes: number;
  sha256: string;
}

export interface WeightIndex {
  repo_id: string;
  config: {
    sample_rate: number;
    hop_length: number;
    latent_dim: number;
    decoder_dim: number;
    decoder_rates: number[];
    encoder_dim: number;
    encoder_rates: number[];
  };
  tensors: Record<string, TensorMeta>;
}

const REBUILD = "cd spike/dacvae && .venv/bin/python dump_weights.py";

function missing(what: string): never {
  throw new Error(
    `${what} is missing. The decoder weights are deliberately not in git — they are\n` +
      `~260 MB and regenerable. Rebuild them with\n  ${REBUILD}`,
  );
}

let cachedIndex: WeightIndex | null = null;

export function weightIndex(): WeightIndex {
  if (cachedIndex) return cachedIndex;
  try {
    cachedIndex = JSON.parse(readFileSync(join(ROOT, "index.json"), "utf8")) as WeightIndex;
  } catch {
    missing("weights/index.json");
  }
  return cachedIndex;
}

/** One tensor, flat, in the checkpoint's own layout. */
export function tensor(name: string): Float32Array {
  const meta = weightIndex().tensors[name];
  if (!meta) throw new Error(`no tensor "${name}" in the weight index`);
  let bytes: Buffer;
  try {
    bytes = readFileSync(join(ROOT, `${name}.f32`));
  } catch {
    missing(`weights/${name}.f32`);
  }
  if (bytes.byteLength !== meta.bytes) {
    throw new Error(`${name}: ${bytes.byteLength} bytes, index says ${meta.bytes}`);
  }
  // Copied out at its own window: a Buffer is a view over a pooled ArrayBuffer
  // shared with unrelated reads.
  return new Float32Array(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
}

export function shapeOf(name: string): number[] {
  const meta = weightIndex().tensors[name];
  if (!meta) throw new Error(`no tensor "${name}" in the weight index`);
  return meta.shape;
}

export function has(name: string): boolean {
  return name in weightIndex().tensors;
}

export interface ConvWeights {
  weight: Float32Array;
  bias: Float32Array | undefined;
  /** `[Cout, Cin, K]` for a conv, `[Cin, Cout, K]` for a transposed one. */
  shape: number[];
}

/** A convolution's effective weight, with `weight_norm` folded in. */
/**
 * Folded weights, kept.
 *
 * `weight_norm` folding builds a new `Float32Array` every call, and the GPU
 * engines cache their uploads on array identity — so calling this per request
 * re-uploaded the whole decode path. Measured at **261 MB a request**, which is
 * the decoder's weights exactly, and is how a long-lived server reached 25.7 GB
 * of VRAM.
 */
const FOLDED = new Map<string, ConvWeights>();

export function convWeights(prefix: string): ConvWeights {
  const cached = FOLDED.get(prefix);
  if (cached) return cached;
  const built = foldConvWeights(prefix);
  FOLDED.set(prefix, built);
  return built;
}

function foldConvWeights(prefix: string): ConvWeights {
  if (has(`${prefix}.weight`)) {
    // No weight_norm on this one (the reference uses norm="none" in places).
    return {
      weight: tensor(`${prefix}.weight`),
      bias: has(`${prefix}.bias`) ? tensor(`${prefix}.bias`) : undefined,
      shape: shapeOf(`${prefix}.weight`),
    };
  }

  const v = tensor(`${prefix}.weight_v`);
  const g = tensor(`${prefix}.weight_g`);
  const shape = shapeOf(`${prefix}.weight_v`);
  const [d0, d1, k] = shape as [number, number, number];
  const perSlice = d1 * k;

  // `weight_g` is stored as [d0, 1, 1] — one magnitude per slice of dim 0.
  const weight = new Float32Array(v.length);
  for (let slice = 0; slice < d0; slice += 1) {
    const base = slice * perSlice;
    let sumOfSquares = 0;
    for (let i = 0; i < perSlice; i += 1) {
      const value = v[base + i]!;
      sumOfSquares += value * value;
    }
    const scale = g[slice]! / Math.sqrt(sumOfSquares);
    for (let i = 0; i < perSlice; i += 1) weight[base + i] = v[base + i]! * scale;
  }
  return {
    weight,
    bias: has(`${prefix}.bias`) ? tensor(`${prefix}.bias`) : undefined,
    shape,
  };
}
