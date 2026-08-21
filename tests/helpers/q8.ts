import type {
  Q8Config,
  Q8TensorEntry,
  WeightsQ8Manifest,
} from "../../src/engine/miotts/lm/weights-q8.js";

/**
 * A synthetic q8 artifact set — the manifest and three bins `convert_weights.py`
 * would produce, at a size a test can hold.
 *
 * The real ones are 583 MB and not in git, so everything CI checks about the
 * loader and the decode loop is checked against a model built here: two layers,
 * hidden 8, a twelve-token vocabulary. Small enough to write expectations for
 * by hand, and structurally identical to the real thing — same tensor names,
 * same layout, same offsets.
 */

/** A two-layer model whose every dimension is distinct, so a swap shows up. */
export const TINY_CONFIG: Q8Config = {
  numLayers: 2,
  hiddenSize: 8,
  numHeads: 2,
  numKvHeads: 1,
  headDim: 4,
  ffnHidden: 16,
  vocabSize: 12,
  ropeTheta: 10_000,
  rmsNormEps: 1e-6,
  tieEmbeddings: true,
  speechTokenBase: 4,
  eosIds: [1],
};

export interface SyntheticQ8 {
  manifest: WeightsQ8Manifest;
  codes: ArrayBuffer;
  scales: ArrayBuffer;
  norms: ArrayBuffer;
}

/** Deterministic int8 codes in [-127, 127]; a fixed stream keeps failures reproducible. */
function codeStream(count: number, seed: number): Int8Array {
  const out = new Int8Array(count);
  let state = seed >>> 0 || 1;
  for (let i = 0; i < count; i += 1) {
    state = (state * 1_664_525 + 1_013_904_223) >>> 0;
    out[i] = ((state >>> 24) % 255) - 127;
  }
  return out;
}

/** Deterministic f32 values around 1, the shape a norm gamma has. */
function normStream(count: number, seed: number): Float32Array {
  const out = new Float32Array(count);
  let state = seed >>> 0 || 1;
  for (let i = 0; i < count; i += 1) {
    state = (state * 1_103_515_245 + 12_345) >>> 0;
    out[i] = 1 + ((state >>> 16) / 65_536 - 0.5) * 0.2;
  }
  return out;
}

interface Quant {
  name: string;
  rows: number;
  cols: number;
}

/** Every quantized tensor the loader looks for, in the order it writes them. */
function quantLayout(config: Q8Config): Quant[] {
  const { hiddenSize: h, numHeads, numKvHeads, headDim, ffnHidden, vocabSize } = config;
  const quants: Quant[] = [{ name: "embedTokens", rows: vocabSize, cols: h }];
  for (let i = 0; i < config.numLayers; i += 1) {
    quants.push(
      { name: `layers.${i}.wq`, rows: numHeads * headDim, cols: h },
      { name: `layers.${i}.wk`, rows: numKvHeads * headDim, cols: h },
      { name: `layers.${i}.wv`, rows: numKvHeads * headDim, cols: h },
      { name: `layers.${i}.wo`, rows: h, cols: numHeads * headDim },
      { name: `layers.${i}.wGate`, rows: ffnHidden, cols: h },
      { name: `layers.${i}.wUp`, rows: ffnHidden, cols: h },
      { name: `layers.${i}.wDown`, rows: h, cols: ffnHidden },
    );
  }
  return quants;
}

/** Every norm tensor the loader looks for. */
function normLayout(config: Q8Config): { name: string; cols: number }[] {
  const norms = [{ name: "finalNorm", cols: config.hiddenSize }];
  for (let i = 0; i < config.numLayers; i += 1) {
    norms.push(
      { name: `layers.${i}.attnNorm`, cols: config.hiddenSize },
      { name: `layers.${i}.ffnNorm`, cols: config.hiddenSize },
      { name: `layers.${i}.qNorm`, cols: config.headDim },
      { name: `layers.${i}.kNorm`, cols: config.headDim },
      { name: `layers.${i}.qNormRaw`, cols: config.headDim },
      { name: `layers.${i}.kNormRaw`, cols: config.headDim },
    );
  }
  return norms;
}

export function buildSyntheticQ8(
  overrides: { config?: Partial<Q8Config>; seed?: number } = {},
): SyntheticQ8 {
  const config: Q8Config = { ...TINY_CONFIG, ...overrides.config };
  const seed = overrides.seed ?? 1;

  const quants = quantLayout(config);
  const norms = normLayout(config);

  const codesBytes = quants.reduce((sum, q) => sum + q.rows * q.cols, 0);
  const scalesBytes = quants.reduce((sum, q) => sum + q.rows * 4, 0);
  const normsBytes = norms.reduce((sum, n) => sum + n.cols * 4, 0);

  const codes = new Int8Array(codesBytes);
  const scales = new Float32Array(scalesBytes / 4);
  const normData = new Float32Array(normsBytes / 4);

  const tensors: Q8TensorEntry[] = [];
  let codesOffset = 0;
  let scaleOffset = 0;
  quants.forEach((quant, index) => {
    const count = quant.rows * quant.cols;
    codes.set(codeStream(count, seed + index * 31), codesOffset);
    for (let r = 0; r < quant.rows; r += 1) {
      // Distinct per row, so a scale read from the wrong row is visible.
      scales[scaleOffset / 4 + r] = 0.001 * (r + 1) + index * 1e-5;
    }
    tensors.push({
      name: quant.name,
      kind: "quant",
      rows: quant.rows,
      cols: quant.cols,
      codesOffset,
      codesBytes: count,
      scaleOffset,
      scaleBytes: quant.rows * 4,
    });
    codesOffset += count;
    scaleOffset += quant.rows * 4;
  });

  let normOffset = 0;
  norms.forEach((norm, index) => {
    normData.set(normStream(norm.cols, seed + 977 + index * 13), normOffset / 4);
    tensors.push({ name: norm.name, kind: "norm", cols: norm.cols, offset: normOffset, bytes: norm.cols * 4 });
    normOffset += norm.cols * 4;
  });

  const manifest: WeightsQ8Manifest = {
    generatedBy: "tests/helpers/q8.ts",
    source: { path: "synthetic", bytes: 0, sha256: "0".repeat(64) },
    config,
    ropePermuted: true,
    ropePermutedNote: "synthetic artifacts are written in ops/rope channel order by construction",
    files: {
      codes: { name: "weights.codes.bin", bytes: codesBytes, sha256: "c".repeat(64) },
      scales: { name: "weights.scales.bin", bytes: scalesBytes, sha256: "s".repeat(64) },
      norms: { name: "weights.norms.bin", bytes: normsBytes, sha256: "n".repeat(64) },
    },
    tensors,
  };

  return {
    manifest,
    codes: codes.buffer,
    scales: scales.buffer,
    norms: normData.buffer,
  };
}
