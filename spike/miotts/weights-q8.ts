import { packQ8 } from "web-xpu-ops/ops/matvec";

/**
 * Loader for the q8 artifacts `convert_weights.py` writes (manifest.json +
 * weights.codes.bin / weights.scales.bin / weights.norms.bin).
 *
 * Environment-neutral on purpose: this module never touches a file. It takes
 * the manifest already parsed and the three bins as `ArrayBuffer`s, so a
 * browser feeds it `fetch(...).arrayBuffer()` and Node feeds it `readFileSync`
 * (see `weights-q8-node.ts`, which also supplies the `node:crypto` hasher —
 * kept out of here so nothing in this file breaks a browser bundle).
 *
 * Every quantized weight comes out **packed** in `matvecQ8`'s wire format
 * (`ops/matvec/reference.ts#packQ8`: 4 int8 codes per u32, least-significant
 * byte first) because that is the only form the GPU engine keeps resident.
 * Packing copies, so nothing returned here retains a view into the ~580 MB
 * codes buffer — the caller can drop it after loading.
 */

export interface Q8Config {
  numLayers: number;
  hiddenSize: number;
  numHeads: number;
  numKvHeads: number;
  headDim: number;
  ffnHidden: number;
  vocabSize: number;
  ropeTheta: number;
  rmsNormEps: number;
  tieEmbeddings: boolean;
  speechTokenBase: number;
  eosIds: number[];
}

export interface Q8FileMeta {
  name: string;
  bytes: number;
  sha256: string;
}

export interface Q8QuantEntry {
  name: string;
  kind: "quant";
  rows: number;
  cols: number;
  /** Byte offset into weights.codes.bin (1 byte per code). */
  codesOffset: number;
  codesBytes: number;
  /** Byte offset into weights.scales.bin (f32 LE, one per row). */
  scaleOffset: number;
  scaleBytes: number;
}

export interface Q8NormEntry {
  name: string;
  kind: "norm";
  cols: number;
  /** Byte offset into weights.norms.bin (f32 LE). */
  offset: number;
  bytes: number;
}

export type Q8TensorEntry = Q8QuantEntry | Q8NormEntry;

export interface WeightsQ8Manifest {
  generatedBy: string;
  source: { path: string; bytes: number; sha256: string };
  config: Q8Config;
  /**
   * `true` means wq/wk rows and the qNorm/kNorm gammas are already relabeled
   * from HF rotate-half channel order into ops/rope's adjacent-pair order
   * (`llm/weights.ts#permuteRopeChannels`). The loader refuses anything else:
   * an unpermuted checkpoint fed to the ops/rope kernel produces garbage that
   * still looks like numbers.
   */
  ropePermuted: boolean;
  ropePermutedNote: string;
  files: { codes: Q8FileMeta; scales: Q8FileMeta; norms: Q8FileMeta };
  tensors: Q8TensorEntry[];
}

/** A per-row absmax int8 weight in `matvecQ8`'s packed wire format. */
export interface PackedQ8 {
  /** `[rows, ceil(cols/4)]` u32 row-major, 4 sign-extended codes per word, LSB first. */
  packed: Uint32Array;
  /** `[rows]`, one absmax-derived scale per row (`ops/quantize`'s convention). */
  scale: Float32Array;
  rows: number;
  cols: number;
}

export interface Qwen3LayerQ8 {
  attnNorm: Float32Array;
  /** Rows in ops/rope channel order (see `WeightsQ8Manifest.ropePermuted`). */
  wq: PackedQ8;
  /** Rows in ops/rope channel order. */
  wk: PackedQ8;
  wv: PackedQ8;
  wo: PackedQ8;
  ffnNorm: Float32Array;
  wGate: PackedQ8;
  wUp: PackedQ8;
  wDown: PackedQ8;
  /** `[headDim]`, permuted with the same pi as wq's rows — use with ops/rope. */
  qNorm: Float32Array;
  /** `[headDim]`, permuted with the same pi as wk's rows. */
  kNorm: Float32Array;
  /** `[headDim]`, the checkpoint's original order — for rotate-half references. */
  qNormRaw: Float32Array;
  kNormRaw: Float32Array;
}

export interface Qwen3WeightsQ8 {
  config: Q8Config;
  /**
   * `[vocabSize, hiddenSize]`. Tied: serves both as the embedding gather table
   * (`gatherDequantRow`) and as the lm_head matvec matrix (the checkpoint has
   * no lm_head tensor).
   */
  embedTokens: PackedQ8;
  perLayer: Qwen3LayerQ8[];
  finalNorm: Float32Array;
}

/** Hex-encoded SHA-256 of `bytes`. Injectable so browser builds can skip or use SubtleCrypto. */
export type Sha256Fn = (bytes: Uint8Array) => string;

export interface LoadWeightsQ8Input {
  manifest: WeightsQ8Manifest;
  codes: ArrayBuffer;
  scales: ArrayBuffer;
  norms: ArrayBuffer;
  /** When provided, each bin is digested and checked against the manifest. */
  sha256?: Sha256Fn;
}

function checkFile(label: string, meta: Q8FileMeta, buffer: ArrayBuffer, sha256?: Sha256Fn): void {
  if (buffer.byteLength !== meta.bytes) {
    throw new Error(
      `${label} (${meta.name}) is ${buffer.byteLength} bytes, manifest says ${meta.bytes}`,
    );
  }
  if (!sha256) return;
  const digest = sha256(new Uint8Array(buffer));
  if (digest !== meta.sha256) {
    throw new Error(
      `${label} (${meta.name}) sha256 ${digest.slice(0, 12)}... does not match the manifest's ` +
        `${meta.sha256.slice(0, 12)}... — a stale or truncated artifact, regenerate rather than guessing`,
    );
  }
}

export function loadWeightsQ8({ manifest, codes, scales, norms, sha256 }: LoadWeightsQ8Input): Qwen3WeightsQ8 {
  if (manifest.ropePermuted !== true) {
    throw new Error("manifest.ropePermuted is not true; this loader only accepts permuted artifacts");
  }
  checkFile("codes", manifest.files.codes, codes, sha256);
  checkFile("scales", manifest.files.scales, scales, sha256);
  checkFile("norms", manifest.files.norms, norms, sha256);

  const quantByName = new Map<string, Q8QuantEntry>();
  const normByName = new Map<string, Q8NormEntry>();
  for (const entry of manifest.tensors) {
    if (entry.kind === "quant") quantByName.set(entry.name, entry);
    else normByName.set(entry.name, entry);
  }

  const getQuant = (name: string): PackedQ8 => {
    const entry = quantByName.get(name);
    if (!entry) throw new Error(`no quantized tensor named ${JSON.stringify(name)} in manifest`);
    const { rows, cols } = entry;
    if (entry.codesBytes !== rows * cols || entry.scaleBytes !== rows * 4) {
      throw new Error(`${name}: byte counts disagree with rows=${rows} cols=${cols}`);
    }
    const raw = new Int8Array(codes, entry.codesOffset, rows * cols);
    // packQ8's loop only does indexed reads and `& 0xff`, so an Int8Array with
    // the same values packs byte-identically to the Int32Array the signature
    // names — the same cast web-xpu-ops' own packInt8Rows uses (and its
    // weights-q8.test.ts verifies against a real Int32Array).
    const packed = packQ8({ codes: raw as unknown as Int32Array, N: rows, K: cols });
    // slice(): a view would pin the whole scales buffer; the copy is tiny.
    const scale = new Float32Array(scales, entry.scaleOffset, rows).slice();
    return { packed, scale, rows, cols };
  };
  const getNorm = (name: string): Float32Array => {
    const entry = normByName.get(name);
    if (!entry) throw new Error(`no norm tensor named ${JSON.stringify(name)} in manifest`);
    return new Float32Array(norms, entry.offset, entry.cols).slice();
  };

  const perLayer: Qwen3LayerQ8[] = [];
  for (let i = 0; i < manifest.config.numLayers; i += 1) {
    perLayer.push({
      attnNorm: getNorm(`layers.${i}.attnNorm`),
      wq: getQuant(`layers.${i}.wq`),
      wk: getQuant(`layers.${i}.wk`),
      wv: getQuant(`layers.${i}.wv`),
      wo: getQuant(`layers.${i}.wo`),
      ffnNorm: getNorm(`layers.${i}.ffnNorm`),
      wGate: getQuant(`layers.${i}.wGate`),
      wUp: getQuant(`layers.${i}.wUp`),
      wDown: getQuant(`layers.${i}.wDown`),
      qNorm: getNorm(`layers.${i}.qNorm`),
      kNorm: getNorm(`layers.${i}.kNorm`),
      qNormRaw: getNorm(`layers.${i}.qNormRaw`),
      kNormRaw: getNorm(`layers.${i}.kNormRaw`),
    });
  }

  return {
    config: manifest.config,
    embedTokens: getQuant("embedTokens"),
    perLayer,
    finalNorm: getNorm("finalNorm"),
  };
}

/** Sign-extends the low byte of a packed word's lane — `matvecQ8`'s own unpackI8. */
function signExtend(byte: number): number {
  return byte >= 128 ? byte - 256 : byte;
}

/** The exact integer codes of one row, unpacked from the wire format. */
export function unpackRowCodes(weight: PackedQ8, row: number): Int32Array {
  if (row < 0 || row >= weight.rows) throw new Error(`row ${row} out of [0, ${weight.rows})`);
  const wordsPerRow = Math.ceil(weight.cols / 4);
  const base = row * wordsPerRow;
  const out = new Int32Array(weight.cols);
  for (let col = 0; col < weight.cols; col += 1) {
    const word = weight.packed[base + (col >> 2)]!;
    out[col] = signExtend((word >>> ((col & 3) * 8)) & 0xff);
  }
  return out;
}

/** One row dequantized to f32: `code * scale[row]`. */
export function dequantizeRow(weight: PackedQ8, row: number): Float32Array {
  const codes = unpackRowCodes(weight, row);
  const s = weight.scale[row]!;
  const out = new Float32Array(weight.cols);
  for (let col = 0; col < weight.cols; col += 1) out[col] = codes[col]! * s;
  return out;
}

/**
 * The embedding gather for one token id. Ids outside `[0, rows)` return zeros
 * — `ops/gather`'s convention — because an out-of-vocab id from an externally
 * encoded prompt must not read undefined memory and NaN its way through 28
 * layers into a plausible-looking argmax.
 */
export function gatherDequantRow(table: PackedQ8, id: number): Float32Array {
  if (id < 0 || id >= table.rows) return new Float32Array(table.cols);
  return dequantizeRow(table, id);
}
