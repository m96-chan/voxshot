import { createHash } from "node:crypto";
import { closeSync, existsSync, openSync, readFileSync, readSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it } from "vitest";
import { quantize } from "web-xpu-ops/ops/quantize";
import { matvec, matvecQ8, packQ8 } from "web-xpu-ops/ops/matvec";

import {
  dequantizeRow,
  gatherDequantRow,
  loadWeightsQ8,
  unpackRowCodes,
  type PackedQ8,
  type Q8TensorEntry,
  type Qwen3WeightsQ8,
  type WeightsQ8Manifest,
} from "../../src/engine/miotts/lm/weights-q8.js";
import { loadWeightsQ8FromDir, sha256Hex } from "./weights-q8-node.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const Q8_DIR = join(HERE, "q8");
const GOLDEN_INDEX = join(HERE, "golden", "index.json");

/**
 * The RoPE channel relabeling, restated here independently of both the
 * converter (python) and `llm/weights.ts#permuteRopeChannels` so a bug in
 * either transcription is a disagreement, not a shared blind spot:
 * HF channel j inside a head becomes ops/rope channel 2j (j < half) or
 * 2(j - half) + 1 (j >= half).
 */
function pi(j: number, headDim: number): number {
  const half = headDim / 2;
  return j < half ? 2 * j : 2 * (j - half) + 1;
}

// ---------------------------------------------------------------------------
// Synthetic fixture: a tiny model laid out exactly the way convert_weights.py
// writes the real one, built in memory so the loader's parsing, packing, and
// hash verification are testable without the 600 MB artifacts on disk.
// ---------------------------------------------------------------------------

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomF32(n: number, rng: () => number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 1) out[i] = Math.fround((rng() - 0.5) * 4);
  return out;
}

interface Fixture {
  manifest: WeightsQ8Manifest;
  codes: ArrayBuffer;
  scales: ArrayBuffer;
  norms: ArrayBuffer;
  /** The pre-quantization f32 source of every quant tensor, by manifest name. */
  raw: Map<string, { data: Float32Array; rows: number; cols: number }>;
}

const TINY = {
  numLayers: 2,
  hiddenSize: 8,
  numHeads: 2,
  numKvHeads: 1,
  headDim: 4,
  ffnHidden: 12,
  vocabSize: 10,
  ropeTheta: 1e6,
  rmsNormEps: 1e-6,
  tieEmbeddings: true,
  speechTokenBase: 5,
  eosIds: [1],
};

function buildFixture(): Fixture {
  const rng = mulberry32(20260821);
  const tensors: Q8TensorEntry[] = [];
  const raw = new Map<string, { data: Float32Array; rows: number; cols: number }>();
  const codeChunks: Int8Array[] = [];
  const scaleChunks: Float32Array[] = [];
  const normChunks: Float32Array[] = [];
  let codesOffset = 0;
  let scaleOffset = 0;
  let normOffset = 0;

  const writeQuant = (name: string, rows: number, cols: number) => {
    const data = randomF32(rows * cols, rng);
    raw.set(name, { data, rows, cols });
    const { output, scales } = quantize({ input: data, N: rows, D: cols });
    codeChunks.push(Int8Array.from(output));
    scaleChunks.push(scales);
    tensors.push({
      name,
      kind: "quant",
      rows,
      cols,
      codesOffset,
      codesBytes: rows * cols,
      scaleOffset,
      scaleBytes: rows * 4,
    });
    codesOffset += rows * cols;
    scaleOffset += rows * 4;
  };
  const writeNorm = (name: string, cols: number) => {
    const data = randomF32(cols, rng);
    normChunks.push(data);
    tensors.push({ name, kind: "norm", cols, offset: normOffset, bytes: cols * 4 });
    normOffset += cols * 4;
  };

  const { hiddenSize: h, numHeads, numKvHeads, headDim, ffnHidden, vocabSize } = TINY;
  writeQuant("embedTokens", vocabSize, h);
  for (let i = 0; i < TINY.numLayers; i += 1) {
    writeNorm(`layers.${i}.attnNorm`, h);
    writeQuant(`layers.${i}.wq`, numHeads * headDim, h);
    writeQuant(`layers.${i}.wk`, numKvHeads * headDim, h);
    writeQuant(`layers.${i}.wv`, numKvHeads * headDim, h);
    writeQuant(`layers.${i}.wo`, h, numHeads * headDim);
    writeNorm(`layers.${i}.ffnNorm`, h);
    writeQuant(`layers.${i}.wGate`, ffnHidden, h);
    writeQuant(`layers.${i}.wUp`, ffnHidden, h);
    writeQuant(`layers.${i}.wDown`, h, ffnHidden);
    writeNorm(`layers.${i}.qNorm`, headDim);
    writeNorm(`layers.${i}.kNorm`, headDim);
    writeNorm(`layers.${i}.qNormRaw`, headDim);
    writeNorm(`layers.${i}.kNormRaw`, headDim);
  }
  writeNorm("finalNorm", h);

  const concatI8 = (chunks: Int8Array[], total: number) => {
    const out = new Int8Array(total);
    let at = 0;
    for (const c of chunks) {
      out.set(c, at);
      at += c.length;
    }
    return out;
  };
  const concatF32 = (chunks: Float32Array[], totalBytes: number) => {
    const out = new Float32Array(totalBytes / 4);
    let at = 0;
    for (const c of chunks) {
      out.set(c, at);
      at += c.length;
    }
    return out;
  };
  const codes = concatI8(codeChunks, codesOffset);
  const scales = concatF32(scaleChunks, scaleOffset);
  const norms = concatF32(normChunks, normOffset);
  const digest = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");

  const manifest: WeightsQ8Manifest = {
    generatedBy: "weights-q8.test.ts (synthetic fixture)",
    source: { path: "(none)", bytes: 0, sha256: "0".repeat(64) },
    config: { ...TINY },
    ropePermuted: true,
    ropePermutedNote: "synthetic fixture; no real permutation applied",
    files: {
      codes: { name: "weights.codes.bin", bytes: codes.byteLength, sha256: digest(new Uint8Array(codes.buffer)) },
      scales: {
        name: "weights.scales.bin",
        bytes: scales.byteLength,
        sha256: digest(new Uint8Array(scales.buffer)),
      },
      norms: {
        name: "weights.norms.bin",
        bytes: norms.byteLength,
        sha256: digest(new Uint8Array(norms.buffer)),
      },
    },
    tensors,
  };
  return { manifest, codes: codes.buffer, scales: scales.buffer, norms: norms.buffer, raw };
}

function maxRel(actual: ArrayLike<number>, expected: ArrayLike<number>): number {
  let peak = 0;
  for (let i = 0; i < expected.length; i += 1) peak = Math.max(peak, Math.abs(expected[i]!));
  let worst = 0;
  for (let i = 0; i < expected.length; i += 1) {
    worst = Math.max(worst, Math.abs(actual[i]! - expected[i]!) / peak);
  }
  return worst;
}

describe("loadWeightsQ8 (synthetic fixture)", () => {
  const fixture = buildFixture();
  let weights: Qwen3WeightsQ8;

  beforeAll(() => {
    weights = loadWeightsQ8({ ...fixture, sha256: sha256Hex });
  });

  it("carries the manifest config through unchanged", () => {
    expect(weights.config).toEqual(TINY);
    expect(weights.perLayer).toHaveLength(TINY.numLayers);
    expect(weights.embedTokens.rows).toBe(TINY.vocabSize);
    expect(weights.embedTokens.cols).toBe(TINY.hiddenSize);
  });

  it("packs every quant tensor into matvecQ8's wire format bit-for-bit", () => {
    const check = (name: string, packed: PackedQ8) => {
      const src = fixture.raw.get(name)!;
      const { output, scales } = quantize({ input: src.data, N: src.rows, D: src.cols });
      const expected = packQ8({ codes: output, N: src.rows, K: src.cols });
      expect(packed.packed.length, name).toBe(expected.length);
      for (let i = 0; i < expected.length; i += 1) {
        if (packed.packed[i] !== expected[i]) {
          throw new Error(`${name}: packed word ${i} is ${packed.packed[i]}, expected ${expected[i]}`);
        }
      }
      expect(Array.from(packed.scale), name).toEqual(Array.from(scales));
    };
    check("embedTokens", weights.embedTokens);
    check("layers.0.wq", weights.perLayer[0]!.wq);
    check("layers.1.wDown", weights.perLayer[1]!.wDown);
  });

  it("roundtrips exact integer codes through unpackRowCodes", () => {
    const src = fixture.raw.get("layers.0.wq")!;
    const { output } = quantize({ input: src.data, N: src.rows, D: src.cols });
    for (let row = 0; row < src.rows; row += 1) {
      const codes = unpackRowCodes(weights.perLayer[0]!.wq, row);
      expect(Array.from(codes)).toEqual(
        Array.from(output.subarray(row * src.cols, (row + 1) * src.cols)),
      );
    }
  });

  it("matvecQ8 on the loaded weight agrees with f32 matvec on the dequantized matrix", () => {
    const src = fixture.raw.get("layers.0.wGate")!;
    const w = weights.perLayer[0]!.wGate;
    const rng = mulberry32(7);
    const vector = randomF32(src.cols, rng);
    const dequant = new Float32Array(src.rows * src.cols);
    for (let r = 0; r < src.rows; r += 1) dequant.set(dequantizeRow(w, r), r * src.cols);
    const expected = matvec({ matrix: dequant, vector, M: src.rows, K: src.cols });
    const actual = matvecQ8({ weight: w.packed, scale: w.scale, vector, N: src.rows, K: src.cols });
    // Same math, different association: matvecQ8 scales once after the int dot,
    // the dequantized matvec scales each term (rounded to f32) first.
    expect(maxRel(actual, expected)).toBeLessThanOrEqual(1e-5);
  });

  it("gatherDequantRow returns codes*scale for in-range ids and zeros out of range", () => {
    const src = fixture.raw.get("embedTokens")!;
    const { output, scales } = quantize({ input: src.data, N: src.rows, D: src.cols });
    const id = 3;
    const row = gatherDequantRow(weights.embedTokens, id);
    const expected = new Float32Array(src.cols);
    for (let c = 0; c < src.cols; c += 1) {
      expected[c] = Math.fround(output[id * src.cols + c]! * scales[id]!);
    }
    expect(Array.from(row)).toEqual(Array.from(expected));
    expect(Array.from(gatherDequantRow(weights.embedTokens, -1))).toEqual(
      Array.from(new Float32Array(src.cols)),
    );
    expect(Array.from(gatherDequantRow(weights.embedTokens, src.rows))).toEqual(
      Array.from(new Float32Array(src.cols)),
    );
  });

  it("throws when a codes byte is corrupted (sha256 mismatch)", () => {
    const corrupted = fixture.codes.slice(0);
    const view = new Int8Array(corrupted);
    view[17] = (view[17]! + 1) << 24 >> 24;
    expect(() =>
      loadWeightsQ8({ ...fixture, codes: corrupted, sha256: sha256Hex }),
    ).toThrow(/sha256/);
  });

  it("throws when a bin file's length disagrees with the manifest even without a hasher", () => {
    expect(() =>
      loadWeightsQ8({ ...fixture, codes: fixture.codes.slice(0, fixture.codes.byteLength - 1) }),
    ).toThrow(/bytes/);
  });
});

// ---------------------------------------------------------------------------
// Integration against the real converted artifacts in q8/. Guarded the same
// way golden tests are: absent artifacts fail loudly with the rebuild command,
// they never skip.
// ---------------------------------------------------------------------------

interface SafetensorsFile {
  fd: number;
  dataStart: number;
  tensors: Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>;
}

function openSafetensors(path: string): SafetensorsFile {
  const fd = openSync(path, "r");
  const lenBuf = Buffer.alloc(8);
  readSync(fd, lenBuf, 0, 8, 0);
  const headerLen = Number(lenBuf.readBigUInt64LE(0));
  const headerBuf = Buffer.alloc(headerLen);
  readSync(fd, headerBuf, 0, headerLen, 8);
  const tensors = JSON.parse(headerBuf.toString("utf8")) as SafetensorsFile["tensors"];
  delete (tensors as Record<string, unknown>)["__metadata__"];
  return { fd, dataStart: 8 + headerLen, tensors };
}

/** One row of a BF16 tensor, widened exactly to f32 (u16 << 16). */
function readBf16Row(st: SafetensorsFile, name: string, row: number): Float32Array {
  const meta = st.tensors[name];
  if (!meta) throw new Error(`no tensor ${name} in checkpoint`);
  if (meta.dtype !== "BF16") throw new Error(`${name} is ${meta.dtype}, expected BF16`);
  const cols = meta.shape.length === 2 ? meta.shape[1]! : meta.shape[0]!;
  if (meta.shape.length === 1 && row !== 0) throw new Error(`${name} is 1-D; row must be 0`);
  const buf = Buffer.alloc(cols * 2);
  readSync(st.fd, buf, 0, cols * 2, st.dataStart + meta.data_offsets[0] + row * cols * 2);
  const out = new Float32Array(cols);
  const scratch = new DataView(new ArrayBuffer(4));
  for (let c = 0; c < cols; c += 1) {
    scratch.setUint32(0, buf.readUInt16LE(c * 2) << 16 >>> 0, true);
    out[c] = scratch.getFloat32(0, true);
  }
  return out;
}

describe("q8 artifacts (converter integration)", () => {
  let weights: Qwen3WeightsQ8;
  let manifest: WeightsQ8Manifest;
  let st: SafetensorsFile;
  let goldenConfig: Record<string, unknown>;

  beforeAll(() => {
    if (!existsSync(join(Q8_DIR, "manifest.json"))) {
      throw new Error(
        "q8/ artifacts are missing. They are deliberately not in git; rebuild with\n" +
          "  cd spike/miotts && python3 convert_weights.py",
      );
    }
    weights = loadWeightsQ8FromDir(Q8_DIR);
    manifest = JSON.parse(readFileSync(join(Q8_DIR, "manifest.json"), "utf8")) as WeightsQ8Manifest;
    const index = JSON.parse(readFileSync(GOLDEN_INDEX, "utf8")) as {
      config: Record<string, unknown>;
    };
    goldenConfig = index.config;
    if (!existsSync(manifest.source.path)) {
      throw new Error(`checkpoint ${manifest.source.path} is missing; cannot cross-check rows`);
    }
    st = openSafetensors(manifest.source.path);
  });

  it("manifest config matches golden/index.json", () => {
    expect(manifest.config.numLayers).toBe(goldenConfig["num_layers"]);
    expect(manifest.config.hiddenSize).toBe(goldenConfig["hidden"]);
    expect(manifest.config.numHeads).toBe(goldenConfig["heads"]);
    expect(manifest.config.numKvHeads).toBe(goldenConfig["kv_heads"]);
    expect(manifest.config.headDim).toBe(goldenConfig["head_dim"]);
    expect(manifest.config.ffnHidden).toBe(goldenConfig["ffn"]);
    expect(manifest.config.vocabSize).toBe(goldenConfig["vocab"]);
    expect(manifest.config.ropeTheta).toBe(goldenConfig["rope_theta"]);
    expect(manifest.config.rmsNormEps).toBe(goldenConfig["rms_eps"]);
    expect(manifest.config.tieEmbeddings).toBe(goldenConfig["tie_word_embeddings"]);
    expect(manifest.config.speechTokenBase).toBe(goldenConfig["speech_token_base"]);
    expect(manifest.config.eosIds).toEqual(goldenConfig["eos_ids"]);
    expect(manifest.ropePermuted).toBe(true);
  });

  it("embed_tokens row 151669 dequantizes to within scale/2 of the bf16 source", () => {
    const id = manifest.config.speechTokenBase;
    const src = readBf16Row(st, "model.embed_tokens.weight", id);
    const deq = gatherDequantRow(weights.embedTokens, id);
    const scale = weights.embedTokens.scale[id]!;
    // Round-half-up to the nearest code puts every non-clamped element within
    // scale/2; the max-magnitude element maps to +-127 exactly. The epsilon
    // covers the scale's own f64 -> f32 narrowing.
    const bound = scale / 2 + scale * 1e-6;
    for (let c = 0; c < src.length; c += 1) {
      const err = Math.abs(deq[c]! - src[c]!);
      if (err > bound) {
        throw new Error(`col ${c}: |${deq[c]} - ${src[c]}| = ${err} > ${bound}`);
      }
    }
    // Stronger: the converter's quantization is bit-for-bit ops/quantize.
    const { output, scales } = quantize({ input: src, N: 1, D: src.length });
    expect(scale).toBe(scales[0]!);
    expect(Array.from(unpackRowCodes(weights.embedTokens, id))).toEqual(Array.from(output));
  });

  it("layer 0 wq rows are RoPE-permuted relative to the checkpoint", () => {
    const headDim = manifest.config.headDim;
    // Head 0, HF channel j=1: pi(1)=2, so checkpoint row 1 must be stored at
    // row 2, and stored row 1 (checkpoint row 64) must differ from it.
    const j = 1;
    const srcRow = readBf16Row(st, "model.layers.0.self_attn.q_proj.weight", j);
    const { output, scales } = quantize({ input: srcRow, N: 1, D: srcRow.length });
    const expectedCodes = Array.from(output);

    // The discriminant is sharp: the row pi(j) receives is genuinely different
    // from the row j would have kept unpermuted.
    const srcRowOther = readBf16Row(st, "model.layers.0.self_attn.q_proj.weight", 64);
    const other = quantize({ input: srcRowOther, N: 1, D: srcRowOther.length });
    expect(Array.from(other.output)).not.toEqual(expectedCodes);

    const wq = weights.perLayer[0]!.wq;
    expect(Array.from(unpackRowCodes(wq, pi(j, headDim)))).toEqual(expectedCodes);
    expect(wq.scale[pi(j, headDim)]).toBe(scales[0]!);
    expect(Array.from(unpackRowCodes(wq, j))).not.toEqual(expectedCodes);
    // Same check in the second head, so the permutation is per-head, not global.
    const j2 = headDim + 70; // head 1, HF channel 70 -> stored row headDim + pi(70)
    const srcRow2 = readBf16Row(st, "model.layers.0.self_attn.q_proj.weight", j2);
    const q2 = quantize({ input: srcRow2, N: 1, D: srcRow2.length });
    expect(Array.from(unpackRowCodes(wq, headDim + pi(70, headDim)))).toEqual(Array.from(q2.output));
  });

  it("layer 0 q_norm gamma is permuted with the same pi, raw copy matches the checkpoint", () => {
    const headDim = manifest.config.headDim;
    const layer = weights.perLayer[0]!;
    const src = readBf16Row(st, "model.layers.0.self_attn.q_norm.weight", 0);
    expect(Array.from(layer.qNormRaw)).toEqual(Array.from(src));
    for (let j = 0; j < headDim; j += 1) {
      expect(layer.qNorm[pi(j, headDim)], `j=${j}`).toBe(layer.qNormRaw[j]!);
    }
    // The permutation actually moved something (a constant gamma would make
    // this test blind; the trained gammas are not constant).
    expect(Array.from(layer.qNorm)).not.toEqual(Array.from(layer.qNormRaw));
    const kSrc = readBf16Row(st, "model.layers.0.self_attn.k_norm.weight", 0);
    expect(Array.from(layer.kNormRaw)).toEqual(Array.from(kSrc));
    for (let j = 0; j < headDim; j += 1) {
      expect(layer.kNorm[pi(j, headDim)], `j=${j}`).toBe(layer.kNormRaw[j]!);
    }
  });

  it("wk/wv/norm shapes line up with the config", () => {
    const { hiddenSize, numHeads, numKvHeads, headDim, ffnHidden, vocabSize, numLayers } =
      manifest.config;
    expect(weights.perLayer).toHaveLength(numLayers);
    expect(weights.embedTokens.rows).toBe(vocabSize);
    expect(weights.embedTokens.cols).toBe(hiddenSize);
    expect(weights.finalNorm).toHaveLength(hiddenSize);
    const l = weights.perLayer[27]!;
    expect(l.wq.rows).toBe(numHeads * headDim);
    expect(l.wk.rows).toBe(numKvHeads * headDim);
    expect(l.wv.rows).toBe(numKvHeads * headDim);
    expect(l.wo.rows).toBe(hiddenSize);
    expect(l.wo.cols).toBe(numHeads * headDim);
    expect(l.wGate.rows).toBe(ffnHidden);
    expect(l.wUp.rows).toBe(ffnHidden);
    expect(l.wDown.rows).toBe(hiddenSize);
    expect(l.wDown.cols).toBe(ffnHidden);
    expect(l.attnNorm).toHaveLength(hiddenSize);
    expect(l.ffnNorm).toHaveLength(hiddenSize);
    expect(l.qNorm).toHaveLength(headDim);
    expect(l.kNorm).toHaveLength(headDim);
  });

  it("closes the checkpoint fd", () => {
    closeSync(st.fd);
  });
});
