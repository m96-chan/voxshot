import { describe, expect, it } from "vitest";

import {
  dequantizeRow,
  gatherDequantRow,
  loadWeightsQ8,
  unpackRowCodes,
} from "../../../../src/engine/miotts/lm/weights-q8.js";
import { TINY_CONFIG, buildSyntheticQ8 } from "../../../helpers/q8.js";

/**
 * The q8 loader against a synthetic artifact set.
 *
 * The real one is 583 MB and not in git. What is checked here is everything
 * that does not depend on the weights *being* Qwen3: the refusals, the layout
 * arithmetic, and the packing into `matvecQ8`'s wire format. Whether the
 * numbers that come out are the checkpoint's is `npm run test:models`' job.
 */

describe("loadWeightsQ8", () => {
  it("loads every tensor the decode loop reaches for", () => {
    const weights = loadWeightsQ8(buildSyntheticQ8());

    expect(weights.config).toEqual(TINY_CONFIG);
    expect(weights.perLayer).toHaveLength(TINY_CONFIG.numLayers);
    expect(weights.embedTokens.rows).toBe(TINY_CONFIG.vocabSize);
    expect(weights.embedTokens.cols).toBe(TINY_CONFIG.hiddenSize);
    expect(weights.finalNorm).toHaveLength(TINY_CONFIG.hiddenSize);

    const layer = weights.perLayer[0]!;
    expect(layer.wq.rows).toBe(TINY_CONFIG.numHeads * TINY_CONFIG.headDim);
    expect(layer.wk.rows).toBe(TINY_CONFIG.numKvHeads * TINY_CONFIG.headDim);
    expect(layer.wDown.cols).toBe(TINY_CONFIG.ffnHidden);
    expect(layer.qNorm).toHaveLength(TINY_CONFIG.headDim);
  });

  it("refuses artifacts whose rope channels were never permuted", () => {
    // An unpermuted checkpoint fed to the ops/rope kernel produces garbage
    // that still looks like numbers, so this has to be a refusal rather than
    // something noticed downstream.
    const artifact = buildSyntheticQ8();
    artifact.manifest.ropePermuted = false;

    expect(() => loadWeightsQ8(artifact)).toThrow(/ropePermuted/);
  });

  it("refuses a bin whose length disagrees with the manifest", () => {
    const artifact = buildSyntheticQ8();
    artifact.manifest.files.scales.bytes += 4;

    expect(() => loadWeightsQ8(artifact)).toThrow(/scales.*weights\.scales\.bin.*manifest says/s);
  });

  it("checks each bin against its own entry", () => {
    const artifact = buildSyntheticQ8();
    artifact.manifest.files.norms.bytes += 4;

    expect(() => loadWeightsQ8(artifact)).toThrow(/norms/);
  });

  it("digests the bins when a hasher is supplied", () => {
    const artifact = buildSyntheticQ8();
    const seen: number[] = [];

    loadWeightsQ8({
      ...artifact,
      sha256: (bytes) => {
        seen.push(bytes.byteLength);
        return { [artifact.manifest.files.codes.bytes]: "c".repeat(64),
                 [artifact.manifest.files.scales.bytes]: "s".repeat(64),
                 [artifact.manifest.files.norms.bytes]: "n".repeat(64) }[bytes.byteLength]!;
      },
    });

    expect(seen).toHaveLength(3);
  });

  it("refuses a bin whose digest does not match, naming what to do about it", () => {
    // A stale or truncated artifact. The message says regenerate, because
    // there is nothing to debug in a file that is not the file.
    const artifact = buildSyntheticQ8();

    expect(() => loadWeightsQ8({ ...artifact, sha256: () => "0".repeat(64) })).toThrow(
      /does not match the manifest.*regenerate/s,
    );
  });

  it("skips the digest entirely when no hasher is given", () => {
    // The browser path does this deliberately: SubtleCrypto is async and a
    // ~600 MiB digest costs seconds on the load path. Byte lengths still run.
    const artifact = buildSyntheticQ8();
    artifact.manifest.files.codes.sha256 = "deadbeef".repeat(8);

    expect(() => loadWeightsQ8(artifact)).not.toThrow();
  });

  it("names a quantized tensor the manifest does not carry", () => {
    const artifact = buildSyntheticQ8();
    artifact.manifest.tensors = artifact.manifest.tensors.filter(
      (tensor) => tensor.name !== "layers.1.wUp",
    );

    expect(() => loadWeightsQ8(artifact)).toThrow(/no quantized tensor named "layers\.1\.wUp"/);
  });

  it("names a norm tensor the manifest does not carry", () => {
    const artifact = buildSyntheticQ8();
    artifact.manifest.tensors = artifact.manifest.tensors.filter(
      (tensor) => tensor.name !== "layers.0.kNormRaw",
    );

    expect(() => loadWeightsQ8(artifact)).toThrow(/no norm tensor named "layers\.0\.kNormRaw"/);
  });

  it("refuses a tensor whose byte counts contradict its own shape", () => {
    const artifact = buildSyntheticQ8();
    const entry = artifact.manifest.tensors.find((tensor) => tensor.name === "layers.0.wv")!;
    (entry as { codesBytes: number }).codesBytes += 1;

    expect(() => loadWeightsQ8(artifact)).toThrow(/byte counts disagree with rows=/);
  });

  it("copies the scales rather than viewing the bin", () => {
    // A view would pin the whole scales buffer alive behind one small array.
    const artifact = buildSyntheticQ8();
    const weights = loadWeightsQ8(artifact);

    expect(weights.perLayer[0]!.wq.scale.buffer).not.toBe(artifact.scales);
    expect(weights.finalNorm.buffer).not.toBe(artifact.norms);
  });
});

describe("the packed wire format", () => {
  const weights = loadWeightsQ8(buildSyntheticQ8());
  const table = weights.embedTokens;

  it("packs four codes per word, least significant byte first", () => {
    // `matvecQ8`'s own convention. Unpacking has to return the codes that went
    // in, or every matvec reads a differently-ordered row.
    const row = unpackRowCodes(table, 3);

    expect(row).toHaveLength(table.cols);
    const words = Math.ceil(table.cols / 4);
    expect(table.packed).toHaveLength(table.rows * words);
  });

  it("gives each row its own codes", () => {
    expect(Array.from(unpackRowCodes(table, 0))).not.toEqual(Array.from(unpackRowCodes(table, 1)));
  });

  it("sign-extends codes back into negatives", () => {
    // The codes are int8 in [-127, 127]; a row read as unsigned would come
    // back entirely non-negative and every product would have the wrong sign.
    const anyNegative = Array.from(unpackRowCodes(table, 5)).some((code) => code < 0);

    expect(anyNegative).toBe(true);
  });

  it("refuses a row outside the table", () => {
    expect(() => unpackRowCodes(table, -1)).toThrow(/out of \[0, 12\)/);
    expect(() => unpackRowCodes(table, 12)).toThrow(/out of \[0, 12\)/);
  });

  it("dequantizes a row as code times that row's scale", () => {
    const row = 4;
    const codes = unpackRowCodes(table, row);
    const scale = table.scale[row]!;

    const values = dequantizeRow(table, row);

    // Expected as a Float32Array, not a number[]: the products are rounded to
    // f32 on the way into the output, and comparing against f64 arithmetic
    // would fail on the rounding rather than on anything about the loader.
    expect(values).toEqual(Float32Array.from(codes, (code) => code * scale));
  });
});

describe("gatherDequantRow", () => {
  const weights = loadWeightsQ8(buildSyntheticQ8());
  const table = weights.embedTokens;

  it("returns the embedding row for an id in the vocabulary", () => {
    expect(Array.from(gatherDequantRow(table, 7))).toEqual(Array.from(dequantizeRow(table, 7)));
  });

  it("returns zeros for an id outside it, rather than reading past the table", () => {
    // `ops/gather`'s convention, and the reason this is worth pinning: an
    // out-of-vocab id from an externally encoded prompt must not read
    // undefined memory and NaN its way through every layer into an argmax
    // that still looks like a token.
    expect(Array.from(gatherDequantRow(table, table.rows))).toEqual(new Array(table.cols).fill(0));
    expect(Array.from(gatherDequantRow(table, -1))).toEqual(new Array(table.cols).fill(0));
  });
});
