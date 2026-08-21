import { describe, expect, it } from "vitest";

import * as miotts from "../../../src/engine/miotts/index.js";

/**
 * What `voxshot/miotts` promises a consumer.
 *
 * The subpath is a second published surface, so what it exports is a decision
 * rather than a side effect of which files exist. `scripts/check-package.mjs`
 * checks the other half — that the subpath actually resolves out of the packed
 * tarball, which this test cannot see.
 */
describe("the voxshot/miotts surface", () => {
  it("names the parts a weight source has to answer", () => {
    expect(miotts.MIOTTS_WEIGHT_PARTS).toContain("lm-codes");
    expect(miotts.MIOTTS_WEIGHT_PARTS).toContain("codec-encoder");
  });

  it("exports the constants a caller needs before constructing anything", () => {
    // Enough to build a VoiceEmbedding by hand from a stored 128-float vector,
    // which is the path for an application whose speaker never changes.
    expect(miotts.MIOTTS_ENGINE_NAME).toBe("miotts");
    expect(miotts.MIOTTS_SAMPLE_RATE).toBe(24_000);
    expect(miotts.SPEAKER_EMBEDDING_SIZE).toBe(128);
  });

  it("does not leak the internals the engine happens to be built from", () => {
    // The tokenizer, the text normalizer and the q8 loader are how this engine
    // works, not what it offers. Exporting them would make every one of them
    // something a release has to keep working.
    expect(miotts).not.toHaveProperty("loadTokenizer");
    expect(miotts).not.toHaveProperty("normalizeText");
    expect(miotts).not.toHaveProperty("loadPart");
  });
});
