import { describe, expect, it } from "vitest";
import { checkEncoderProvenance } from "./weights-cache.js";

/**
 * The consume-time sha cross-check: `export_encoder_weights.py` records both
 * source checkpoints' sha256 in `encoder-weights.json` and diffs them against
 * the golden's `index.json` at EXPORT time — but weights exported before a
 * checkpoint moved stay stale on disk, and nothing used to check at LOAD time,
 * so every stage comparison would chase a phantom. The pure comparison is
 * tested here; `loadEncoderWeights()` calls it against the real files.
 */

const SHA_A = "a".repeat(64);
const SHA_B = "b".repeat(64);
const SHA_C = "c".repeat(64);

const manifest = { sources: { miocodec: { sha256: SHA_A }, wavlm: { sha256: SHA_B } } };
const index = { checkpoint: { sha256: SHA_A }, ssl_checkpoint: { sha256: SHA_B } };

describe("checkEncoderProvenance", () => {
  it("passes when both recorded shas match the golden's", () => {
    expect(() => checkEncoderProvenance(manifest, index)).not.toThrow();
  });

  it("throws with the export command when the miocodec sha drifted", () => {
    const stale = { checkpoint: { sha256: SHA_C }, ssl_checkpoint: { sha256: SHA_B } };
    expect(() => checkEncoderProvenance(manifest, stale)).toThrow(/miocodec/);
    expect(() => checkEncoderProvenance(manifest, stale)).toThrow(/export_encoder_weights\.py/);
  });

  it("throws when the wavlm sha drifted", () => {
    const stale = { checkpoint: { sha256: SHA_A }, ssl_checkpoint: { sha256: SHA_C } };
    expect(() => checkEncoderProvenance(manifest, stale)).toThrow(/wavlm/);
  });

  it("throws when the manifest records no sha at all (a truncated export is not a pass)", () => {
    expect(() => checkEncoderProvenance({}, index)).toThrow(/miocodec/);
  });
});
