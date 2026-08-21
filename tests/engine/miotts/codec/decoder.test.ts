import { describe, expect, it } from "vitest";

import {
  MIOCODEC_24K,
  Weights,
  cpuBackend,
  decode,
  fsqDecode,
  layerNorm,
  transpose2d,
} from "../../../../src/engine/miotts/codec/decoder.js";
import { Safetensors } from "../../../../src/engine/miotts/codec/safetensors.js";
import {
  TINY_DECODER,
  buildTinyDecoderCheckpoint,
  tinyGlobalEmbedding,
  tinyTokens,
} from "../../../helpers/codec.js";

/**
 * The decoder graph, driven end to end on a checkpoint small enough for CI.
 *
 * What this can check and what it cannot are worth being explicit about.
 *
 * It **cannot** check that the numbers are MioCodec's. That needs the real
 * 523 MB checkpoint and goldens from the reference implementation, and it lives
 * in `npm run test:models`, which is a release gate precisely because CI has no
 * way to catch a numerical regression in a ported kernel.
 *
 * It **can** check that the graph is wired: that every stage hands the next one
 * the shape it expects, that a transposed axis or a swapped dimension fails,
 * that the output length follows from the token count, and that the audio is
 * finite. Those are this project's mistakes to make — the kernels themselves
 * belong to web-xpu-ops and are tested there.
 */

const weights = new Weights(Safetensors.parse(buildTinyDecoderCheckpoint()));
const globalEmbedding = tinyGlobalEmbedding();

/** The "aligned" path: two STFT frames per 25 Hz token. */
const framesFor = (tokens: number) => 2 * tokens;

describe("decode", () => {
  it("runs the whole graph and returns finite audio", async () => {
    const tokens = tinyTokens(4);

    const { waveform } = await decode(
      tokens,
      globalEmbedding,
      framesFor(tokens.length),
      TINY_DECODER,
      weights,
      cpuBackend,
    );

    expect(waveform.length).toBeGreaterThan(0);
    expect(waveform.every(Number.isFinite)).toBe(true);
  });

  it("produces audio proportional to the token count", async () => {
    // The token rate is what ties generated tokens to seconds of speech, so a
    // stage that quietly dropped or duplicated frames would show up here.
    const short = await decode(
      tinyTokens(4),
      globalEmbedding,
      framesFor(4),
      TINY_DECODER,
      weights,
      cpuBackend,
    );
    const long = await decode(
      tinyTokens(8),
      globalEmbedding,
      framesFor(8),
      TINY_DECODER,
      weights,
      cpuBackend,
    );

    expect(long.waveform.length - short.waveform.length).toBe(4 * 2 * TINY_DECODER.hopLength);
  });

  it("reports every stage at the shape the next one consumes", async () => {
    const tokens = tinyTokens(4);
    const frames = framesFor(tokens.length);
    const dim = TINY_DECODER.decoder.dim;
    const bins = TINY_DECODER.nFft / 2 + 1;

    const { stages } = await decode(
      tokens,
      globalEmbedding,
      frames,
      TINY_DECODER,
      weights,
      cpuBackend,
    );

    // Length-major through the attention stages, channel-major through the
    // convolutions, and back again — the transposes between them are where a
    // wiring mistake actually lives.
    expect(stages.content_embedding!.shape).toEqual([tokens.length, TINY_DECODER.prenet.dim]);
    expect(stages.after_prenet!.shape).toEqual([tokens.length, TINY_DECODER.prenet.outputDim]);
    expect(stages.after_conv_upsample!.shape).toEqual([dim, 2 * tokens.length]);
    expect(stages.after_interpolate!.shape).toEqual([dim, frames]);
    expect(stages.after_prior_net!.shape).toEqual([dim, frames]);
    expect(stages.after_decoder!.shape).toEqual([frames, dim]);
    expect(stages.after_post_net!.shape).toEqual([dim, frames]);
    expect(stages.spec_real!.shape).toEqual([frames, bins]);
    expect(stages.spec_imag!.shape).toEqual([frames, bins]);
  });

  it("lets the speaker embedding change the audio", async () => {
    // Speaker identity enters only here, as AdaLN conditioning — the language
    // model never sees the voice. If the conditioning were dropped, every
    // speaker would render identically and nothing else in the pipeline would
    // notice.
    const tokens = tinyTokens(4);
    const other = tinyGlobalEmbedding().map((value) => -value);

    const a = await decode(tokens, globalEmbedding, framesFor(4), TINY_DECODER, weights, cpuBackend);
    const b = await decode(tokens, other, framesFor(4), TINY_DECODER, weights, cpuBackend);

    expect(Array.from(a.waveform)).not.toEqual(Array.from(b.waveform));
  });

  it("lets the tokens change the audio", async () => {
    const a = await decode(
      tinyTokens(4),
      globalEmbedding,
      framesFor(4),
      TINY_DECODER,
      weights,
      cpuBackend,
    );
    const b = await decode(
      Float32Array.from([1, 1, 1, 1]),
      globalEmbedding,
      framesFor(4),
      TINY_DECODER,
      weights,
      cpuBackend,
    );

    expect(Array.from(a.waveform)).not.toEqual(Array.from(b.waveform));
  });

  it("is deterministic", async () => {
    // Nothing in the decoder samples, so two runs of the same input must agree
    // exactly. VoxShot's synthesis cache depends on it.
    const tokens = tinyTokens(4);
    const a = await decode(tokens, globalEmbedding, framesFor(4), TINY_DECODER, weights, cpuBackend);
    const b = await decode(tokens, globalEmbedding, framesFor(4), TINY_DECODER, weights, cpuBackend);

    expect(Array.from(a.waveform)).toEqual(Array.from(b.waveform));
  });

  it("defaults to the reference backend", async () => {
    const tokens = tinyTokens(4);
    const explicit = await decode(
      tokens,
      globalEmbedding,
      framesFor(4),
      TINY_DECODER,
      weights,
      cpuBackend,
    );
    const implied = await decode(tokens, globalEmbedding, framesFor(4), TINY_DECODER, weights);

    expect(Array.from(implied.waveform)).toEqual(Array.from(explicit.waveform));
  });
});

describe("the seams between stages", () => {
  it("hands the conv stage channel-major data, not length-major", async () => {
    // Random weights cannot see this: every axis ordering produces numbers of
    // the same shape, and the swap only changes which ones. With the upsample
    // made a per-channel identity, each channel's value is repeated into both
    // of its output positions — so the stage before it can be read back out of
    // the stage after it, and a transposed hand-off breaks the equality.
    const identity = new Weights(
      Safetensors.parse(buildTinyDecoderCheckpoint(TINY_DECODER, { identityUpsample: true })),
    );
    const tokens = tinyTokens(4);
    const channels = TINY_DECODER.prenet.outputDim!;

    const { stages } = await decode(
      tokens,
      globalEmbedding,
      framesFor(tokens.length),
      TINY_DECODER,
      identity,
      cpuBackend,
    );

    const before = stages.after_prenet!.data; // [tokens, channels]
    const after = stages.after_conv_upsample!.data; // [channels, 2 * tokens]
    for (let c = 0; c < channels; c += 1) {
      for (let t = 0; t < tokens.length; t += 1) {
        expect(after[c * 2 * tokens.length + 2 * t]).toBeCloseTo(before[t * channels + c]!, 5);
      }
    }
  });

  it("resamples to the requested frame count when the upsample does not land on it", async () => {
    // The conv upsample doubles the token count; anything else the caller asks
    // for is reached by linear interpolation. The real model needs it because
    // an utterance's frame count comes from the audio, not from a power of two.
    const identity = new Weights(
      Safetensors.parse(buildTinyDecoderCheckpoint(TINY_DECODER, { identityUpsample: true })),
    );
    const tokens = tinyTokens(4);
    const frames = 3 * tokens.length; // not 2x, so the interpolation actually runs
    const channels = TINY_DECODER.decoder.dim;

    const { stages } = await decode(
      tokens,
      globalEmbedding,
      frames,
      TINY_DECODER,
      identity,
      cpuBackend,
    );

    const source = stages.after_conv_upsample!.data;
    const sourceLength = 2 * tokens.length;
    const out = stages.after_interpolate!;
    expect(out.shape).toEqual([channels, frames]);

    for (let c = 0; c < channels; c += 1) {
      const row = Array.from(source.slice(c * sourceLength, (c + 1) * sourceLength));
      // Upsampling puts the first output sample exactly on the first input
      // sample, so this is an equality rather than a bound — and it is read at
      // `c * frames`, which is what pins the channel-major indexing.
      expect(out.data[c * frames]).toBeCloseTo(row[0]!, 6);
      // Every other output is a convex combination of two samples of THIS
      // channel, so it cannot leave the channel's own range. Reading a
      // neighbouring channel's data would.
      for (let t = 0; t < frames; t += 1) {
        expect(out.data[c * frames + t]!).toBeGreaterThanOrEqual(Math.min(...row) - 1e-6);
        expect(out.data[c * frames + t]!).toBeLessThanOrEqual(Math.max(...row) + 1e-6);
      }
    }
  });

  it("splits the head's output into log-magnitude and phase, in that order", async () => {
    // `chunk(2, dim=1)`: the first half of each row is log-magnitude, the
    // second is phase. Reading both halves from the same offsets would still
    // produce audio — of the wrong voice — so the relation is asserted against
    // the linear output the split was taken from.
    const tokens = tinyTokens(4);
    const frames = framesFor(tokens.length);
    const bins = TINY_DECODER.nFft / 2 + 1;

    const { stages } = await decode(
      tokens,
      globalEmbedding,
      frames,
      TINY_DECODER,
      weights,
      cpuBackend,
    );

    const linear = stages.istft_linear!.data;
    for (let t = 0; t < frames; t += 1) {
      for (let bin = 0; bin < bins; bin += 1) {
        const magnitude = Math.min(Math.exp(linear[t * 2 * bins + bin]!), 1e2);
        const phase = linear[t * 2 * bins + bins + bin]!;
        expect(stages.spec_real!.data[t * bins + bin]).toBeCloseTo(magnitude * Math.cos(phase), 5);
        expect(stages.spec_imag!.data[t * bins + bin]).toBeCloseTo(magnitude * Math.sin(phase), 5);
      }
    }
  });
});

describe("the AdaLN-conditioned stack", () => {
  it("is the identity when its projection is zero", async () => {
    // AdaLN-Zero's `x_norm * (1 + scale) + shift` with a zero-initialised
    // projection has to reduce to plain LayerNorm — that is what makes an
    // untrained model start as the identity. Dropping the `1 +` scales
    // everything to zero instead, which is invisible in any test that only
    // looks at shapes or at whether two runs differ.
    const { transformer } = await import("../../../../src/engine/miotts/codec/decoder.js");
    const zeroed = new Weights(
      Safetensors.parse(buildTinyDecoderCheckpoint(TINY_DECODER, { zeroResidualBranches: true })),
    );
    const dim = TINY_DECODER.decoder.dim;
    const rows = 3;
    const input = {
      data: Float32Array.from({ length: rows * dim }, (_, i) => (i % 5) - 2),
      shape: [rows, dim],
    };

    const out = await transformer(
      input,
      TINY_DECODER.decoder,
      "wave_decoder",
      zeroed,
      globalEmbedding,
      cpuBackend,
    );

    const identityNorm = layerNorm(
      input,
      { data: new Float32Array(dim).fill(1), shape: [dim] },
      { data: new Float32Array(dim), shape: [dim] },
      dim,
    );
    expect(Array.from(out.data)).toEqual(Array.from(identityNorm.data));
    // And it is not the all-zero output a missing `1 +` would produce.
    expect(out.data.some((value) => value !== 0)).toBe(true);
  });

  it("refuses to run without a condition", async () => {
    // `wave_decoder` is conditioned on the speaker; running it unconditioned
    // would be a different model, not a degraded one.
    const { transformer } = await import("../../../../src/engine/miotts/codec/decoder.js");
    const dim = TINY_DECODER.decoder.dim;

    await expect(
      transformer(
        { data: new Float32Array(2 * dim), shape: [2, dim] },
        TINY_DECODER.decoder,
        "wave_decoder",
        weights,
        null,
        cpuBackend,
      ),
    ).rejects.toThrow(/needs a condition/);
  });
});

describe("fsqDecode", () => {
  it("maps a codebook index to the embedding of its per-dimension codes", async () => {
    // `(index // basis) % levels`, first dimension varying fastest. Index 5
    // over levels [4, 4] is code (1, 1); index 1 is (1, 0). They share their
    // first code and differ in the second, which is what pins the basis order.
    const one = await fsqDecode(Float32Array.from([1]), TINY_DECODER.fsqLevels, weights, cpuBackend);
    const five = await fsqDecode(Float32Array.from([5]), TINY_DECODER.fsqLevels, weights, cpuBackend);
    const four = await fsqDecode(Float32Array.from([4]), TINY_DECODER.fsqLevels, weights, cpuBackend);

    expect(Array.from(one.data)).not.toEqual(Array.from(five.data));
    expect(Array.from(four.data)).not.toEqual(Array.from(one.data));
  });

  it("emits one embedding row per token", async () => {
    const out = await fsqDecode(tinyTokens(6), TINY_DECODER.fsqLevels, weights, cpuBackend);

    expect(out.shape).toEqual([6, TINY_DECODER.prenet.dim]);
  });
});

describe("transpose2d", () => {
  it("turns row-major into column-major", () => {
    expect(Array.from(transpose2d(Float32Array.from([1, 2, 3, 4, 5, 6]), 2, 3))).toEqual([
      1, 4, 2, 5, 3, 6,
    ]);
  });

  it("is its own inverse when applied with the swapped dimensions", () => {
    const original = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8]);

    expect(Array.from(transpose2d(transpose2d(original, 2, 4), 4, 2))).toEqual(
      Array.from(original),
    );
  });
});

describe("layerNorm", () => {
  it("centres and scales each row independently", () => {
    const x = { data: Float32Array.from([1, 2, 3, 100, 200, 300]), shape: [2, 3] };
    const weight = { data: Float32Array.from([1, 1, 1]), shape: [3] };
    const bias = { data: Float32Array.from([0, 0, 0]), shape: [3] };

    const out = layerNorm(x, weight, bias, 3);

    // Each row has mean 0 and the same normalised shape, however different
    // their scales were.
    for (const row of [0, 1]) {
      const values = Array.from(out.data.slice(row * 3, row * 3 + 3));
      expect(values.reduce((a, b) => a + b, 0)).toBeCloseTo(0, 5);
    }
    expect(out.data[0]).toBeCloseTo(out.data[3]!, 4);
  });
});

describe("MIOCODEC_24K", () => {
  it("carries the 24 kHz rung's own config", () => {
    // Pinned because everything downstream — the 25 Hz token rate, the 2
    // frames per token, the 960 samples per token — follows from these.
    expect(MIOCODEC_24K.sampleRate).toBe(24_000);
    expect(MIOCODEC_24K.nFft).toBe(1920);
    expect(MIOCODEC_24K.hopLength).toBe(480);
    expect(MIOCODEC_24K.fsqLevels).toEqual([8, 8, 8, 5, 5]);
    expect(MIOCODEC_24K.decoder.adaLnConditionDim).toBe(128);
  });
});
