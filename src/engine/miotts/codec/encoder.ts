import { activation, ACTIVATION } from "web-xpu-ops/ops/activation";
import { attention } from "web-xpu-ops/ops/attention";
import { conv1d, conv1dOutputLength } from "web-xpu-ops/ops/conv";
import { gather } from "web-xpu-ops/ops/gather";
import { groupNorm } from "web-xpu-ops/ops/group_norm";
import { softmax } from "web-xpu-ops/ops/softmax";
import {
  cpuBackend,
  layerNorm as layerNormTensor,
  linear as linearTensor,
  transpose2d,
  type Backend,
  type Tensor,
  type Weights,
} from "./decoder.js";

/**
 * MioCodec's encoder, global path only — reference audio to the 128-dim
 * speaker embedding — expressed in web-xpu-ops.
 *
 * A port of `MioCodecModel.encode(w, return_content=False, return_global=True)`
 * for the 24 kHz rung. The graph, and where each piece comes from:
 *
 *   24 kHz mono ─▶ symmetric zero-pad     `_calculate_waveform_padding`
 *               ─▶ resample 24k → 16k     polyphase `conv1d` [2, 1, 23] stride 3
 *               ─▶ WavLM conv frontend    7x `conv1d`, GroupNorm(512) after
 *                                         block 0 only, exact-erf GELU each
 *               ─▶ feature projection     LayerNorm(512) → Linear 512 → 768
 *               ─▶ pos_conv               grouped `conv1d` k128 g16, crop 1, GELU, +residual
 *               ─▶ encoder LayerNorm      `Transformer(layer_norm_first=not …)`:
 *                                         base+ is post-norm, so this LN runs
 *                                         *before* layer 1 — easy to miss
 *               ─▶ WavLM layers 1..2      post-norm, gated relative position bias
 *               ─▶ mean(layer1, layer2)   the global branch reads only these two
 *               ─▶ ConvNextBackbone       embed conv → LN → 4 blocks → final LN
 *               ─▶ AttentiveStatsPool     softmax over time; cat(mean, std)
 *               ─▶ Linear 768 → 128 → LayerNorm(128) ─▶ global_embedding
 *
 * The heavy ops — every `conv1d` (resample, WavLM frontend, pos_conv,
 * ConvNeXt embed and depthwise) and every projection `matmul` — go through the
 * decoder's {@link Backend} seam, defaulting to the reference `cpuBackend`;
 * the browser passes the WGSL-kernel backend instead, because the reference
 * frontend alone costs ~40 s per clip. Everything that is linear in its data
 * (norms, attention softmax, activations, the gather) stays on the reference
 * implementations regardless of backend, exactly as `decoder.ts` does. The
 * weights come from `export_encoder_weights.py`, which folds pos_conv's
 * weight-norm and precomputes the resample kernel so nothing here reimplements
 * either.
 */

/** Same contract as `spike/miotts`'s TraceFn: no trace, no copies, no cost. */
export type TraceFn = (stage: string, data: Float32Array) => void;

const SAMPLE_RATE = 24000;
const SSL_SAMPLE_RATE = 16000;

/**
 * torchaudio's polyphase constants for 24k → 16k: after dividing by the gcd the
 * rates are 3 and 2, and `transforms.Resample`'s windowed-sinc kernel is
 * `[2, 1, 23]` with `width = 10` — recorded in `encoder-weights.json` by the
 * export, and pinned here because `_apply_sinc_resample_kernel` pads
 * `(width, width + orig)` and strides by `orig`.
 */
const RESAMPLE_ORIG = 3;
const RESAMPLE_NEW = 2;
const RESAMPLE_WIDTH = 10;

/** WavLM base+'s conv frontend: `[kernel, stride]` per block, 512 channels out. */
const FRONTEND: readonly [number, number][] = [
  [10, 5],
  [3, 2],
  [3, 2],
  [3, 2],
  [3, 2],
  [2, 2],
  [2, 2],
];
const FRONTEND_DIM = 512;
/** Product of the frontend strides: one output frame per 320 input samples. */
const SSL_HOP = 320;

const EMBED_DIM = 768;
const NUM_HEADS = 12;
const HEAD_DIM = EMBED_DIM / NUM_HEADS;
/** Relative-position bias: 320 buckets (160 per direction), log-spaced past 40. */
const NUM_BUCKETS = 320;
const MAX_DISTANCE = 800;

const BACKBONE_DIM = 384;

/**
 * Hard cap on the input, enforced in {@link encodeGlobal} itself so no caller
 * can OOM it silently: WavLM attention is O(T²) — at 50 SSL frames per second
 * a 3-minute clip means T≈9000 and ~12·T² f32s of mask/bias/score scratch,
 * ~3.9 GB, a dead tab. 30 s (T≈1500, ~27 MB per T² array) is well past any
 * sane reference clip — the MioTTS server itself trims references to 20 s —
 * so anything longer is a caller bug, not a use case.
 */
const MAX_INPUT_SECONDS = 30;

const NORM_EPS = 1e-5;
/** ConvNeXt passes this explicitly; it is **not** torch's 1e-5 default. */
const CONVNEXT_EPS = 1e-6;

/* -------------------------------------------------------------------------- *
 * Small helpers
 * -------------------------------------------------------------------------- */

/**
 * `y = x @ W^T + b` over flat arrays — `decoder.ts`'s exported `linear` (one
 * implementation, one turn-once-per-weight cache), minus the Tensor wrapping
 * this file's call sites never wanted.
 */
async function linear(
  x: Float32Array,
  rows: number,
  weight: Tensor,
  bias: Tensor | null,
  backend: Backend,
): Promise<Float32Array> {
  const inFeatures = weight.shape[1]!;
  return (await linearTensor({ data: x, shape: [rows, inFeatures] }, weight, bias, backend)).data;
}

/** Exact-erf GELU — `nn.GELU()`'s default, which is what WavLM and ConvNeXt use. */
function gelu(x: Float32Array): Float32Array {
  return activation({ input: x, kind: ACTIVATION.gelu });
}

/** LayerNorm over the last axis with learned affine — `decoder.ts`'s, flat-array shaped. */
function layerNorm(x: Float32Array, weight: Tensor, bias: Tensor, dim: number, eps: number): Float32Array {
  return layerNormTensor({ data: x, shape: [x.length / dim, dim] }, weight, bias, dim, eps).data;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/* -------------------------------------------------------------------------- *
 * Padding and resampling
 * -------------------------------------------------------------------------- */

/**
 * `MioCodecModel._calculate_waveform_padding`: the zero samples added to each
 * side at 24 kHz so the SSL frontend's output length is `ceil(samples / hop)`.
 *
 * The float arithmetic mirrors the Python line for line — both run IEEE
 * float64, so `ceil` lands on the same side of every boundary.
 */
export function calculateWaveformPadding(audioLength: number): number {
  const afterResampling = (audioLength / SAMPLE_RATE) * SSL_SAMPLE_RATE;
  let length = Math.ceil(afterResampling / SSL_HOP);
  // `get_minimum_input_length`: walk the conv stack backwards.
  for (let i = FRONTEND.length - 1; i >= 0; i -= 1) {
    const [kernel, stride] = FRONTEND[i]!;
    length = (length - 1) * stride + kernel;
  }
  const required = (length / SSL_SAMPLE_RATE) * SAMPLE_RATE;
  return Math.ceil((required - audioLength) / 2);
}

/**
 * `torchaudio.functional._apply_sinc_resample_kernel` for 24k → 16k: pad
 * `(width, width + orig)`, conv with the precomputed `[new, 1, K]` kernel at
 * stride `orig`, interleave the phase channels back into time order, trim to
 * `ceil(new * L / orig)`.
 */
async function resample(waveform: Float32Array, kernel: Tensor, backend: Backend): Promise<Float32Array> {
  const K = kernel.shape[2]!;
  const padded = new Float32Array(RESAMPLE_WIDTH + waveform.length + RESAMPLE_WIDTH + RESAMPLE_ORIG);
  padded.set(waveform, RESAMPLE_WIDTH);

  const phases = await backend.conv1d(
    padded,
    kernel.data,
    null,
    1,
    RESAMPLE_NEW,
    padded.length,
    K,
    0,
    RESAMPLE_ORIG,
  );
  const perPhase = phases.length / RESAMPLE_NEW;

  // `transpose(1, 2).reshape(-1)`: sample t comes from phase t % new.
  const target = Math.ceil((RESAMPLE_NEW * waveform.length) / RESAMPLE_ORIG);
  const out = new Float32Array(target);
  for (let t = 0; t < target; t += 1) {
    out[t] = phases[(t % RESAMPLE_NEW) * perPhase + Math.floor(t / RESAMPLE_NEW)]!;
  }
  return out;
}

/* -------------------------------------------------------------------------- *
 * WavLM
 * -------------------------------------------------------------------------- */

/** The conv frontend: `[1, L16]` in, `[T, 512]` out (time-major, like torch). */
async function featureExtractor(
  wave16: Float32Array,
  weights: Weights,
  backend: Backend,
): Promise<Float32Array> {
  let x = wave16;
  let length = wave16.length;
  let cin = 1;
  for (let i = 0; i < FRONTEND.length; i += 1) {
    const [kernel, stride] = FRONTEND[i]!;
    x = await backend.conv1d(
      x,
      weights.get(`wavlm.feature_extractor.conv_layers.${i}.conv.weight`).data,
      // bias-free throughout: `extractor_conv_bias = False` for the base arch.
      null,
      cin,
      FRONTEND_DIM,
      length,
      kernel,
      0,
      stride,
    );
    length = conv1dOutputLength({ L: length, K: kernel, stride });
    if (i === 0) {
      // "group_norm" extractor mode: one GroupNorm with as many groups as
      // channels — per-channel statistics over time — after block 0 only.
      x = groupNorm({
        input: x,
        weight: weights.get("wavlm.feature_extractor.conv_layers.0.layer_norm.weight").data,
        bias: weights.get("wavlm.feature_extractor.conv_layers.0.layer_norm.bias").data,
        N: 1,
        C: FRONTEND_DIM,
        L: length,
        G: FRONTEND_DIM,
        eps: NORM_EPS,
      });
    }
    x = gelu(x);
    cin = FRONTEND_DIM;
  }
  return transpose2d(x, FRONTEND_DIM, length); // [C, T] -> [T, C]
}

/** pos_conv: grouped conv over `[C, T]`, crop the extra sample, GELU. */
async function posConv(x: Float32Array, T: number, weights: Weights, backend: Backend): Promise<Float32Array> {
  const weight = weights.get("wavlm.encoder.transformer.pos_conv_embed.conv.weight");
  const K = weight.shape[2]!; // 128
  const convOut = await backend.conv1d(
    transpose2d(x, T, EMBED_DIM),
    weight.data,
    weights.get("wavlm.encoder.transformer.pos_conv_embed.conv.bias").data,
    EMBED_DIM,
    EMBED_DIM,
    T,
    K,
    K / 2,
    1,
    16,
  );
  // Even kernel with pad K/2 emits T + 1 samples; `num_remove = 1` crops the last.
  const cropped = new Float32Array(EMBED_DIM * T);
  for (let c = 0; c < EMBED_DIM; c += 1) {
    cropped.set(convOut.subarray(c * (T + 1), c * (T + 1) + T), c * T);
  }
  return transpose2d(gelu(cropped), EMBED_DIM, T);
}

/**
 * The relative-position bucket, `WavLMSelfAttention._relative_positions_bucket`
 * with `bidirectional=True`.
 *
 * The log branch is evaluated through `Math.fround` because torch runs it in
 * float32: a distance whose f32 product lands on the far side of an integer
 * from its f64 value would pick a different bucket, and "the bias is off in
 * row 217 only" is a miserable failure to chase.
 */
function relativePositionBucket(relativePosition: number): number {
  const halfBuckets = NUM_BUCKETS / 2; // bidirectional split
  const maxExact = halfBuckets / 2;
  let bucket = relativePosition > 0 ? halfBuckets : 0;
  const distance = Math.abs(relativePosition);
  if (distance < maxExact) return bucket + distance;
  const ifLarge =
    maxExact +
    Math.trunc(
      Math.fround(
        (Math.fround(Math.log(Math.fround(distance / maxExact))) / Math.log(MAX_DISTANCE / maxExact)) *
          (halfBuckets - maxExact),
      ),
    );
  return bucket + Math.min(ifLarge, halfBuckets - 1);
}

/** `compute_bias`: the ungated `[H, T, T]` bias, built once and reused by layer 2. */
function positionBias(T: number, weights: Weights): Float32Array {
  const table = weights.get("wavlm.encoder.transformer.layers.0.attention.rel_attn_embed.weight");
  const indices = new Int32Array(T * T);
  for (let i = 0; i < T; i += 1) {
    for (let j = 0; j < T; j += 1) {
      indices[i * T + j] = relativePositionBucket(j - i);
    }
  }
  const values = gather({ table: table.data, indices, rows: NUM_BUCKETS, D: NUM_HEADS });
  // [T*T, H] -> [H, T, T] (the `permute([2, 0, 1])`).
  const bias = new Float32Array(NUM_HEADS * T * T);
  for (let p = 0; p < T * T; p += 1) {
    for (let h = 0; h < NUM_HEADS; h += 1) {
      bias[h * T * T + p] = values[p * NUM_HEADS + h]!;
    }
  }
  return bias;
}

/**
 * One WavLM encoder layer, post-norm (`layer_norm_first=False` for base+):
 * `x = LN(x + Attn(x)); x = LN2(x + FFN(x))`.
 *
 * The position bias enters SDPA as an additive float mask, per-layer gated:
 * the gate reads the **pre-projection** input, per head, through a tiny
 * Linear(64 → 8) summed in fours and squashed — `gate_a * (gate_b * const - 1)
 * + 2`, one scalar per (head, query row), multiplied into the shared bias.
 */
async function wavlmLayer(
  x: Float32Array,
  T: number,
  layer: number,
  bias: Float32Array,
  weights: Weights,
  backend: Backend,
): Promise<Float32Array> {
  const prefix = `wavlm.encoder.transformer.layers.${layer}`;

  // -- gate. `x` as [T, 768] row-major is exactly [T * H, 64] in (t, h) order.
  const gate = await linear(
    x,
    T * NUM_HEADS,
    weights.get(`${prefix}.attention.gru_rel_pos_linear.weight`),
    weights.get(`${prefix}.attention.gru_rel_pos_linear.bias`),
    backend,
  );
  const gateConst = weights.get(`${prefix}.attention.gru_rel_pos_const`).data; // [1, H, 1, 1]
  const mask = new Float32Array(NUM_HEADS * T * T);
  for (let t = 0; t < T; t += 1) {
    for (let h = 0; h < NUM_HEADS; h += 1) {
      const row = (t * NUM_HEADS + h) * 8;
      let sumA = 0;
      let sumB = 0;
      for (let j = 0; j < 4; j += 1) {
        sumA += gate[row + j]!;
        sumB += gate[row + 4 + j]!;
      }
      const gateA = sigmoid(sumA);
      const gateB = sigmoid(sumB);
      const scale = gateA * (gateB * gateConst[h]! - 1) + 2;
      const base = (h * T + t) * T;
      for (let j = 0; j < T; j += 1) mask[base + j] = scale * bias[base + j]!;
    }
  }

  // -- qkv: one packed projection, chunked thirds.
  const qkv = await linear(
    x,
    T,
    weights.get(`${prefix}.attention.attention.in_proj_weight`),
    weights.get(`${prefix}.attention.attention.in_proj_bias`),
    backend,
  );
  const q = new Float32Array(T * EMBED_DIM);
  const k = new Float32Array(T * EMBED_DIM);
  const v = new Float32Array(T * EMBED_DIM);
  for (let t = 0; t < T; t += 1) {
    for (let h = 0; h < NUM_HEADS; h += 1) {
      for (let d = 0; d < HEAD_DIM; d += 1) {
        const to = (h * T + t) * HEAD_DIM + d; // [H, T, D] head-major
        const from = t * 3 * EMBED_DIM + h * HEAD_DIM + d;
        q[to] = qkv[from]!;
        k[to] = qkv[from + EMBED_DIM]!;
        v[to] = qkv[from + 2 * EMBED_DIM]!;
      }
    }
  }

  const { output: scores } = attention({
    q,
    k,
    v,
    B: 1,
    H: NUM_HEADS,
    L: T,
    S: T,
    D: HEAD_DIM,
    Dv: HEAD_DIM,
    // SDPA's own default, passed so the two cannot drift apart silently.
    scale: 1 / Math.sqrt(HEAD_DIM),
    mask,
    maskShape: [1, NUM_HEADS, T],
  });

  // [H, T, D] back to [T, H * D].
  const merged = new Float32Array(T * EMBED_DIM);
  for (let h = 0; h < NUM_HEADS; h += 1) {
    for (let t = 0; t < T; t += 1) {
      for (let d = 0; d < HEAD_DIM; d += 1) {
        merged[(t * NUM_HEADS + h) * HEAD_DIM + d] = scores[(h * T + t) * HEAD_DIM + d]!;
      }
    }
  }
  const attended = await linear(
    merged,
    T,
    weights.get(`${prefix}.attention.attention.out_proj.weight`),
    weights.get(`${prefix}.attention.attention.out_proj.bias`),
    backend,
  );

  for (let i = 0; i < attended.length; i += 1) attended[i] = attended[i]! + x[i]!;
  let out = layerNorm(
    attended,
    weights.get(`${prefix}.layer_norm.weight`),
    weights.get(`${prefix}.layer_norm.bias`),
    EMBED_DIM,
    NORM_EPS,
  );

  const hidden = gelu(
    await linear(
      out,
      T,
      weights.get(`${prefix}.feed_forward.intermediate_dense.weight`),
      weights.get(`${prefix}.feed_forward.intermediate_dense.bias`),
      backend,
    ),
  );
  const forwarded = await linear(
    hidden,
    T,
    weights.get(`${prefix}.feed_forward.output_dense.weight`),
    weights.get(`${prefix}.feed_forward.output_dense.bias`),
    backend,
  );
  for (let i = 0; i < forwarded.length; i += 1) forwarded[i] = forwarded[i]! + out[i]!;
  return layerNorm(
    forwarded,
    weights.get(`${prefix}.final_layer_norm.weight`),
    weights.get(`${prefix}.final_layer_norm.bias`),
    EMBED_DIM,
    NORM_EPS,
  );
}

/* -------------------------------------------------------------------------- *
 * Global encoder
 * -------------------------------------------------------------------------- */

/** ConvNeXt backbone: `[T, 768]` in, `[T, 384]` out; blocks traced channel-major. */
async function backbone(
  x: Float32Array,
  T: number,
  weights: Weights,
  backend: Backend,
  trace?: TraceFn,
): Promise<Float32Array> {
  let c = await backend.conv1d(
    transpose2d(x, T, EMBED_DIM),
    weights.get("global_encoder.backbone.embed.weight").data,
    weights.get("global_encoder.backbone.embed.bias").data,
    EMBED_DIM,
    BACKBONE_DIM,
    T,
    7,
    3,
  );
  c = transpose2d(
    layerNorm(
      transpose2d(c, BACKBONE_DIM, T),
      weights.get("global_encoder.backbone.norm.weight"),
      weights.get("global_encoder.backbone.norm.bias"),
      BACKBONE_DIM,
      CONVNEXT_EPS,
    ),
    T,
    BACKBONE_DIM,
  );

  for (let block = 0; block < 4; block += 1) {
    const prefix = `global_encoder.backbone.convnext.${block}`;
    // Depthwise = groups 384, still ONE dispatch: the WGSL kernel takes the
    // per-group channel counts in its uniform, so no per-group slicing here.
    const depthwise = await backend.conv1d(
      c,
      weights.get(`${prefix}.dwconv.weight`).data,
      weights.get(`${prefix}.dwconv.bias`).data,
      BACKBONE_DIM,
      BACKBONE_DIM,
      T,
      7,
      3,
      1,
      BACKBONE_DIM,
    );
    let h = layerNorm(
      transpose2d(depthwise, BACKBONE_DIM, T),
      weights.get(`${prefix}.norm.weight`),
      weights.get(`${prefix}.norm.bias`),
      BACKBONE_DIM,
      CONVNEXT_EPS,
    );
    h = gelu(
      await linear(h, T, weights.get(`${prefix}.pwconv1.weight`), weights.get(`${prefix}.pwconv1.bias`), backend),
    );
    h = await linear(h, T, weights.get(`${prefix}.pwconv2.weight`), weights.get(`${prefix}.pwconv2.bias`), backend);

    const gamma = weights.get(`${prefix}.gamma`).data;
    const next = new Float32Array(c.length);
    for (let ch = 0; ch < BACKBONE_DIM; ch += 1) {
      for (let t = 0; t < T; t += 1) {
        next[ch * T + t] = c[ch * T + t]! + gamma[ch]! * h[t * BACKBONE_DIM + ch]!;
      }
    }
    c = next;
    trace?.(`convnext_block${block + 1}`, c.slice());
  }

  return layerNorm(
    transpose2d(c, BACKBONE_DIM, T),
    weights.get("global_encoder.backbone.final_layer_norm.weight"),
    weights.get("global_encoder.backbone.final_layer_norm.bias"),
    BACKBONE_DIM,
    CONVNEXT_EPS,
  );
}

/**
 * AttentiveStatsPool over `[C, T]`: α-weighted mean and std, concatenated.
 *
 * Its two k=1 convs stay on the reference implementation whatever backend
 * runs the rest: ~15 MFLOP between them, less than a GPU round-trip costs.
 */
function attentiveStatsPool(x: Float32Array, T: number, weights: Weights, trace?: TraceFn): Float32Array {
  const attnDim = weights.get("global_encoder.pooling.attn.0.weight").shape[0]!; // 128
  const scores = activation({
    input: conv1d({
      input: x,
      weight: weights.get("global_encoder.pooling.attn.0.weight").data,
      bias: weights.get("global_encoder.pooling.attn.0.bias").data,
      N: 1,
      Cin: BACKBONE_DIM,
      Cout: attnDim,
      L: T,
      K: 1,
    }),
    kind: ACTIVATION.tanh,
  });
  const logits = conv1d({
    input: scores,
    weight: weights.get("global_encoder.pooling.attn.2.weight").data,
    bias: weights.get("global_encoder.pooling.attn.2.bias").data,
    N: 1,
    Cin: attnDim,
    Cout: BACKBONE_DIM,
    L: T,
    K: 1,
  });
  // `Softmax(dim=2)` on [B, C, T]: over *time* — the axis a port gets wrong.
  const alpha = softmax({ input: logits, N: BACKBONE_DIM, D: T });
  trace?.("attn_weights", alpha.slice());

  const pooled = new Float32Array(2 * BACKBONE_DIM);
  for (let c = 0; c < BACKBONE_DIM; c += 1) {
    let mean = 0;
    let meanOfSquares = 0;
    for (let t = 0; t < T; t += 1) {
      const value = x[c * T + t]!;
      mean += alpha[c * T + t]! * value;
      meanOfSquares += alpha[c * T + t]! * value * value;
    }
    const variance = Math.min(Math.max(meanOfSquares - mean * mean, 1e-4), 1e4);
    pooled[c] = mean;
    pooled[BACKBONE_DIM + c] = Math.sqrt(variance);
  }
  return pooled;
}

/* -------------------------------------------------------------------------- *
 * The whole encoder
 * -------------------------------------------------------------------------- */

export async function encodeGlobal(
  waveform24k: Float32Array,
  weights: Weights,
  opts?: { backend?: Backend; trace?: TraceFn },
): Promise<Float32Array> {
  const backend = opts?.backend ?? cpuBackend;
  const trace = opts?.trace;

  if (waveform24k.length > MAX_INPUT_SECONDS * SAMPLE_RATE) {
    throw new Error(
      `encodeGlobal: ${(waveform24k.length / SAMPLE_RATE).toFixed(1)} s of audio is over the ` +
        `${MAX_INPUT_SECONDS} s cap — WavLM attention is O(T²) and a long clip allocates ` +
        `gigabytes of scratch. Trim the reference before encoding.`,
    );
  }

  trace?.("waveform_24k", waveform24k.slice());

  // -- pad + resample. The pad is at 24 kHz, before the rate change, which is
  // why `after_resample` checks both at once.
  const padding = calculateWaveformPadding(waveform24k.length);
  const padded = new Float32Array(waveform24k.length + 2 * padding);
  padded.set(waveform24k, padding);
  const wave16 = await resample(padded, weights.get("resample.kernel"), backend);
  trace?.("after_resample", wave16.slice());

  // -- WavLM frontend and projection.
  const features = await featureExtractor(wave16, weights, backend);
  const T = features.length / FRONTEND_DIM;
  trace?.("after_feature_extractor", features.slice());

  const projected = await linear(
    layerNorm(
      features,
      weights.get("wavlm.encoder.feature_projection.layer_norm.weight"),
      weights.get("wavlm.encoder.feature_projection.layer_norm.bias"),
      FRONTEND_DIM,
      NORM_EPS,
    ),
    T,
    weights.get("wavlm.encoder.feature_projection.projection.weight"),
    weights.get("wavlm.encoder.feature_projection.projection.bias"),
    backend,
  );
  trace?.("after_feature_projection", projected.slice());

  // -- positional conv, residual add, then the encoder-level LayerNorm: the
  // Transformer is built with `layer_norm_first = not layer_norm_first`, so
  // for post-norm base+ this LN runs here, before layer 1.
  const positional = await posConv(projected, T, weights, backend);
  trace?.("after_pos_conv", positional.slice());
  for (let i = 0; i < projected.length; i += 1) positional[i] = positional[i]! + projected[i]!;
  let x = layerNorm(
    positional,
    weights.get("wavlm.encoder.transformer.layer_norm.weight"),
    weights.get("wavlm.encoder.transformer.layer_norm.bias"),
    EMBED_DIM,
    NORM_EPS,
  );

  // -- layers 1..2. The bias is computed once at layer 1 and reused, ungated;
  // each layer applies its own gate to its own copy.
  const bias = positionBias(T, weights);
  const layer1 = await wavlmLayer(x, T, 0, bias, weights, backend);
  trace?.("ssl_layer1", layer1.slice());
  const layer2 = await wavlmLayer(layer1, T, 1, bias, weights, backend);
  trace?.("ssl_layer2", layer2.slice());

  // -- the global branch reads the mean of the two, un-normalised (the z-norm
  // in `forward_ssl_features` is local-branch only).
  const globalInput = new Float32Array(layer1.length);
  for (let i = 0; i < layer1.length; i += 1) globalInput[i] = (layer1[i]! + layer2[i]!) / 2;
  trace?.("global_input", globalInput.slice());

  // -- ConvNeXt backbone and attentive stats pooling.
  const pooledInput = await backbone(globalInput, T, weights, backend, trace);
  trace?.("after_backbone", pooledInput.slice());

  const pooled = attentiveStatsPool(transpose2d(pooledInput, T, BACKBONE_DIM), T, weights, trace);
  trace?.("pooled_stats", pooled.slice());

  const embedding = layerNorm(
    await linear(
      pooled,
      1,
      weights.get("global_encoder.pooling.proj.weight"),
      weights.get("global_encoder.pooling.proj.bias"),
      backend,
    ),
    weights.get("global_encoder.pooling.norm.weight"),
    weights.get("global_encoder.pooling.norm.bias"),
    128,
    NORM_EPS,
  );
  trace?.("global_embedding", embedding.slice());
  return embedding;
}
