import { ACTIVATION, activation } from "web-xpu-ops/ops/activation";
import { rmsnorm } from "web-xpu-ops/ops/rmsnorm";

import { linear } from "./blocks.js";

/**
 * What turns the two encoders' output into what the DiT attends to.
 *
 * Both the text and the caption path are the same module —
 * `PretrainedConditionProjector` — over the same ModernBERT-ja backbone, with
 * their own weights. Neither is a second transformer, which is easy to assume
 * from the name `caption_encoder`: 6.8 MB each against the backbone's 1.26 GB.
 *
 * The checkpoint's `pretrained_projector_type` is `residual_mlp`, so the
 * projection is a plain linear map **plus** a zero-initialised residual branch
 * that training brought online. The `linear` branch exists in the reference and
 * this checkpoint does not take it.
 */

export interface ProjectorWeights {
  projector: Float32Array;
  projectorBias: Float32Array;
  residualNorm: Float32Array;
  residualUp: Float32Array;
  residualUpBias: Float32Array;
  residualDown: Float32Array;
  residualDownBias: Float32Array;
  inDim: number;
  outDim: number;
  hidden: number;
}

/**
 * `projector(state) + residual_down(silu(residual_up(rmsnorm(state))))`, then
 * masked.
 *
 * The masking at the end is not redundant with the backbone's own: the residual
 * branch has biases, so a padded row that arrived as zeros leaves this as
 * `bias_down + projector_bias`, which is not zero and would be a real key for
 * the DiT to attend to.
 */
export function project(
  state: Float32Array,
  keep: boolean[],
  w: ProjectorWeights,
  eps: number,
): Float32Array {
  const rows = keep.length;
  const out = linear(state, w.projector, rows, w.inDim, w.outDim, w.projectorBias);

  const normed = rmsnorm({ input: state, weight: w.residualNorm, N: rows, D: w.inDim, eps });
  const up = activation({
    input: linear(normed, w.residualUp, rows, w.inDim, w.hidden, w.residualUpBias),
    kind: ACTIVATION.silu,
  });
  const down = linear(up, w.residualDown, rows, w.hidden, w.outDim, w.residualDownBias);
  for (let i = 0; i < out.length; i += 1) out[i]! += down[i]!;

  for (let row = 0; row < rows; row += 1) {
    if (!keep[row]) out.fill(0, row * w.outDim, (row + 1) * w.outDim);
  }
  return out;
}

/** RMSNorm with a learned scale — `text_norm`, `speaker_norm`, `caption_norm`. */
export function conditionNorm(
  state: Float32Array,
  weight: Float32Array,
  rows: number,
  dim: number,
  eps: number,
): Float32Array {
  return rmsnorm({ input: state, weight, N: rows, D: dim, eps });
}

/**
 * Prepend one summary token: the masked mean over time.
 *
 * This is why the speaker encoder's 123 frames reach the DiT as 124. Its mask
 * flag is `mask.any()` — true when anything at all was kept — so the summary
 * survives even where every individual frame would not.
 */
export function prependMeanToken(
  state: Float32Array,
  keep: boolean[],
  dim: number,
): { state: Float32Array; keep: boolean[] } {
  const rows = keep.length;
  const mean = new Float32Array(dim);
  let kept = 0;
  for (let row = 0; row < rows; row += 1) {
    if (!keep[row]) continue;
    kept += 1;
    for (let d = 0; d < dim; d += 1) mean[d]! += state[row * dim + d]!;
  }
  // `clamp_min(1.0)`: an all-masked reference divides by one rather than by
  // zero, and the mean it produces is the zero vector.
  const denom = Math.max(kept, 1);
  for (let d = 0; d < dim; d += 1) mean[d]! /= denom;

  const out = new Float32Array((rows + 1) * dim);
  out.set(mean, 0);
  out.set(state, dim);
  return { state: out, keep: [kept > 0, ...keep] };
}
