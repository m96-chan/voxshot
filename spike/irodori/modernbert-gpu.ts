import { ACTIVATION } from "web-xpu-ops/ops/activation";

import type { BertWeights } from "./bert-weights.js";
import { filled } from "./dit-gpu.js";
import type { Gpu, Tensor } from "./gpu.js";

/**
 * ModernBERT-ja on the device — the same graph as `modernbert.ts`.
 *
 * That file is the definition of correct and `check-bert.ts` compares against
 * the reference; this one exists because it was **six seconds of every
 * request**, the largest single item in the server's breakdown and larger than
 * the twelve-block DiT it sits in front of.
 *
 * ## Splits are views, not copies
 *
 * `Wqkv` produces one `[tokens, 3 * hidden]` tensor and the GeGLU one
 * `[tokens, 2 * intermediate]`. Both need slicing, and both are laid out
 * token-major — token `t`'s three projections are adjacent, so a slice is not
 * contiguous. `permute` reorders `[tokens, 3, hidden]` to `[3, tokens, hidden]`
 * in one dispatch, and then each part *is* contiguous and can be bound with an
 * offset. One dispatch instead of three copies, and no new kernel.
 *
 * ## What is not here
 *
 * The Q/K interleave that lets web-xpu-ops' adjacent-lane `rope` stand in for
 * HF's split-half one. `bert-weights.ts` folds it into the weights at load, so
 * both paths inherit it and neither has to know.
 *
 * The padding truncation is not here either, for the same reason: `realLength`
 * decides how many positions to run and the caller passes only those. On a
 * seven-token sentence that is 36x less work than the 256 the reference pads
 * to, and it is exact.
 */

const ADD = 0;
const MULTIPLY = 1;

export interface GpuBertLayer {
  /** Absent on layer 0, where the reference uses `nn.Identity`. */
  attnNorm: Tensor | undefined;
  wqkv: Tensor;
  wo: Tensor;
  mlpNorm: Tensor;
  mlpWi: Tensor;
  mlpWo: Tensor;
  ropeTheta: number;
  sliding: boolean;
}

export interface GpuBert {
  gpu: Gpu;
  weights: BertWeights;
  layers: GpuBertLayer[];
  tokenEmbeddings: Tensor;
  embeddingNorm: Tensor;
  finalNorm: Tensor;
  /** `norm_bias: false`, and the kernel's bias binding is not optional. */
  zeroBias: Tensor;
  /** One additive mask per attention pattern; 25 layers share two. */
  fullMask: Tensor;
  slidingMask: Tensor;
  ids: Tensor;
  tokens: number;
}

/**
 * Additive bias for one layer type, `[1, 1, tokens] x tokens`.
 *
 * Same predicate as the CPU path: a column is blocked where the key is padding,
 * and a sliding layer additionally blocks anything further than the window,
 * inclusive on both sides.
 */
function maskFor(keep: boolean[], window: number | null): Float32Array {
  const size = keep.length;
  const bias = new Float32Array(size * size);
  for (let i = 0; i < size; i += 1) {
    for (let j = 0; j < size; j += 1) {
      const blocked = !keep[j] || (window !== null && Math.abs(i - j) > window);
      bias[i * size + j] = blocked ? -Infinity : 0;
    }
  }
  return bias;
}

/**
 * Upload the weights once and the input per call.
 *
 * The masks and the ids depend on the sentence, so they are written into named
 * slots rather than allocated — a server renders many sentences and `alloc`
 * never frees.
 */
export function prepareBert(gpu: Gpu, weights: BertWeights, ids: Int32Array, keep: boolean[]): GpuBert {
  const { config } = weights;
  const tokens = keep.length;
  return {
    gpu,
    weights,
    layers: weights.layers.map((layer) => ({
      attnNorm: layer.attnNorm ? gpu.weight(layer.attnNorm) : undefined,
      wqkv: gpu.weight(layer.wqkv),
      wo: gpu.weight(layer.wo),
      mlpNorm: gpu.weight(layer.mlpNorm),
      mlpWi: gpu.weight(layer.mlpWi),
      mlpWo: gpu.weight(layer.mlpWo),
      ropeTheta: layer.ropeTheta,
      sliding: layer.sliding,
    })),
    tokenEmbeddings: gpu.weight(weights.tokenEmbeddings),
    embeddingNorm: gpu.weight(weights.embeddingNorm),
    finalNorm: gpu.weight(weights.finalNorm),
    // Memoised: a fresh array here would miss `Gpu.weight`'s cache and upload
    // again on every request.
    zeroBias: gpu.weight(filled(0, config.hiddenSize)),
    fullMask: gpu.writeInto("bert.mask.full", maskFor(keep, null)),
    slidingMask: gpu.writeInto("bert.mask.slide", maskFor(keep, config.slidingWindow)),
    ids: gpu.writeIntsInto("bert.ids", ids),
    tokens,
  };
}

/** `[tokens, hidden]` after `final_norm`, still on the device. */
export function runBertGpu(bert: GpuBert): Tensor {
  const { gpu, weights, tokens } = bert;
  const { config } = weights;
  const H = config.hiddenSize;
  const heads = config.numHeads;
  const headDim = H / heads;
  const I = config.intermediateSize;
  const eps = config.normEps;

  const norm = (input: Tensor, weight: Tensor, slot: string) =>
    gpu.layernorm(input, weight, bert.zeroBias, tokens, H, eps, slot);

  let hidden = norm(
    // `rows` is the table's row count; passing the sequence length instead
    // embeds nothing for every id above it, in silence.
    gpu.gather(bert.tokenEmbeddings, bert.ids, tokens, H, config.vocabSize, "bert.emb"),
    bert.embeddingNorm,
    "bert.emb.norm",
  );

  for (let index = 0; index < bert.layers.length; index += 1) {
    const layer = bert.layers[index]!;
    const at = (name: string) => `bert.l.${name}`;

    // Layer 0's `attn_norm` is `nn.Identity`, and the checkpoint has no tensor.
    const normed = layer.attnNorm ? norm(hidden, layer.attnNorm, at("an")) : hidden;

    // `[tokens, 3, hidden]` to `[3, tokens, hidden]`, so each projection is a
    // contiguous window rather than a copy.
    const qkv = gpu.permute(
      gpu.matmul(normed, layer.wqkv, tokens, 3 * H, H, at("qkv")),
      tokens,
      3,
      H,
      at("qkvp"),
    );
    const part = (which: number) => gpu.view(qkv, which * tokens * H, tokens * H);
    const rotate = (input: Tensor, slot: string) =>
      gpu.rope(input, tokens, heads, headDim, layer.ropeTheta, at(slot));

    const attended = gpu.attention({
      q: gpu.permute(rotate(part(0), "qr"), tokens, heads, headDim, at("qh")),
      k: gpu.permute(rotate(part(1), "kr"), tokens, heads, headDim, at("kh")),
      v: gpu.permute(part(2), tokens, heads, headDim, at("vh")),
      mask: layer.sliding ? bert.slidingMask : bert.fullMask,
      maskShape: [1, 1, tokens],
      B: 1,
      H: heads,
      L: tokens,
      S: tokens,
      D: headDim,
      scale: 1 / Math.sqrt(headDim),
      slot: at("attn"),
    });

    hidden = gpu.elementwise(
      hidden,
      gpu.matmul(
        gpu.permute(attended, heads, tokens, headDim, at("yh")),
        layer.wo,
        tokens,
        H,
        H,
        at("wo"),
      ),
      ADD,
      at("res1"),
    );

    // `Wi(x).chunk(2, -1)` names them `input, gate`; the activation goes on the
    // first half. Same permute trick to make each half contiguous.
    const wide = gpu.permute(
      gpu.matmul(norm(hidden, layer.mlpNorm, at("mn")), layer.mlpWi, tokens, 2 * I, H, at("wi")),
      tokens,
      2,
      I,
      at("wip"),
    );
    const gated = gpu.elementwise(
      gpu.activation(gpu.view(wide, 0, tokens * I), ACTIVATION.gelu, at("gelu")),
      gpu.view(wide, tokens * I, tokens * I),
      MULTIPLY,
      at("geglu"),
    );
    hidden = gpu.elementwise(hidden, gpu.matmul(gated, layer.mlpWo, tokens, H, I, at("wo2")), ADD, at("res2"));

    // The block writes into pooled slots, so its result is copied out before
    // the next layer reuses them.
    const kept = gpu.scratch(`bert.layer.${index % 2}`, hidden.length);
    gpu.copy(hidden, 0, kept, 0, hidden.length * 4);
    hidden = kept;
  }

  return norm(hidden, bert.finalNorm, "bert.final");
}
