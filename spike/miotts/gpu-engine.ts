import { ACTIVATION } from "web-xpu-ops/ops/activation";
import { ELEMENTWISE } from "web-xpu-ops/ops/elementwise";
import {
  ACTIVATION as ACTIVATION_WGSL,
  ELEMENTWISE as ELEMENTWISE_WGSL,
  GQA_CONTEXT,
  GQA_SCORES,
  MATVEC_Q8,
  RMSNORM,
  ROPE,
} from "./kernels.js";
import { gatherDequantRow, type PackedQ8, type Qwen3WeightsQ8 } from "./weights-q8.js";

/**
 * A WebGPU-resident Qwen3 q8 decode engine for MioTTS-0.6B.
 *
 * Structure mirrors web-xpu-ops' `llm/engine-q8-resident.ts` (read, not
 * re-derived — that class already solved the problems this one would
 * otherwise rediscover): every weight buffer, pipeline, bind group and
 * activation buffer is built once in `createGpuEngine`; one decode step is
 * one flat op list — ~480 dispatches plus the per-layer KV-cache
 * `copyBufferToBuffer` appends — recorded into one `GPUCommandEncoder`,
 * submitted once, with exactly one readback: the final logits
 * (`vocabSize * 4` = 657,920 bytes across three staging buffers). The decode
 * loop allocates no GPU resources; `stats` proves it the same way upstream's
 * do.
 *
 * Differences from the upstream llama engine, all Qwen3-specific:
 *
 * - **QK-norm**: q/k get a per-head RMSNorm (one shared `[headDim]` gamma)
 *   after projection, before RoPE. Two extra `rmsnorm` dispatches per layer,
 *   `N = numHeads` (or `numKvHeads`) rows of `D = headDim` — the q8
 *   artifacts' gammas are ALREADY permuted into ops/rope adjacent-pair
 *   channel order (`weights-q8.ts`'s `ropePermuted` contract), so they bind
 *   as-is.
 * - **No fused QKV**: separate wq/wk/wv projections, each its own buffer —
 *   upstream's own alignment reasoning (`harness/resident.ts#bindGroup`'s
 *   doc: offset-sliced fused buffers break on
 *   `minStorageBufferOffsetAlignment`) applies unchanged.
 * - **Tied lm_head**: the logits matvec reads `embedTokens` itself, packed,
 *   chunked at 65,535 rows (`maxComputeWorkgroupsPerDimension`) into three
 *   separate weight/scale/out buffers — 164,480 rows = 65,535 + 65,535 +
 *   33,410 — because a chunk boundary offset into one buffer is not
 *   256-byte-aligned either (upstream's `lmHeadChunks` reasoning verbatim).
 * - **Embedding gather on CPU**: `gatherDequantRow` unpacks one 1024-f32 row
 *   per token and `queue.writeBuffer`s it — same as upstream; 4 KiB per step
 *   is noise next to the 600 MiB of weights each step streams.
 *
 * **Prefill is the decode path, token by token.** The MioTTS prompt is ~15
 * tokens; 15 extra submits cost a few ms total on hardware, so a batched
 * `matmul` prefill path (what upstream's `runPrefillResident` exists for at
 * 360-token prompts) would be complexity with no measurable payoff here.
 * The caller simply feeds the prompt ids through `decodeStep` and discards
 * every logits vector but the last.
 *
 * Platform-neutral: takes an already-created `GPUDevice`. A browser caller
 * must request the adapter's own `maxStorageBufferBindingSize` /
 * `maxBufferSize` (the embedTokens codes alone are ~168 MiB packed, and the
 * largest single chunk buffer is ~67 MiB — over some adapters' 128 MiB
 * default budget when everything else is counted) — see
 * `spike/miocodec/gpu.ts#Gpu.create` for the mirror.
 */

/** `maxComputeWorkgroupsPerDimension` — copied from web-xpu-ops `llm/kernels.ts`, not re-derived. */
export const MAX_WORKGROUPS_PER_DISPATCH = 65535;

/** `ROPE`'s uniform field 3 is `pos_offset` (u32) — see kernels.ts' contract comment. */
const ROPE_POS_OFFSET_BYTE = 3 * 4;
/** `GQA_SCORES`' field 7 is `query_offset` (i32). */
const GQA_QUERY_OFFSET_BYTE = 7 * 4;
/** `GQA_SCORES`' field 11 is `s_eff` (u32). */
const GQA_SCORES_S_EFF_BYTE = 11 * 4;
/** `GQA_CONTEXT`'s field 5 is `s_eff` (u32). */
const GQA_CONTEXT_S_EFF_BYTE = 5 * 4;

type UniformField = ["u32" | "i32" | "f32", number];

/** `harness/wgsl.ts#params`' packing: 4 bytes per field, little-endian, min 16 bytes. */
function packFields(fields: UniformField[]): ArrayBuffer {
  const buffer = new ArrayBuffer(Math.max(16, fields.length * 4));
  const view = new DataView(buffer);
  fields.forEach(([kind, value], index) => {
    if (kind === "f32") view.setFloat32(index * 4, value, true);
    else if (kind === "i32") view.setInt32(index * 4, value, true);
    else view.setUint32(index * 4, value, true);
  });
  return buffer;
}

type Workgroups = [number] | [number, number] | [number, number, number];

type Op =
  | { kind: "dispatch"; pipeline: GPUComputePipeline; bindGroup: GPUBindGroup; workgroups: Workgroups }
  | { kind: "copy"; src: GPUBuffer; srcOffset: number; dst: GPUBuffer; dstOffset: number; size: number };

export interface GpuEngineStats {
  /** GPU buffers created since construction started — must not grow inside the decode loop. */
  buffersCreated: number;
  submits: number;
  /** Compute dispatches recorded per decode step (constant). */
  dispatchesPerStep: number;
  /** KV-cache copyBufferToBuffer ops recorded per decode step (constant). */
  copiesPerStep: number;
  /** Bytes read back per decode step: the logits, nothing else. */
  readbackBytesPerStep: number;
}

export interface GpuEngine {
  /**
   * One token through all 28 layers: one submit, logits-only readback.
   * Prefill = call this once per prompt token and keep only the last logits.
   */
  decodeStep(tokenId: number): Promise<Float32Array>;
  /** Positions already resident in the KV cache. */
  readonly position: number;
  /** Start a new generation: rewinds the position; the KV cache is overwritten in place. */
  reset(): void;
  readonly stats: GpuEngineStats;
  /** Frees the device-side weights and caches. The engine is unusable afterwards. */
  destroy(): void;
}

interface LayerResident {
  kCacheBuf: GPUBuffer;
  vCacheBuf: GPUBuffer;
  attnNormGroup: GPUBindGroup;
  ffnNormGroup: GPUBindGroup;
  wqGroup: GPUBindGroup;
  wkGroup: GPUBindGroup;
  wvGroup: GPUBindGroup;
  qNormGroup: GPUBindGroup;
  kNormGroup: GPUBindGroup;
  scoresGroup: GPUBindGroup;
  contextGroup: GPUBindGroup;
  woGroup: GPUBindGroup;
  gateGroup: GPUBindGroup;
  upGroup: GPUBindGroup;
  downGroup: GPUBindGroup;
}

interface LmHeadChunk {
  rowCount: number;
  group: GPUBindGroup;
  outBuf: GPUBuffer;
  staging: GPUBuffer;
}

export async function createGpuEngine(
  device: GPUDevice,
  weights: Qwen3WeightsQ8,
  opts: { maxSeqLen: number },
): Promise<GpuEngine> {
  const cfg = weights.config;
  const { numLayers, hiddenSize, numHeads, numKvHeads, headDim, ffnHidden, vocabSize, ropeTheta, rmsNormEps } = cfg;
  const { maxSeqLen } = opts;
  const qDim = numHeads * headDim;
  const kvDim = numKvHeads * headDim;

  const stats: GpuEngineStats = {
    buffersCreated: 0,
    submits: 0,
    dispatchesPerStep: 0,
    copiesPerStep: 0,
    readbackBytesPerStep: vocabSize * 4,
  };

  // ---- ResidentDevice-shaped helpers, inlined against the raw GPUDevice so
  // this module bundles for the browser (harness/resident.ts imports the
  // Node-native `webgpu` package and cannot). Direct ports of that file's
  // `pipelineFor`/`bindGroup`/`batch`, error scopes included: `layout:
  // "auto"` failures otherwise surface as silent zero outputs (upstream's
  // issue #46 / PR #116 history). ----

  const pipelines = new Map<string, GPUComputePipeline>();
  const modules = new Map<string, GPUShaderModule>();

  function createStorageBuffer(bytes: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC): GPUBuffer {
    stats.buffersCreated += 1;
    return device.createBuffer({ size: Math.max(4, bytes), usage });
  }

  function upload(buffer: GPUBuffer, offset: number, data: ArrayBufferView): void {
    device.queue.writeBuffer(buffer, offset, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
  }

  function uniformOf(fields: UniformField[]): GPUBuffer {
    stats.buffersCreated += 1;
    const data = packFields(fields);
    const buffer = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }

  async function pipelineFor(code: string): Promise<GPUComputePipeline> {
    const cached = pipelines.get(code);
    if (cached) return cached;
    let module = modules.get(code);
    if (!module) {
      module = device.createShaderModule({ code });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === "error");
      if (errors.length > 0) {
        throw new Error(`shader failed to compile\n${errors.map((m) => `${m.lineNum}:${m.linePos}: ${m.message}`).join("\n")}`);
      }
      modules.set(code, module);
    }
    device.pushErrorScope("validation");
    const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
    const invalid = await device.popErrorScope();
    if (invalid) throw new Error(`pipeline is not valid: ${invalid.message}`);
    pipelines.set(code, pipeline);
    return pipeline;
  }

  async function bindGroup(pipeline: GPUComputePipeline, buffers: GPUBuffer[]): Promise<GPUBindGroup> {
    device.pushErrorScope("validation");
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const invalid = await device.popErrorScope();
    if (invalid) throw new Error(`bind group is not valid: ${invalid.message}`);
    return group;
  }

  /** Uploads one packed q8 weight (or a row-slice of one) as its own weight+scale buffers. */
  function uploadPacked(w: { packed: Uint32Array; scale: Float32Array }): { weightBuf: GPUBuffer; scaleBuf: GPUBuffer } {
    const weightBuf = createStorageBuffer(w.packed.byteLength);
    upload(weightBuf, 0, w.packed);
    const scaleBuf = createStorageBuffer(w.scale.byteLength);
    upload(scaleBuf, 0, w.scale);
    return { weightBuf, scaleBuf };
  }

  const [matvecPipeline, rmsnormPipeline, ropePipeline, scoresPipeline, contextPipeline, activationPipeline, elementwisePipeline] =
    await Promise.all([
      pipelineFor(MATVEC_Q8),
      pipelineFor(RMSNORM),
      pipelineFor(ROPE),
      pipelineFor(GQA_SCORES),
      pipelineFor(GQA_CONTEXT),
      pipelineFor(ACTIVATION_WGSL),
      pipelineFor(ELEMENTWISE_WGSL),
    ]);

  // ---- N = 1 activation buffers, shared by all layers and every step. ----
  const hiddenA = createStorageBuffer(hiddenSize * 4); // residual in
  const hiddenB = createStorageBuffer(hiddenSize * 4); // residual after attention
  const normedBuf = createStorageBuffer(hiddenSize * 4);
  const normed2Buf = createStorageBuffer(hiddenSize * 4);
  const qProjBuf = createStorageBuffer(qDim * 4);
  const kProjBuf = createStorageBuffer(kvDim * 4);
  const vProjBuf = createStorageBuffer(kvDim * 4);
  const qNormedBuf = createStorageBuffer(qDim * 4);
  const kNormedBuf = createStorageBuffer(kvDim * 4);
  const qRopedBuf = createStorageBuffer(qDim * 4);
  const kRopedBuf = createStorageBuffer(kvDim * 4);
  const attnOutBuf = createStorageBuffer(qDim * 4);
  const projOutBuf = createStorageBuffer(hiddenSize * 4);
  const gateOutBuf = createStorageBuffer(ffnHidden * 4);
  const upOutBuf = createStorageBuffer(ffnHidden * 4);
  const gateActBuf = createStorageBuffer(ffnHidden * 4);
  const gatedBuf = createStorageBuffer(ffnHidden * 4);
  const downOutBuf = createStorageBuffer(hiddenSize * 4);
  const finalNormedBuf = createStorageBuffer(hiddenSize * 4);
  // ROPE's cache binding — bound but never read (`cache_positions = 0`); a
  // fresh buffer reads as zero, no upload needed.
  const dummyCacheBuf = createStorageBuffer(8);
  // GQA's additive bias — always the all-zero "no mask" case here.
  const maskBuf = createStorageBuffer(maxSeqLen * 4);
  const probsBuf = createStorageBuffer(numHeads * maxSeqLen * 4);

  // ---- Uniforms; per-step fields (pos_offset / query_offset / s_eff) are rewritten in place. ----
  const hiddenNormUniform = uniformOf([["u32", 1], ["u32", hiddenSize], ["f32", rmsNormEps], ["u32", 1]]);
  // QK-norm: N = heads rows of D = headDim, one shared gamma (groups = 1).
  const qNormUniform = uniformOf([["u32", numHeads], ["u32", headDim], ["f32", rmsNormEps], ["u32", 1]]);
  const kNormUniform = uniformOf([["u32", numKvHeads], ["u32", headDim], ["f32", rmsNormEps], ["u32", 1]]);
  const qUniform = uniformOf([["u32", qDim], ["u32", hiddenSize]]);
  const kUniform = uniformOf([["u32", kvDim], ["u32", hiddenSize]]);
  const vUniform = uniformOf([["u32", kvDim], ["u32", hiddenSize]]);
  const oUniform = uniformOf([["u32", hiddenSize], ["u32", qDim]]);
  const gateUniform = uniformOf([["u32", ffnHidden], ["u32", hiddenSize]]);
  const upUniform = uniformOf([["u32", ffnHidden], ["u32", hiddenSize]]);
  const downUniform = uniformOf([["u32", hiddenSize], ["u32", ffnHidden]]);
  const siluUniform = uniformOf([["u32", ffnHidden], ["u32", ACTIVATION.silu], ["f32", 1]]);
  const addUniform = uniformOf([["u32", hiddenSize], ["u32", ELEMENTWISE.add]]);
  const mulUniform = uniformOf([["u32", ffnHidden], ["u32", ELEMENTWISE.multiply]]);
  const ropeQUniform = uniformOf([
    ["u32", 1], ["u32", numHeads], ["u32", headDim], ["u32", 0], ["u32", 0],
    ["f32", ropeTheta], ["f32", 1], ["f32", 0], ["f32", 1], ["f32", 1],
    ["u32", 0], ["u32", numHeads],
  ]);
  const ropeKUniform = uniformOf([
    ["u32", 1], ["u32", numKvHeads], ["u32", headDim], ["u32", 0], ["u32", 0],
    ["f32", ropeTheta], ["f32", 1], ["f32", 0], ["f32", 1], ["f32", 1],
    ["u32", 0], ["u32", numKvHeads],
  ]);
  // S = maxSeqLen always (the cache's stride); query_offset and s_eff are
  // rewritten to `at` / `at + 1` every step so the scan stays tight.
  const scoresUniform = uniformOf([
    ["u32", numHeads], ["u32", numKvHeads], ["u32", 1], ["u32", maxSeqLen], ["u32", headDim],
    ["f32", 1 / Math.sqrt(headDim)], ["u32", 1], ["i32", 0], ["u32", 1], ["u32", 1], ["u32", 1], ["u32", maxSeqLen],
  ]);
  const contextUniform = uniformOf([
    ["u32", numHeads], ["u32", numKvHeads], ["u32", 1], ["u32", maxSeqLen], ["u32", headDim], ["u32", maxSeqLen],
  ]);

  // ---- Shared bind groups (layer-independent). ----
  const ropeQGroup = await bindGroup(ropePipeline, [qNormedBuf, dummyCacheBuf, qRopedBuf, ropeQUniform]);
  const ropeKGroup = await bindGroup(ropePipeline, [kNormedBuf, dummyCacheBuf, kRopedBuf, ropeKUniform]);
  const add1Group = await bindGroup(elementwisePipeline, [hiddenA, projOutBuf, hiddenB, addUniform]);
  const siluGroup = await bindGroup(activationPipeline, [gateOutBuf, gateActBuf, siluUniform]);
  const mulGroup = await bindGroup(elementwisePipeline, [gateActBuf, upOutBuf, gatedBuf, mulUniform]);
  const add2Group = await bindGroup(elementwisePipeline, [hiddenB, downOutBuf, hiddenA, addUniform]);

  // ---- Per-layer weights, bind groups and KV cache. ----
  const trackedBuffers: GPUBuffer[] = [];
  const track = (buf: GPUBuffer): GPUBuffer => {
    trackedBuffers.push(buf);
    return buf;
  };

  async function projectionGroup(w: PackedQ8, uniform: GPUBuffer, vectorBuf: GPUBuffer, outBuf: GPUBuffer): Promise<GPUBindGroup> {
    const { weightBuf, scaleBuf } = uploadPacked(w);
    track(weightBuf);
    track(scaleBuf);
    return bindGroup(matvecPipeline, [weightBuf, scaleBuf, vectorBuf, outBuf, uniform]);
  }

  const layers: LayerResident[] = [];
  for (const lw of weights.perLayer) {
    const attnNormBuf = track(createStorageBuffer(lw.attnNorm.byteLength));
    upload(attnNormBuf, 0, lw.attnNorm);
    const ffnNormBuf = track(createStorageBuffer(lw.ffnNorm.byteLength));
    upload(ffnNormBuf, 0, lw.ffnNorm);
    // Permuted gammas (ops/rope channel order), per weights-q8.ts's contract.
    const qGammaBuf = track(createStorageBuffer(lw.qNorm.byteLength));
    upload(qGammaBuf, 0, lw.qNorm);
    const kGammaBuf = track(createStorageBuffer(lw.kNorm.byteLength));
    upload(kGammaBuf, 0, lw.kNorm);

    const kCacheBuf = track(createStorageBuffer(numKvHeads * maxSeqLen * headDim * 4));
    const vCacheBuf = track(createStorageBuffer(numKvHeads * maxSeqLen * headDim * 4));

    layers.push({
      kCacheBuf,
      vCacheBuf,
      attnNormGroup: await bindGroup(rmsnormPipeline, [hiddenA, attnNormBuf, normedBuf, hiddenNormUniform]),
      ffnNormGroup: await bindGroup(rmsnormPipeline, [hiddenB, ffnNormBuf, normed2Buf, hiddenNormUniform]),
      wqGroup: await projectionGroup(lw.wq, qUniform, normedBuf, qProjBuf),
      wkGroup: await projectionGroup(lw.wk, kUniform, normedBuf, kProjBuf),
      wvGroup: await projectionGroup(lw.wv, vUniform, normedBuf, vProjBuf),
      qNormGroup: await bindGroup(rmsnormPipeline, [qProjBuf, qGammaBuf, qNormedBuf, qNormUniform]),
      kNormGroup: await bindGroup(rmsnormPipeline, [kProjBuf, kGammaBuf, kNormedBuf, kNormUniform]),
      scoresGroup: await bindGroup(scoresPipeline, [qRopedBuf, kCacheBuf, maskBuf, probsBuf, scoresUniform]),
      contextGroup: await bindGroup(contextPipeline, [probsBuf, vCacheBuf, attnOutBuf, contextUniform]),
      woGroup: await projectionGroup(lw.wo, oUniform, attnOutBuf, projOutBuf),
      gateGroup: await projectionGroup(lw.wGate, gateUniform, normed2Buf, gateOutBuf),
      upGroup: await projectionGroup(lw.wUp, upUniform, normed2Buf, upOutBuf),
      downGroup: await projectionGroup(lw.wDown, downUniform, gatedBuf, downOutBuf),
    });
  }

  const finalNormBuf = track(createStorageBuffer(weights.finalNorm.byteLength));
  upload(finalNormBuf, 0, weights.finalNorm);
  const finalNormGroup = await bindGroup(rmsnormPipeline, [hiddenA, finalNormBuf, finalNormedBuf, hiddenNormUniform]);

  // ---- Tied lm_head: embedTokens as three row-chunks (dispatchRowsChunked's
  // arithmetic, resident form — separate buffers, not offsets). ----
  const wordsPerRow = Math.ceil(hiddenSize / 4);
  const lmHeadChunks: LmHeadChunk[] = [];
  for (let rowStart = 0; rowStart < vocabSize; rowStart += MAX_WORKGROUPS_PER_DISPATCH) {
    const rowCount = Math.min(MAX_WORKGROUPS_PER_DISPATCH, vocabSize - rowStart);
    const chunkUniform = uniformOf([["u32", rowCount], ["u32", hiddenSize]]);
    const outBuf = track(createStorageBuffer(rowCount * 4));
    const staging = track(createStorageBuffer(rowCount * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ));
    const group = await projectionGroup(
      {
        packed: weights.embedTokens.packed.subarray(rowStart * wordsPerRow, (rowStart + rowCount) * wordsPerRow),
        scale: weights.embedTokens.scale.subarray(rowStart, rowStart + rowCount),
        rows: rowCount,
        cols: hiddenSize,
      },
      chunkUniform,
      finalNormedBuf,
      outBuf,
    );
    lmHeadChunks.push({ rowCount, group, outBuf, staging });
  }

  const wg256 = (n: number) => Math.ceil(n / 256);
  let position = 0;
  let firstStepCounted = false;

  async function decodeStep(tokenId: number): Promise<Float32Array> {
    const at = position;
    if (at + 1 > maxSeqLen) {
      // Without this, the KV copy below lands in the next head's region — a
      // validation error outside any error scope, which invalidates the whole
      // command buffer and resolves with stale logits (upstream PR #116, item 1).
      throw new Error(`decodeStep: position ${at + 1} exceeds maxSeqLen=${maxSeqLen}`);
    }

    // The only CPU work per step: one embedding row and three 4-byte counters.
    const embedRow = gatherDequantRow(weights.embedTokens, tokenId);
    upload(hiddenA, 0, embedRow);
    upload(ropeQUniform, ROPE_POS_OFFSET_BYTE, new Uint32Array([at]));
    upload(ropeKUniform, ROPE_POS_OFFSET_BYTE, new Uint32Array([at]));
    upload(scoresUniform, GQA_QUERY_OFFSET_BYTE, new Int32Array([at]));
    const sEff = new Uint32Array([at + 1]);
    upload(scoresUniform, GQA_SCORES_S_EFF_BYTE, sEff);
    upload(contextUniform, GQA_CONTEXT_S_EFF_BYTE, sEff);

    const ops: Op[] = [];
    const dispatch = (pipeline: GPUComputePipeline, group: GPUBindGroup, workgroups: Workgroups) =>
      ops.push({ kind: "dispatch", pipeline, bindGroup: group, workgroups });
    const copy = (src: GPUBuffer, srcOffset: number, dst: GPUBuffer, dstOffset: number, size: number) =>
      ops.push({ kind: "copy", src, srcOffset, dst, dstOffset, size });

    for (const layer of layers) {
      dispatch(rmsnormPipeline, layer.attnNormGroup, [1]);
      dispatch(matvecPipeline, layer.wqGroup, [qDim]);
      dispatch(matvecPipeline, layer.wkGroup, [kvDim]);
      dispatch(matvecPipeline, layer.wvGroup, [kvDim]);
      // Qwen3 QK-norm, then RoPE — order matters and matches model.ts' oracle.
      dispatch(rmsnormPipeline, layer.qNormGroup, [numHeads]);
      dispatch(rmsnormPipeline, layer.kNormGroup, [numKvHeads]);
      dispatch(ropePipeline, ropeQGroup, [wg256(qDim / 2)]);
      dispatch(ropePipeline, ropeKGroup, [wg256(kvDim / 2)]);
      // KV append: GPU-to-GPU copies in the same encoder — the new K/V were
      // written by the dispatches just above. One copy per head (heads are
      // not contiguous in the [kvHeads, maxSeqLen, headDim] cache layout).
      for (let h = 0; h < numKvHeads; h += 1) {
        copy(kRopedBuf, h * headDim * 4, layer.kCacheBuf, (h * maxSeqLen + at) * headDim * 4, headDim * 4);
        copy(vProjBuf, h * headDim * 4, layer.vCacheBuf, (h * maxSeqLen + at) * headDim * 4, headDim * 4);
      }
      dispatch(scoresPipeline, layer.scoresGroup, [1, numHeads, 1]);
      dispatch(contextPipeline, layer.contextGroup, [1, numHeads, 1]);
      dispatch(matvecPipeline, layer.woGroup, [hiddenSize]);
      dispatch(elementwisePipeline, add1Group, [wg256(hiddenSize)]);
      dispatch(rmsnormPipeline, layer.ffnNormGroup, [1]);
      dispatch(matvecPipeline, layer.gateGroup, [ffnHidden]);
      dispatch(matvecPipeline, layer.upGroup, [ffnHidden]);
      dispatch(activationPipeline, siluGroup, [wg256(ffnHidden)]);
      dispatch(elementwisePipeline, mulGroup, [wg256(ffnHidden)]);
      dispatch(matvecPipeline, layer.downGroup, [hiddenSize]);
      dispatch(elementwisePipeline, add2Group, [wg256(hiddenSize)]);
    }
    dispatch(rmsnormPipeline, finalNormGroup, [1]);
    for (const chunk of lmHeadChunks) dispatch(matvecPipeline, chunk.group, [chunk.rowCount]);

    if (!firstStepCounted) {
      firstStepCounted = true;
      stats.dispatchesPerStep = ops.filter((op) => op.kind === "dispatch").length;
      stats.copiesPerStep = ops.filter((op) => op.kind === "copy").length;
    }

    // One encoder, one submit, logits-only readback.
    const encoder = device.createCommandEncoder();
    let pass: GPUComputePassEncoder | null = null;
    const endPass = () => {
      if (pass) {
        pass.end();
        pass = null;
      }
    };
    for (const op of ops) {
      if (op.kind === "dispatch") {
        if (!pass) pass = encoder.beginComputePass();
        pass.setPipeline(op.pipeline);
        pass.setBindGroup(0, op.bindGroup);
        pass.dispatchWorkgroups(...(op.workgroups as [number, number?, number?]));
      } else {
        // copyBufferToBuffer cannot be recorded inside a compute pass.
        endPass();
        encoder.copyBufferToBuffer(op.src, op.srcOffset, op.dst, op.dstOffset, op.size);
      }
    }
    endPass();
    for (const chunk of lmHeadChunks) {
      encoder.copyBufferToBuffer(chunk.outBuf, 0, chunk.staging, 0, chunk.rowCount * 4);
    }
    device.queue.submit([encoder.finish()]);
    stats.submits += 1;

    const logits = new Float32Array(vocabSize);
    let offset = 0;
    for (const chunk of lmHeadChunks) {
      await chunk.staging.mapAsync(GPUMapMode.READ);
      logits.set(new Float32Array(chunk.staging.getMappedRange().slice(0)), offset);
      chunk.staging.unmap();
      offset += chunk.rowCount;
    }

    position += 1;
    return logits;
  }

  return {
    decodeStep,
    get position() {
      return position;
    },
    reset() {
      // The cache is overwritten from position 0 and every attention scan is
      // bounded by s_eff, so stale tail positions are never read.
      position = 0;
    },
    stats,
    destroy() {
      for (const buf of trackedBuffers) buf.destroy();
    },
  };
}
