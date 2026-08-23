import {
  ACTIVATION,
  ATTENTION_CONTEXT,
  ATTENTION_SCORES,
  ELEMENTWISE,
  GATHER,
  LAYERNORM,
  MATMUL,
  PERMUTE,
  RMSNORM,
  ROPE,
} from "./kernels.js";

/**
 * A device-resident tensor engine for the Irodori half.
 *
 * ## Why the tensors stay on the device
 *
 * `spike/dacvae`'s `Gpu` reads every result back to a `Float32Array`, which is
 * right there: its decode path is a dozen dispatches over very large buffers,
 * so the copies are lost in the arithmetic. This graph is the opposite shape —
 * **twelve blocks, thirty-two steps, roughly thirty dispatches each** over
 * tensors of a few megabytes. Reading each one back would spend the whole run
 * waiting on `mapAsync`, and the CPU reference is only slow because it *is* the
 * arithmetic.
 *
 * So a {@link Tensor} here is a `GPUBuffer` and a length. Nothing crosses back
 * to JavaScript until {@link Gpu.read}, which the sampler calls once per step
 * and the synthesizer once at the end.
 *
 * ## Weights are uploaded once
 *
 * Keyed on the `Float32Array` itself, in a `WeakMap`: handing the same weight
 * over twice uploads once, and dropping the checkpoint drops the buffers with
 * it rather than pinning a gigabyte behind a cache nobody can reach.
 *
 * ## One submit per step, not one per dispatch
 *
 * Dispatches are recorded into a single `GPUCommandEncoder` and flushed when a
 * readback needs them. A submit per dispatch is the other thing that turns a
 * compute-bound graph into a latency-bound one.
 *
 * ## Scratch is pooled and bind groups are cached
 *
 * One utterance records about fifteen thousand dispatches. Allocating a fresh
 * output buffer, a fresh uniform and a fresh bind group for each would make the
 * host the bottleneck — the GPU would sit idle behind `createBindGroup`.
 *
 * Every block and every step runs the *same shapes*, so {@link Gpu.scratch}
 * hands back the same buffer for the same `(slot, length)` and bind groups are
 * cached on the identities that go into them. After the first block, recording
 * is a `setBindGroup` and a `dispatchWorkgroups`.
 *
 * The cost of pooling is that two live tensors must never share a slot. Slots
 * are named by what they hold, so aliasing is a naming mistake rather than an
 * arithmetic one — and `check-dit.ts` compares against the reference, which is
 * what would catch it.
 */

/** A buffer on the device, and how many f32 it holds. */
export interface Tensor {
  readonly buffer: GPUBuffer;
  readonly length: number;
  /** Identity for the bind-group cache; buffers have none of their own. */
  readonly uid: number;
}

const WORKGROUP = 256;
/** `const TILE: u32 = 16u` in the matmul kernel. */
const MATMUL_TILE = 16;

export class Gpu {
  private readonly pipelines = new Map<string, GPUComputePipeline>();
  private readonly resident = new WeakMap<Float32Array, Tensor>();
  private readonly owned: GPUBuffer[] = [];
  private readonly pool = new Map<string, Tensor>();
  private readonly groups = new Map<string, GPUBindGroup>();
  private readonly uniforms = new Map<string, GPUBuffer>();
  private nextUid = 1;
  private encoder: GPUCommandEncoder | null = null;
  private pass: GPUComputePassEncoder | null = null;
  private recorded = 0;

  /** Dispatch and submit counts, so "it is slow" can be attributed. */
  readonly stats = { dispatches: 0, submits: 0, readbacks: 0, uploaded: 0 };

  private constructor(
    readonly device: GPUDevice,
    readonly info: string,
    /**
     * Held, not used. Dawn's Node binding does not keep the `GPU` alive from
     * the `GPUDevice`, and a collected instance takes the device down mid-run —
     * the cause of four upstream issues before it was found.
     */
    private readonly retain: unknown,
  ) {}

  static fromDevice(device: GPUDevice, info: string, retain?: unknown): Gpu {
    return new Gpu(device, info, retain);
  }

  /** A device with the adapter's limits, not the spec minimums. */
  static async requestDevice(adapter: GPUAdapter): Promise<GPUDevice> {
    const wanted = [
      "maxStorageBufferBindingSize",
      "maxBufferSize",
      "maxComputeWorkgroupsPerDimension",
    ] as const;
    const requiredLimits: Record<string, number> = {};
    for (const name of wanted) {
      const supported = adapter.limits[name];
      if (typeof supported === "number") requiredLimits[name] = supported;
    }
    return adapter.requestDevice({ requiredLimits });
  }

  // ---- buffers ----------------------------------------------------------

  /** An uninitialised tensor of `length` f32, freshly allocated. */
  alloc(length: number): Tensor {
    const buffer = this.device.createBuffer({
      size: Math.max(4, length * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.owned.push(buffer);
    this.nextUid += 1;
    return { buffer, length, uid: this.nextUid };
  }

  /**
   * A reusable buffer, named by what it holds.
   *
   * The same `(slot, length)` returns the same tensor every time, which is what
   * makes the bind-group cache hit. Two tensors that are live at the same
   * moment must not share a slot — the names below are chosen so that is
   * visible at the call site.
   */
  scratch(slot: string, length: number): Tensor {
    const key = `${slot}:${length}`;
    let found = this.pool.get(key);
    if (!found) {
      found = this.alloc(length);
      this.pool.set(key, found);
    }
    return found;
  }

  /** A tensor holding a copy of `data`, uploaded now. */
  upload(data: Float32Array): Tensor {
    const tensor = this.alloc(data.length);
    this.write(tensor, data);
    this.stats.uploaded += data.byteLength;
    return tensor;
  }

  /**
   * A weight, uploaded at most once.
   *
   * Weights never change, so this is the difference between uploading 1.8 GB
   * once and uploading it every step.
   */
  weight(data: Float32Array): Tensor {
    let tensor = this.resident.get(data);
    if (!tensor) {
      tensor = this.upload(data);
      this.resident.set(data, tensor);
    }
    return tensor;
  }

  private write(tensor: Tensor, data: Float32Array | Int32Array): void {
    // Through a plain `ArrayBuffer` view: `writeBuffer` refuses a
    // `SharedArrayBuffer`-backed one and TypeScript cannot tell them apart.
    this.device.queue.writeBuffer(
      tensor.buffer,
      0,
      data.buffer as ArrayBuffer,
      data.byteOffset,
      data.byteLength,
    );
  }

  /** Integer indices, for `gather`. */
  uploadInts(data: Int32Array): Tensor {
    const tensor = this.alloc(data.length);
    this.write(tensor, data);
    return tensor;
  }

  // ---- dispatch ---------------------------------------------------------

  private pipelineFor(code: string): GPUComputePipeline {
    let pipeline = this.pipelines.get(code);
    if (!pipeline) {
      pipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: { module: this.device.createShaderModule({ code }), entryPoint: "main" },
      });
      this.pipelines.set(code, pipeline);
    }
    return pipeline;
  }

  private uniform(words: number[], floats: Set<number>): GPUBuffer {
    // Cached on the values: every block runs the same shapes, so this is a few
    // dozen buffers for a whole utterance rather than fifteen thousand.
    const key = `${words.join(",")}|${[...floats].join(",")}`;
    const cached = this.uniforms.get(key);
    if (cached) return cached;
    // A uniform block rounds up to 16 bytes. Fields are packed in declaration
    // order; `floats` names the ones the kernel reads as f32, because a `u32`
    // and an `f32` are the same four bytes and only the writer knows which.
    const size = Math.max(4, Math.ceil(words.length / 4) * 4);
    const view = new ArrayBuffer(size * 4);
    const u32 = new Uint32Array(view);
    const f32 = new Float32Array(view);
    words.forEach((value, index) => {
      if (floats.has(index)) f32[index] = value;
      else u32[index] = value >>> 0;
    });
    const buffer = this.device.createBuffer({
      size: view.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.owned.push(buffer);
    this.device.queue.writeBuffer(buffer, 0, view);
    this.uniforms.set(key, buffer);
    return buffer;
  }

  private record(
    code: string,
    tensors: Tensor[],
    words: number[],
    floats: Set<number>,
    workgroups: [number, number?, number?],
  ): void {
    const pipeline = this.pipelineFor(code);
    const key = `${code.length}:${tensors.map((t) => t.uid).join(",")}|${words.join(",")}`;
    let bindGroup = this.groups.get(key);
    if (!bindGroup) {
      const entries: GPUBindGroupEntry[] = tensors.map((tensor, binding) => ({
        binding,
        resource: { buffer: tensor.buffer },
      }));
      entries.push({ binding: tensors.length, resource: { buffer: this.uniform(words, floats) } });
      bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries,
      });
      this.groups.set(key, bindGroup);
    }

    if (!this.pass) {
      // Only make an encoder when there is none. `copy` ends the pass and
      // keeps its encoder, so creating one here would orphan every command
      // recorded before that copy — which is how this port first produced a
      // tensor of zeros that took no time at all to compute.
      if (!this.encoder) this.encoder = this.device.createCommandEncoder();
      this.pass = this.encoder.beginComputePass();
    }
    this.pass.setPipeline(pipeline);
    this.pass.setBindGroup(0, bindGroup);
    this.pass.dispatchWorkgroups(workgroups[0], workgroups[1] ?? 1, workgroups[2] ?? 1);
    this.stats.dispatches += 1;
    this.recorded += 1;
  }

  /**
   * Copy bytes between two tensors.
   *
   * A buffer-to-buffer copy is not a compute pass, so the open one has to end
   * first. That is why concatenation is done once per run rather than per step:
   * every copy here breaks the batch of dispatches in two.
   */
  copy(src: Tensor, srcOffset: number, dst: Tensor, dstOffset: number, bytes: number): void {
    if (this.pass && this.encoder) {
      this.pass.end();
      this.pass = null;
    }
    if (!this.encoder) this.encoder = this.device.createCommandEncoder();
    this.encoder.copyBufferToBuffer(src.buffer, srcOffset, dst.buffer, dstOffset, bytes);
  }

  /** Close the pass and submit whatever is recorded. */
  flush(): void {
    if (!this.encoder) return;
    if (this.pass) {
      this.pass.end();
      this.pass = null;
    }
    this.device.queue.submit([this.encoder.finish()]);
    this.stats.submits += 1;
    this.pass = null;
    this.encoder = null;
    this.recorded = 0;
  }

  /** Read a tensor back, submitting anything still recorded. */
  async read(tensor: Tensor): Promise<Float32Array> {
    const bytes = Math.max(4, tensor.length * 4);
    const staging = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    if (this.encoder) {
      if (this.pass) {
        this.pass.end();
        this.pass = null;
      }
      this.encoder.copyBufferToBuffer(tensor.buffer, 0, staging, 0, bytes);
      this.device.queue.submit([this.encoder.finish()]);
      this.encoder = null;
      this.recorded = 0;
    } else {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(tensor.buffer, 0, staging, 0, bytes);
      this.device.queue.submit([encoder.finish()]);
    }
    this.stats.submits += 1;
    this.stats.readbacks += 1;

    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0, tensor.length * 4));
    staging.unmap();
    staging.destroy();
    return out;
  }

  /** Surface any validation error recorded so far, with a useful message. */
  async check(what: string): Promise<void> {
    this.flush();
    const invalid = await this.device.popErrorScope();
    if (invalid) throw new Error(`${what}: ${invalid.message}`);
    this.device.pushErrorScope("validation");
  }

  begin(): void {
    this.device.pushErrorScope("validation");
  }

  destroy(): void {
    this.flush();
    for (const buffer of this.owned) buffer.destroy();
    this.owned.length = 0;
    void this.retain;
  }

  // ---- ops --------------------------------------------------------------

  /** Elementwise `a + b` or `a * b`; both sides the same length. */
  elementwise(a: Tensor, b: Tensor, kind: 0 | 1, slot: string): Tensor {
    if (a.length !== b.length) throw new Error(`elementwise: ${a.length} vs ${b.length}`);
    const out = this.scratch(slot, a.length);
    this.record(ELEMENTWISE, [a, b, out], [a.length, kind], new Set(), [
      Math.ceil(a.length / WORKGROUP),
    ]);
    return out;
  }

  /** `[dim0, dim1, D]` to `[dim1, dim0, D]`. */
  permute(input: Tensor, dim0: number, dim1: number, D: number, slot: string): Tensor {
    const out = this.scratch(slot, input.length);
    this.record(PERMUTE, [input, out], [dim0, dim1, D], new Set(), [
      Math.ceil((dim0 * dim1 * D) / WORKGROUP),
    ]);
    return out;
  }

  /** `[M, K] @ [K, N]`. */
  matmul(a: Tensor, b: Tensor, M: number, N: number, K: number, slot: string): Tensor {
    const out = this.scratch(slot, M * N);
    this.record(
      MATMUL,
      [a, b, out],
      [M, N, K],
      new Set(),
      [Math.ceil(N / MATMUL_TILE), Math.ceil(M / MATMUL_TILE), 1],
    );
    return out;
  }

  /** `x * rsqrt(mean(x^2) + eps) * weight[row % groups]`. */
  rmsnorm(
    input: Tensor,
    weight: Tensor,
    N: number,
    D: number,
    eps: number,
    slot: string,
    groups = 1,
  ): Tensor {
    const out = this.scratch(slot, N * D);
    this.record(
      RMSNORM,
      [input, weight, out],
      [N, D, eps, groups],
      new Set([2]),
      [N],
    );
    return out;
  }

  layernorm(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    N: number,
    D: number,
    eps: number,
    slot: string,
  ): Tensor {
    const out = this.scratch(slot, N * D);
    this.record(
      LAYERNORM,
      [input, weight, bias, out],
      [N, D, eps],
      new Set([2]),
      [N],
    );
    return out;
  }

  /**
   * RoPE over `[N, heads, headDim]`, rotating only `[headOffset, +headCount)`.
   *
   * The `cache` binding is required by the kernel and unused here: this port
   * has no KV cache, so `cache_positions` is 0 and a one-element buffer stands
   * in. Leaving the binding unfilled is a validation error, not a shortcut.
   */
  rope(
    input: Tensor,
    N: number,
    heads: number,
    headDim: number,
    thetaBase: number,
    slot: string,
    headOffset = 0,
    headCount = heads,
  ): Tensor {
    const out = this.scratch(slot, input.length);
    this.record(
      ROPE,
      [input, this.noCache(), out],
      // The five float fields are `ropeFrequencyParams` with no scaling:
      // effective_base = thetaBase, and the other four are the identities that
      // make the kernel's one expression reduce to plain RoPE bit for bit.
      [N, heads, headDim, 0, 0, thetaBase, 1, 0, 1, 1, headOffset, headCount],
      new Set([5, 6, 7, 8, 9]),
      [Math.ceil((N * heads * headDim) / 2 / WORKGROUP)],
    );
    return out;
  }

  private cacheStub: Tensor | null = null;
  private noCache(): Tensor {
    if (!this.cacheStub) this.cacheStub = this.alloc(1);
    return this.cacheStub;
  }

  activation(input: Tensor, kind: number, slot: string): Tensor {
    const out = this.scratch(slot, input.length);
    this.record(
      ACTIVATION,
      [input, out],
      [input.length, kind, 1],
      new Set([2]),
      [Math.ceil(input.length / WORKGROUP)],
    );
    return out;
  }

  gather(table: Tensor, indices: Tensor, N: number, D: number, rows: number, slot: string): Tensor {
    const out = this.scratch(slot, N * D);
    this.record(
      GATHER,
      [table, indices, out],
      [N, D, rows],
      new Set(),
      [Math.ceil((N * D) / WORKGROUP)],
    );
    return out;
  }

  /**
   * Attention in two dispatches: scores (softmax included) then context.
   *
   * `mask` is an additive bias of `[maskBatch, maskHeads, maskRows] x S`, each
   * of the first three either 1 or its full extent — the same broadcast the
   * reference's `resolveMask` implements.
   */
  attention(args: {
    q: Tensor;
    k: Tensor;
    v: Tensor;
    mask: Tensor;
    maskShape: [number, number, number];
    B: number;
    H: number;
    L: number;
    S: number;
    D: number;
    scale: number;
    slot: string;
  }): Tensor {
    const { q, k, v, mask, maskShape, B, H, L, S, D, scale, slot } = args;
    const probs = this.scratch(`${slot}.probs`, B * H * L * S);
    this.record(
      ATTENTION_SCORES,
      [q, k, mask, probs],
      [H, L, S, D, scale, 0, 0, maskShape[0], maskShape[1], maskShape[2]],
      new Set([4]),
      [L, H, B],
    );
    const out = this.scratch(slot, B * H * L * D);
    this.record(
      ATTENTION_CONTEXT,
      [probs, v, out],
      [H, L, S, D],
      new Set(),
      [L, H, B],
    );
    return out;
  }
}
