import { ACTIVATION, CONV1D, CONV_TRANSPOSE1D, ELEMENTWISE, SNAKE } from "./kernels.js";

/**
 * The decode path with its tensors kept on the device.
 *
 * `gpu.ts` reads every result back to a `Float32Array`, which was the right
 * shape for what it was written against — the `check.ts` harness, where a stage
 * has to be compared against a golden anyway. It is the wrong shape for
 * **running** the decoder, and by how much is measurable:
 *
 * ```
 *   decoder_model_3   2030 ms   127.5 MB
 *   decoder_model_4   1859 ms   127.5 MB
 * ```
 *
 * Those two blocks are nine operations each over a 127.5 MB tensor, and each
 * operation copied it down and back up. Four seconds of a five-second request,
 * spent moving numbers rather than computing them.
 *
 * Nothing here crosses back until {@link ResidentGpu.read}.
 *
 * ## The same shape as `spike/irodori`'s engine
 *
 * Scratch pooled by name, bind groups cached on the identities that go into
 * them, dispatches batched into one submit. Two spikes now want this and it is
 * written twice, which is the point at which it should move somewhere shared —
 * noted here rather than done, because moving it is a change to two working
 * things at once.
 */

export interface Tensor {
  readonly buffer: GPUBuffer;
  readonly length: number;
  readonly uid: number;
  readonly offset?: number;
}

const WORKGROUP = 256;

export class ResidentGpu {
  private readonly pipelines = new Map<string, GPUComputePipeline>();
  private readonly weights = new WeakMap<Float32Array, Tensor>();
  /** slot -> the buffer backing it, and how many f32 it can hold. */
  private readonly pool = new Map<string, { buffer: GPUBuffer; capacity: number; uid: number }>();
  private readonly groups = new Map<string, GPUBindGroup>();
  private readonly uniforms = new Map<string, GPUBuffer>();
  private readonly owned: GPUBuffer[] = [];
  private encoder: GPUCommandEncoder | null = null;
  private pass: GPUComputePassEncoder | null = null;
  private nextUid = 1;

  readonly stats = { dispatches: 0, submits: 0, buffers: 0, bytes: 0 };

  constructor(
    readonly device: GPUDevice,
    readonly info: string,
    /** Held, not used: Dawn's Node binding does not keep the `GPU` alive. */
    private readonly retain?: unknown,
  ) {}

  /** How many workgroups one dimension may carry — 65535 is the spec floor. */
  get maxWorkgroups(): number {
    return this.device.limits.maxComputeWorkgroupsPerDimension;
  }

  /** A device with the adapter's limits rather than the spec minimums. */
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

  alloc(length: number): Tensor {
    const buffer = this.device.createBuffer({
      size: Math.max(4, length * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.owned.push(buffer);
    this.stats.buffers += 1;
    this.stats.bytes += Math.max(4, length * 4);
    this.nextUid += 1;
    return { buffer, length, uid: this.nextUid };
  }

  scratch(slot: string, length: number): Tensor {
    let held = this.pool.get(slot);
    if (!held || held.capacity < length) {
      // Grown, not added to. Pooling on `(slot, length)` instead meant every
      // new utterance length allocated a fresh set and nothing was ever freed —
      // the decoder's last block is 127.5 MB a slot, so a handful of different
      // sentence lengths reached 25.7 GB of VRAM and took the machine's other
      // work down with it.
      if (held) {
        // Anything already recorded names the buffer that is about to go, and
        // a submit after `destroy()` is an error rather than a wrong answer —
        // "used in submit while destroyed". Send it first.
        this.flush();
        held.buffer.destroy();
        const at = this.owned.indexOf(held.buffer);
        if (at >= 0) this.owned.splice(at, 1);
        // The cached bind groups point at the buffer that just went away.
        for (const key of [...this.groups.keys()]) {
          if (key.includes(`#${held.uid}`)) this.groups.delete(key);
        }
      }
      const grown = this.alloc(length);
      held = { buffer: grown.buffer, capacity: length, uid: grown.uid };
      this.pool.set(slot, held);
    }
    // `length` is what gets bound and dispatched; the buffer may be larger.
    return { buffer: held.buffer, length, uid: held.uid };
  }


  /** Write into a named slot, reusing its buffer — for values that change. */
  writeInto(slot: string, data: Float32Array): Tensor {
    const tensor = this.scratch(slot, data.length);
    this.device.queue.writeBuffer(
      tensor.buffer,
      0,
      data.buffer as ArrayBuffer,
      data.byteOffset,
      data.byteLength,
    );
    return tensor;
  }

  upload(data: Float32Array): Tensor {
    const tensor = this.alloc(data.length);
    this.device.queue.writeBuffer(
      tensor.buffer,
      0,
      data.buffer as ArrayBuffer,
      data.byteOffset,
      data.byteLength,
    );
    return tensor;
  }

  /** A weight, uploaded at most once, keyed on the array itself. */
  weight(data: Float32Array): Tensor {
    let tensor = this.weights.get(data);
    if (!tensor) {
      tensor = this.upload(data);
      this.weights.set(data, tensor);
    }
    return tensor;
  }

  /**
   * A window onto part of a tensor.
   *
   * The offset has to be a multiple of `minStorageBufferOffsetAlignment`, 256
   * bytes; {@link snake}'s channel chunking is the only caller and falls back
   * to a device copy where a chunk does not land on one.
   */
  view(tensor: Tensor, offset: number, length: number): Tensor | null {
    const bytes = ((tensor.offset ?? 0) + offset) * 4;
    if (bytes % 256 !== 0) return null;
    return {
      buffer: tensor.buffer,
      length,
      uid: tensor.uid * 1_000_003 + offset,
      offset: (tensor.offset ?? 0) + offset,
    };
  }

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

  private uniform(words: number[]): GPUBuffer {
    const key = words.join(",");
    const cached = this.uniforms.get(key);
    if (cached) return cached;
    const size = Math.max(4, Math.ceil(words.length / 4) * 4);
    const data = new Uint32Array(size);
    data.set(words.map((value) => value >>> 0));
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.owned.push(buffer);
    this.device.queue.writeBuffer(buffer, 0, data);
    this.uniforms.set(key, buffer);
    return buffer;
  }

  private record(
    code: string,
    tensors: Tensor[],
    words: number[],
    workgroups: [number, number?, number?],
  ): void {
    const pipeline = this.pipelineFor(code);
    const key = `${code.length}:${tensors.map((t) => `#${t.uid}@${t.offset ?? 0}+${t.length}`).join(",")}|${words.join(",")}`;
    let group = this.groups.get(key);
    if (!group) {
      const entries: GPUBindGroupEntry[] = tensors.map((tensor, binding) => ({
        binding,
        resource: tensor.offset
          ? { buffer: tensor.buffer, offset: tensor.offset * 4, size: tensor.length * 4 }
          : { buffer: tensor.buffer, size: tensor.length * 4 },
      }));
      entries.push({ binding: tensors.length, resource: { buffer: this.uniform(words) } });
      group = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
      this.groups.set(key, group);
    }
    if (!this.pass) {
      // Only make an encoder when there is none: `copy` ends the pass and keeps
      // its encoder, and making a second would orphan what it recorded.
      if (!this.encoder) this.encoder = this.device.createCommandEncoder();
      this.pass = this.encoder.beginComputePass();
    }
    this.pass.setPipeline(pipeline);
    this.pass.setBindGroup(0, group);
    this.pass.dispatchWorkgroups(workgroups[0], workgroups[1] ?? 1, workgroups[2] ?? 1);
    this.stats.dispatches += 1;
  }

  copy(src: Tensor, srcOffset: number, dst: Tensor, dstOffset: number, bytes: number): void {
    if (this.pass) {
      this.pass.end();
      this.pass = null;
    }
    if (!this.encoder) this.encoder = this.device.createCommandEncoder();
    this.encoder.copyBufferToBuffer(
      src.buffer,
      (src.offset ?? 0) * 4 + srcOffset,
      dst.buffer,
      (dst.offset ?? 0) * 4 + dstOffset,
      bytes,
    );
  }

  flush(): void {
    if (!this.encoder) return;
    if (this.pass) {
      this.pass.end();
      this.pass = null;
    }
    this.device.queue.submit([this.encoder.finish()]);
    this.stats.submits += 1;
    this.encoder = null;
  }

  async read(tensor: Tensor): Promise<Float32Array> {
    const bytes = Math.max(4, tensor.length * 4);
    const staging = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    if (this.pass) {
      this.pass.end();
      this.pass = null;
    }
    if (!this.encoder) this.encoder = this.device.createCommandEncoder();
    this.encoder.copyBufferToBuffer(tensor.buffer, (tensor.offset ?? 0) * 4, staging, 0, bytes);
    this.device.queue.submit([this.encoder.finish()]);
    this.encoder = null;
    this.stats.submits += 1;
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0, bytes));
    staging.unmap();
    staging.destroy();
    return out;
  }

  async check(what: string): Promise<void> {
    this.flush();
    const invalid = await this.device.popErrorScope();
    if (invalid) throw new Error(`${what}: ${invalid.message}`);
    this.device.pushErrorScope("validation");
  }

  begin(): void {
    this.device.pushErrorScope("validation");
  }

  /**
   * Where the memory went, grouped by the first part of each slot name.
   *
   * Weights and pooled scratch are different problems — one is the model and
   * the other is the shapes it has been asked for — so a total is not an
   * answer. This separates them and names the biggest pools.
   */
  breakdown(): { weights: number; scratch: number; groups: [string, number, number][] } {
    let scratch = 0;
    const groups = new Map<string, [number, number]>();
    for (const [slot, held] of this.pool) {
      const bytes = held.capacity * 4;
      scratch += bytes;
      const key = slot.split(".").slice(0, 2).join(".");
      const entry = groups.get(key) ?? [0, 0];
      groups.set(key, [entry[0] + 1, entry[1] + bytes]);
    }
    const pooled = new Set([...this.pool.values()].map((h) => h.buffer));
    let weights = 0;
    for (const buffer of this.owned) if (!pooled.has(buffer)) weights += buffer.size;
    return {
      weights,
      scratch,
      groups: [...groups.entries()]
        .map(([name, [count, bytes]]) => [name, count, bytes] as [string, number, number])
        .sort((a, b) => b[2] - a[2]),
    };
  }

  /**
   * Drop every pooled buffer, keeping the uploaded weights.
   *
   * The codec's encoder holds the largest scratch in the process — a
   * twenty-second clip needs 243 MB per activation and a residual unit has four
   * — and it runs only when a voice arrives. Holding that between voices costs
   * gigabytes for nothing.
   */
  releaseScratch(prefix?: string): number {
    let freed = 0;
    for (const [slot, held] of [...this.pool]) {
      if (prefix !== undefined && !slot.startsWith(prefix)) continue;
      freed += held.capacity * 4;
      held.buffer.destroy();
      const at = this.owned.indexOf(held.buffer);
      if (at >= 0) this.owned.splice(at, 1);
      this.pool.delete(slot);
    }
    // Cached bind groups name buffers that have gone. Dropping all of them
    // costs a rebuild rather than a wrong answer.
    this.groups.clear();
    return freed;
  }

  destroy(): void {
    this.flush();
    for (const buffer of this.owned) buffer.destroy();
    this.owned.length = 0;
    void this.retain;
  }

  // ---- the four the decode path is made of ------------------------------

  conv1d(args: {
    input: Tensor;
    weight: Tensor;
    bias: Tensor;
    Cin: number;
    Cout: number;
    L: number;
    K: number;
    padding: number;
    dilation: number;
    stride?: number;
    slot: string;
  }): Tensor {
    const { input, weight, bias, Cin, Cout, L, K, padding, dilation, slot } = args;
    const stride = args.stride ?? 1;
    const outLength = Math.floor((L + 2 * padding - dilation * (K - 1) - 1) / stride) + 1;
    const out = this.scratch(slot, Cout * outLength);
    this.record(
      CONV1D,
      [input, weight, bias, out],
      [Cin, Cout, L, K, outLength, stride, padding, dilation, Cin, Cout, 0, 0],
      [Math.ceil(outLength / WORKGROUP), Cout, 1],
    );
    return out;
  }

  convTranspose1d(args: {
    input: Tensor;
    weight: Tensor;
    bias: Tensor;
    Cin: number;
    Cout: number;
    L: number;
    K: number;
    stride: number;
    padding: number;
    outputPadding: number;
    slot: string;
  }): Tensor {
    const { input, weight, bias, Cin, Cout, L, K, stride, padding, outputPadding, slot } = args;
    const outLength = (L - 1) * stride - 2 * padding + K + outputPadding;
    const out = this.scratch(slot, Cout * outLength);
    this.record(
      CONV_TRANSPOSE1D,
      [input, weight, bias, out],
      [Cin, Cout, L, K, outLength, stride, padding, 1, Cin, Cout, 0, 0],
      [Math.ceil(outLength / WORKGROUP), Cout, 1],
    );
    return out;
  }

  /**
   * Snake, chunked by channel where one dispatch would exceed the limit.
   *
   * `[C', L]` with a sliced alpha is a valid snake, so the chunking stays
   * inside the op's contract rather than reaching for a dispatch dimension the
   * kernel does not read. Chunks are bound as windows when they land on a
   * 256-byte boundary and copied device-side when they do not — either way the
   * host never sees the data.
   */
  snake(input: Tensor, alpha: Float32Array, C: number, L: number, slot: string): Tensor {
    const out = this.scratch(slot, C * L);
    const perChannel = Math.ceil(L / WORKGROUP);
    const perChunk = Math.max(1, Math.floor(this.maxWorkgroups / perChannel));
    if (perChunk >= C) {
      this.record(SNAKE, [input, this.weight(alpha), out], [1, C, L], [
        Math.ceil((C * L) / WORKGROUP),
      ]);
      return out;
    }
    for (let first = 0; first < C; first += perChunk) {
      const count = Math.min(perChunk, C - first);
      // A copy of alpha, not a subarray: a subarray would keep the whole array
      // alive as a resident-buffer key.
      const alphaChunk = this.weight(alpha.slice(first, first + count));
      const inView = this.view(input, first * L, count * L);
      const outView = this.view(out, first * L, count * L);
      const words: [number, number, number] = [1, count, L];
      const groups: [number] = [Math.ceil((count * L) / WORKGROUP)];
      if (inView && outView) {
        this.record(SNAKE, [inView, alphaChunk, outView], words, groups);
        continue;
      }
      const staging = this.scratch(`${slot}.chunk`, count * L);
      this.copy(input, first * L * 4, staging, 0, count * L * 4);
      const result = this.scratch(`${slot}.chunk.out`, count * L);
      this.record(SNAKE, [staging, alphaChunk, result], words, groups);
      this.copy(result, 0, out, first * L * 4, count * L * 4);
    }
    return out;
  }

  /**
   * Split a flat op into pieces no dispatch exceeds the limit on.
   *
   * At 48 kHz the tail carries 96 channels of 332160 samples — 31.9 M
   * elements, 124560 workgroups against a limit of 65535. `snake` already
   * chunked for this reason; `add` and `tanh` did not, because the shapes that
   * reach them were smaller in every test until a real utterance arrived.
   *
   * Pieces are a multiple of 64 elements so every offset lands on the 256-byte
   * boundary a bound window needs.
   */
  private chunks(length: number): { offset: number; count: number }[] {
    const most = Math.floor((this.maxWorkgroups * WORKGROUP) / 64) * 64;
    if (length <= most) return [{ offset: 0, count: length }];
    const out: { offset: number; count: number }[] = [];
    for (let offset = 0; offset < length; offset += most) {
      out.push({ offset, count: Math.min(most, length - offset) });
    }
    return out;
  }

  /** The output tail's `tanh`; kind 3 in the activation kernel. */
  tanh(input: Tensor, slot: string): Tensor {
    const out = this.scratch(slot, input.length);
    for (const { offset, count } of this.chunks(input.length)) {
      const a = offset === 0 ? input : this.view(input, offset, count);
      const o = offset === 0 ? out : this.view(out, offset, count);
      if (!a || !o) throw new Error(`tanh: chunk at ${offset} is not bindable`);
      this.record(ACTIVATION, [a, o], [count, 3, 0], [Math.ceil(count / WORKGROUP)]);
    }
    return out;
  }

  /** Elementwise add, for the residual units. */
  add(a: Tensor, b: Tensor, slot: string): Tensor {
    if (a.length !== b.length) throw new Error(`add: ${a.length} vs ${b.length}`);
    const out = this.scratch(slot, a.length);
    for (const { offset, count } of this.chunks(a.length)) {
      const x = offset === 0 ? a : this.view(a, offset, count);
      const y = offset === 0 ? b : this.view(b, offset, count);
      const o = offset === 0 ? out : this.view(out, offset, count);
      if (!x || !y || !o) throw new Error(`add: chunk at ${offset} is not bindable`);
      this.record(ELEMENTWISE, [x, y, o], [count, 0], [Math.ceil(count / WORKGROUP)]);
    }
    return out;
  }
}
