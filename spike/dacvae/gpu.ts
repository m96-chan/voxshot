import type { Backend } from "./backend.js";
import { CONV1D, CONV_TRANSPOSE1D, SNAKE } from "./kernels.js";

/**
 * The decode path on WebGPU.
 *
 * Structure follows `spike/miocodec/gpu.ts` — read, not re-derived — with one
 * difference that matters: MioCodec dispatched three ops out of a dozen and
 * left the rest on the CPU, because the rest was cheap. Here the three ops ARE
 * the graph, so nothing is left behind and the activations never need to be on
 * the host between stages.
 *
 * **Weights stay resident.** The decoder asks for the same hundred filters on
 * every call, and the largest is 1536x768x24 — 113 MB as f32. Uploading that
 * per call would move more bytes than the multiply reads. Activations are a
 * different matter and are uploaded and read back per call for now; keeping
 * them on the device between stages is the obvious next tuning step and is
 * deliberately not done yet, so the first number measured is the honest
 * unoptimised one.
 *
 * ## Why the sizes here are worth watching
 *
 * DACVAE decodes to 48 kHz and carries 96 channels at full rate right to the
 * end: `decoder_model_4` and the snake after it are 4.6 M floats per second of
 * audio each, and the whole graph moves about 61 MB per second of audio. That
 * is several times what MioCodec's 24 kHz decoder ever held, and it is a
 * bandwidth problem rather than an arithmetic one — which is exactly the sort
 * of thing a CPU timing cannot predict.
 */

const WORKGROUP = 256;

/**
 * `maxComputeWorkgroupsPerDimension`, the spec's guaranteed minimum and what
 * most implementations actually report — including this one.
 *
 * It binds on `snake` and nothing else. The convolutions dispatch
 * `[ceil(Lout/256), Cout]`, and both stay small. `snake` flattens `C * L` into
 * one dimension, so at 48 kHz it crosses 65,535 after about 3.5 seconds of
 * audio: 192 channels x 96,000 samples / 256 = 72,000 workgroups. Measured, as
 * a validation failure rather than as wrong numbers.
 */
const MAX_WORKGROUPS_PER_DIMENSION = 65535;

export class Gpu {
  private readonly pipelines = new Map<string, GPUComputePipeline>();
  /**
   * Keyed on the weight's own array, so a caller that hands over the same
   * tensor twice uploads once. `WeakMap`, so dropping the checkpoint drops the
   * buffers with it rather than pinning a quarter of a gigabyte of VRAM behind
   * a cache nobody can reach.
   */
  private readonly resident = new WeakMap<Float32Array, GPUBuffer>();

  private constructor(
    readonly device: GPUDevice,
    readonly adapterInfo: string,
    /**
     * Whatever produced the device, held so it cannot be collected.
     *
     * **Not defensive.** Dawn's Node binding does not keep its `GPU` instance
     * alive from the `GPUDevice`, so once the instance becomes unreachable the
     * collector takes it and later dispatches crash — as a futex abort, a
     * segfault, or a hang, whichever the race lands on. Measured: 200
     * dispatches with the instance dropped fail 3 times out of 3, and the
     * identical code holding a reference passes 3 out of 3.
     *
     * This is very likely what web-xpu-ops #107 / #49 / #68 are: "a test that
     * takes more than a few milliseconds before its first dispatch" and "a file
     * that holds too many dispatches" are both descriptions of *more time for
     * the collector to run*, which is why no vitest pool configuration helped.
     */
    private readonly retain: unknown,
  ) {}

  static fromDevice(device: GPUDevice, info: string, retain?: unknown): Gpu {
    return new Gpu(device, info, retain);
  }

  private pipeline(code: string): GPUComputePipeline {
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

  private upload(data: Float32Array): GPUBuffer {
    const buffer = this.device.createBuffer({
      size: Math.max(4, data.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    // Written through a plain `ArrayBuffer` view: `writeBuffer` refuses a
    // `SharedArrayBuffer`-backed one, and TypeScript cannot tell which kind an
    // op's `Float32Array` result is backed by.
    this.device.queue.writeBuffer(
      buffer,
      0,
      data.buffer as ArrayBuffer,
      data.byteOffset,
      data.byteLength,
    );
    return buffer;
  }

  /** Upload once and keep, for anything that does not change between calls. */
  private residentBuffer(data: Float32Array): GPUBuffer {
    let buffer = this.resident.get(data);
    if (!buffer) {
      buffer = this.upload(data);
      this.resident.set(data, buffer);
    }
    return buffer;
  }

  private uniform(values: number[]): GPUBuffer {
    // A uniform block rounds up to 16 bytes; the kernels that need padding
    // declare it themselves, so the caller passes every word including those.
    const words = new Uint32Array(Math.max(4, Math.ceil(values.length / 4) * 4));
    words.set(values);
    const buffer = this.device.createBuffer({
      size: words.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buffer, 0, words);
    return buffer;
  }

  private async dispatch(
    code: string,
    inputs: GPUBuffer[],
    outputLength: number,
    uniforms: number[],
    workgroups: [number, number?, number?],
  ): Promise<Float32Array> {
    const device = this.device;
    const pipeline = this.pipeline(code);
    const bytes = Math.max(4, outputLength * 4);
    const output = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const params = this.uniform(uniforms);

    const entries: GPUBindGroupEntry[] = inputs.map((buffer, index) => ({
      binding: index,
      resource: { buffer },
    }));
    entries.push({ binding: inputs.length, resource: { buffer: output } });
    entries.push({ binding: inputs.length + 1, resource: { buffer: params } });
    const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });

    const staging = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // A validation scope around the whole recording. Without it the first
    // failure surfaces as "[Invalid CommandBuffer] is invalid due to a previous
    // error" at submit time, which names the symptom and not the cause.
    device.pushErrorScope("validation");
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups[0], workgroups[1] ?? 1, workgroups[2] ?? 1);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, staging, 0, bytes);
    device.queue.submit([encoder.finish()]);
    const invalid = await device.popErrorScope();
    if (invalid) {
      throw new Error(
        `dispatch failed: ${invalid.message}\n` +
          `  output ${outputLength} elements, workgroups ${workgroups.join(" x ")}, ` +
          `uniforms [${uniforms.join(", ")}]`,
      );
    }

    await staging.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(staging.getMappedRange().slice(0, outputLength * 4));
    staging.unmap();

    staging.destroy();
    output.destroy();
    params.destroy();
    return result;
  }

  /** `conv1d` at stride 1 — the only stride the decode path uses. */
  async conv1d(
    input: Float32Array,
    weight: Float32Array,
    bias: Float32Array | undefined,
    Cin: number,
    Cout: number,
    L: number,
    K: number,
    padding: number,
    dilation: number,
  ): Promise<Float32Array> {
    const outLength = L + 2 * padding - dilation * (K - 1);
    const inputBuffer = this.upload(input);
    try {
      return await this.dispatch(
        CONV1D,
        [inputBuffer, this.residentBuffer(weight), this.residentBuffer(bias ?? zeros(Cout))],
        Cout * outLength,
        [Cin, Cout, L, K, outLength, 1, padding, dilation, Cin, Cout, 0, 0],
        [Math.ceil(outLength / WORKGROUP), Cout, 1],
      );
    } finally {
      inputBuffer.destroy();
    }
  }

  /**
   * `convTranspose1d` at dilation 1.
   *
   * `output_padding` is not a uniform field — see kernels.ts — so it enters
   * only through the output length, and the kernel writes zeros wherever no
   * input contributes.
   */
  async convTranspose1d(
    input: Float32Array,
    weight: Float32Array,
    bias: Float32Array | undefined,
    Cin: number,
    Cout: number,
    L: number,
    K: number,
    stride: number,
    padding: number,
    outputPadding: number,
  ): Promise<Float32Array> {
    const outLength = (L - 1) * stride - 2 * padding + K + outputPadding;
    const inputBuffer = this.upload(input);
    try {
      return await this.dispatch(
        CONV_TRANSPOSE1D,
        [inputBuffer, this.residentBuffer(weight), this.residentBuffer(bias ?? zeros(Cout))],
        Cout * outLength,
        [Cin, Cout, L, K, outLength, stride, padding, 1, Cin, Cout, 0, 0],
        [Math.ceil(outLength / WORKGROUP), Cout, 1],
      );
    } finally {
      inputBuffer.destroy();
    }
  }

  /**
   * The learned per-channel periodic activation.
   *
   * Chunked by channel when the flat dispatch would exceed
   * {@link MAX_WORKGROUPS_PER_DIMENSION}. A run of whole channels is itself a
   * valid `[C', L]` snake with a sliced alpha, so this stays inside the op's
   * own contract rather than reaching for a second dispatch dimension the
   * kernel does not read.
   */
  async snake(input: Float32Array, alpha: Float32Array, C: number, L: number): Promise<Float32Array> {
    const perChannel = Math.ceil(L / WORKGROUP);
    const channelsPerChunk = Math.max(1, Math.floor(MAX_WORKGROUPS_PER_DIMENSION / perChannel));
    if (channelsPerChunk >= C) {
      const inputBuffer = this.upload(input);
      try {
        return await this.dispatch(
          SNAKE,
          [inputBuffer, this.residentBuffer(alpha)],
          C * L,
          [1, C, L],
          [Math.ceil((C * L) / WORKGROUP)],
        );
      } finally {
        inputBuffer.destroy();
      }
    }

    const out = new Float32Array(C * L);
    for (let first = 0; first < C; first += channelsPerChunk) {
      const count = Math.min(channelsPerChunk, C - first);
      const slice = input.subarray(first * L, (first + count) * L);
      // `subarray` on alpha would keep the whole array alive as a resident
      // buffer key; a copy is 4 bytes per channel and keeps the cache honest.
      const alphaSlice = alpha.slice(first, first + count);
      const inputBuffer = this.upload(slice);
      try {
        out.set(
          await this.dispatch(
            SNAKE,
            [inputBuffer, this.residentBuffer(alphaSlice)],
            count * L,
            [1, count, L],
            [Math.ceil((count * L) / WORKGROUP)],
          ),
          first * L,
        );
      } finally {
        inputBuffer.destroy();
      }
    }
    return out;
  }

  destroy(): void {
    this.device.destroy();
  }
}

/** One shared zero bias per width, so the resident cache does not fill with copies. */
const ZEROS = new Map<number, Float32Array>();
function zeros(length: number): Float32Array {
  let array = ZEROS.get(length);
  if (!array) {
    array = new Float32Array(length);
    ZEROS.set(length, array);
  }
  return array;
}

/**
 * The {@link Backend} face of {@link Gpu}.
 *
 * Separate from the class so `decoder.ts` never imports WebGPU types, and so
 * the CPU and GPU paths are the same graph with one object swapped.
 */
export function gpuBackend(gpu: Gpu): Backend {
  return {
    name: `WebGPU (${gpu.adapterInfo})`,
    conv1d: ({ input, weight, bias, Cin, Cout, L, K, padding, dilation }) =>
      gpu.conv1d(input, weight, bias, Cin, Cout, L, K, padding, dilation),
    convTranspose1d: ({ input, weight, bias, Cin, Cout, L, K, stride, padding, outputPadding }) =>
      gpu.convTranspose1d(input, weight, bias, Cin, Cout, L, K, stride, padding, outputPadding),
    snake: ({ input, alpha, C, L }) => gpu.snake(input, alpha, C, L),
  };
}
