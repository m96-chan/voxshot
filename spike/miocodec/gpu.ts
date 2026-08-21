import { CONV1D, ISTFT, MATMUL } from "./kernels.js";

/**
 * A WebGPU backend for the three ops that carry the arithmetic.
 *
 * `matmul`, `conv1d` and the inverse transform are where the work is: the rest
 * of the decoder is linear in its data and costs nothing beside them. So this
 * dispatches those three and leaves everything else on the reference
 * implementations — a hybrid, stated plainly, rather than a half-finished
 * rewrite pretending to be a GPU pipeline.
 *
 * **Weights stay resident.** Uploading a `[1536, 512]` matrix on every call
 * would move more bytes than the multiply reads, and the decoder asks for the
 * same two hundred matrices on every decode. Activations are uploaded per call
 * and read back per call, because the stages between these ops run on the CPU
 * and would need the data there anyway.
 */

const TILE = 16;
const WORKGROUP = 256;

export interface GpuLimits {
  maxStorageBufferBindingSize: number;
}

export class Gpu {
  private readonly pipelines = new Map<string, GPUComputePipeline>();
  /**
   * Keyed on the weight's own array, so a caller that hands over the same
   * tensor twice uploads once. `WeakMap`, so dropping the checkpoint drops the
   * buffers with it rather than pinning half a gigabyte of VRAM behind a cache
   * nobody can reach.
   */
  private readonly resident = new WeakMap<Float32Array, GPUBuffer>();

  private constructor(
    readonly device: GPUDevice,
    readonly adapterInfo: string,
  ) {}

  /**
   * A device, or null where WebGPU is absent or refuses.
   *
   * Null rather than a throw: the demo has a working CPU path and falling back
   * to it is a better answer than a broken page. What must not happen is
   * falling back **silently** — the caller reports which one ran.
   */
  /**
   * Wrap a device obtained some other way.
   *
   * Node has no `navigator.gpu`; the `webgpu` package hands back a `GPU` after
   * installing its globals, and web-xpu-ops' own harness uses it that way. The
   * tests need a real device rather than a mock — a kernel that compiles and
   * computes the wrong thing is exactly what a mock cannot catch.
   */
  static fromDevice(device: GPUDevice, info: string): Gpu {
    return new Gpu(device, info);
  }

  static async create(): Promise<Gpu | null> {
    const gpu = (globalThis.navigator as Navigator | undefined)?.gpu;
    if (!gpu) return null;
    const adapter = await gpu.requestAdapter();
    if (!adapter) return null;
    // Asked for explicitly. The default binding size is 128 MiB, and while
    // nothing here is near it, the limit a device *reports* is free and the
    // default is not — the same reasoning web-xpu-ops' own runner gives.
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    const info = adapter.info
      ? [adapter.info.vendor, adapter.info.architecture, adapter.info.description]
          .filter(Boolean)
          .join(" ") || "unknown adapter"
      : "unknown adapter";
    return new Gpu(device, info);
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
    this.device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
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
    const output = device.createBuffer({
      size: Math.max(4, outputLength * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const params = this.uniform(uniforms);

    const entries: GPUBindGroupEntry[] = [];
    inputs.forEach((buffer, index) => entries.push({ binding: index, resource: { buffer } }));
    entries.push({ binding: inputs.length, resource: { buffer: output } });
    entries.push({ binding: inputs.length + 1, resource: { buffer: params } });

    const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });

    const staging = device.createBuffer({
      size: Math.max(4, outputLength * 4),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups[0], workgroups[1] ?? 1, workgroups[2] ?? 1);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, staging, 0, Math.max(4, outputLength * 4));
    device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(staging.getMappedRange().slice(0, outputLength * 4));
    staging.unmap();

    staging.destroy();
    output.destroy();
    params.destroy();
    return result;
  }

  /**
   * `[M, K] x [K, N]`, with `b` kept on the device between calls.
   *
   * `b` is the transposed weight, which is the same array every time this layer
   * runs; `a` is the activation, which is not.
   */
  async matmul(a: Float32Array, b: Float32Array, M: number, N: number, K: number): Promise<Float32Array> {
    const bBuffer = this.residentBuffer(b);
    const aBuffer = this.upload(a);
    try {
      return await this.dispatch(MATMUL, [aBuffer, bBuffer], M * N, [M, N, K], [
        Math.ceil(N / TILE),
        Math.ceil(M / TILE),
      ]);
    } finally {
      aBuffer.destroy();
    }
  }

  /** `conv1d`, with the weight and bias resident. `N` is always 1 here. */
  async conv1d(
    input: Float32Array,
    weight: Float32Array,
    bias: Float32Array | null,
    Cin: number,
    Cout: number,
    L: number,
    K: number,
    padding: number,
  ): Promise<Float32Array> {
    const outLength = L + 2 * padding - (K - 1) - 1 + 1;
    const inputBuffer = this.upload(input);
    const weightBuffer = this.residentBuffer(weight);
    // The kernel binds a bias buffer whether or not there is one, and since #46
    // an unreferenced binding throws rather than reading zeros — so an absent
    // bias is an explicit array of zeros, not a missing binding.
    const biasBuffer = this.residentBuffer(bias ?? zeros(Cout));
    try {
      const dispatchLength = Math.ceil(outLength / WORKGROUP) * WORKGROUP;
      return await this.dispatch(
        CONV1D,
        [inputBuffer, weightBuffer, biasBuffer],
        Cout * outLength,
        [Cin, Cout, L, K, outLength, 1, padding, 1, Cin, Cout, 0, 0],
        [dispatchLength / WORKGROUP, Cout, 1],
      );
    } finally {
      inputBuffer.destroy();
    }
  }

  /** The inverse transform. `pad` carries the padding convention, resolved by the caller. */
  async istft(
    real: Float32Array,
    imag: Float32Array,
    window: Float32Array,
    frames: number,
    nFft: number,
    hop: number,
    pad: number,
    outLength: number,
  ): Promise<Float32Array> {
    const bins = Math.floor(nFft / 2) + 1;
    const realBuffer = this.upload(real);
    const imagBuffer = this.upload(imag);
    const windowBuffer = this.residentBuffer(window);
    try {
      return await this.dispatch(
        ISTFT,
        [realBuffer, imagBuffer, windowBuffer],
        outLength,
        [nFft, hop, bins, frames, outLength, pad],
        [Math.ceil(outLength / WORKGROUP)],
      );
    } finally {
      realBuffer.destroy();
      imagBuffer.destroy();
    }
  }

  destroy(): void {
    this.device.destroy();
  }
}

const ZEROS = new Map<number, Float32Array>();

/** One shared zero bias per width, so the resident cache does not fill with copies. */
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
export function gpuBackend(gpu: Gpu): import("./decoder.js").Backend {
  return {
    name: `WebGPU (${gpu.adapterInfo})`,
    matmul: (a, b, M, N, K) => gpu.matmul(a, b, M, N, K),
    conv1d: (input, weight, bias, Cin, Cout, L, K, padding) =>
      gpu.conv1d(input, weight, bias, Cin, Cout, L, K, padding),
    istft: (real, imag, window, frames, nFft, hop) =>
      // `"same"` resolved here, because the kernel takes a number and has no
      // convention of its own: crop `(nFft - hop) / 2` from each end, which
      // leaves `hop * frames` samples. The reference does the same arithmetic
      // behind the mode name.
      gpu.istft(real, imag, window, frames, nFft, hop, (nFft - hop) / 2, hop * frames),
  };
}
