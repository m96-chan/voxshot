/**
 * A `GPUDevice` that records instead of computing.
 *
 * The engine's load-bearing claims are about *shape of work*, not arithmetic:
 * every buffer is allocated once at construction, one decode step is one
 * submit, and the only thing read back per step is the logits. Those are
 * properties of the wiring — the part voxshot owns — and they are exactly what
 * a recording device can check, on a machine with no GPU at all.
 *
 * What it deliberately does not do is compute. Buffers read back as zeros, so
 * nothing here says anything about whether the kernels are right; that is
 * web-xpu-ops' question upstream and `npm run test:models`' question here.
 *
 * WebGPU's enum globals do not exist in Node, so {@link installGpuGlobals} puts
 * the two the engine reads in place. They are plain bit constants from the
 * spec, not behaviour.
 */

export interface FakeBuffer {
  readonly id: number;
  readonly size: number;
  readonly usage: number;
  destroyed: boolean;
  destroy(): void;
  mapAsync(mode: number): Promise<void>;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
}

export interface RecordedCopy {
  readonly src: FakeBuffer;
  readonly dst: FakeBuffer;
  readonly size: number;
}

export interface FakeGpuLog {
  /** Every buffer ever created, in order. */
  buffers: FakeBuffer[];
  /** `queue.submit` calls. */
  submits: number;
  /** Compute passes begun. */
  passes: number;
  /** `dispatchWorkgroups` calls, across every encoder. */
  dispatches: number;
  /** `copyBufferToBuffer` calls, across every encoder. */
  copies: RecordedCopy[];
  /** Bytes handed to `queue.writeBuffer`. */
  bytesWritten: number;
  /** Bytes actually mapped for reading — the readback. */
  bytesRead: number;
  /** Shader modules created; one per distinct WGSL source the engine compiles. */
  shaderModules: number;
  pipelines: number;
  bindGroups: number;
}

export interface FakeGpu {
  device: GPUDevice;
  log: FakeGpuLog;
  /** Zero the per-step counters so one decode step can be measured on its own. */
  resetCounters(): void;
}

/** The `GPUBufferUsage` / `GPUMapMode` bits the engine reads, from the spec. */
export function installGpuGlobals(): void {
  const globals = globalThis as Record<string, unknown>;
  globals.GPUBufferUsage ??= {
    MAP_READ: 0x0001,
    MAP_WRITE: 0x0002,
    COPY_SRC: 0x0004,
    COPY_DST: 0x0008,
    INDEX: 0x0010,
    VERTEX: 0x0020,
    UNIFORM: 0x0040,
    STORAGE: 0x0080,
    INDIRECT: 0x0100,
    QUERY_RESOLVE: 0x0200,
  };
  globals.GPUMapMode ??= { READ: 0x0001, WRITE: 0x0002 };
}

export function createFakeGpu(): FakeGpu {
  installGpuGlobals();

  const log: FakeGpuLog = {
    buffers: [],
    submits: 0,
    passes: 0,
    dispatches: 0,
    copies: [],
    bytesWritten: 0,
    bytesRead: 0,
    shaderModules: 0,
    pipelines: 0,
    bindGroups: 0,
  };

  let nextId = 0;

  const createBuffer = ({ size, usage }: { size: number; usage: number }): FakeBuffer => {
    // Zeros: this device records, it does not compute. A test that cared what
    // came back would be asserting arithmetic a fake cannot provide.
    const backing = new ArrayBuffer(size);
    const buffer: FakeBuffer = {
      id: (nextId += 1),
      size,
      usage,
      destroyed: false,
      destroy() {
        buffer.destroyed = true;
      },
      async mapAsync() {
        log.bytesRead += size;
      },
      getMappedRange: () => backing,
      unmap: () => {},
    };
    log.buffers.push(buffer);
    return buffer;
  };

  const pass = {
    setPipeline: () => {},
    setBindGroup: () => {},
    dispatchWorkgroups: () => {
      log.dispatches += 1;
    },
    end: () => {},
  };

  const createCommandEncoder = () => ({
    beginComputePass: () => {
      log.passes += 1;
      return pass;
    },
    copyBufferToBuffer: (
      src: FakeBuffer,
      _srcOffset: number,
      dst: FakeBuffer,
      _dstOffset: number,
      size: number,
    ) => {
      log.copies.push({ src, dst, size });
    },
    finish: () => ({}),
  });

  const device = {
    createBuffer,
    createShaderModule: () => {
      log.shaderModules += 1;
      // The engine reads compilation diagnostics so a WGSL error is reported
      // against the kernel that has it rather than as a pipeline failure.
      // Nothing here is compiled, so the report is always empty.
      return { getCompilationInfo: async () => ({ messages: [] }) };
    },
    createComputePipeline: () => {
      log.pipelines += 1;
      return { getBindGroupLayout: () => ({}) };
    },
    createBindGroup: () => {
      log.bindGroups += 1;
      return {};
    },
    createCommandEncoder,
    // The engine wraps pipeline and bind-group creation in a validation scope
    // so a WGSL or binding mistake is reported where it was made. Nothing here
    // is invalid, so the scope always pops clean.
    pushErrorScope: () => {},
    popErrorScope: async () => null,
    queue: {
      writeBuffer: (
        _buffer: FakeBuffer,
        _offset: number,
        _data: ArrayBuffer | ArrayBufferView,
        _dataOffset?: number,
        size?: number,
      ) => {
        log.bytesWritten += size ?? 0;
      },
      submit: () => {
        log.submits += 1;
      },
    },
  };

  return {
    device: device as unknown as GPUDevice,
    log,
    resetCounters() {
      log.submits = 0;
      log.passes = 0;
      log.dispatches = 0;
      log.copies = [];
      log.bytesRead = 0;
      log.bytesWritten = 0;
    },
  };
}
