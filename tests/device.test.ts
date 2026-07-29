import { describe, expect, it } from "vitest";

import { resolveDevice } from "../src/device.js";
import { DeviceUnavailableError, InvalidInputError } from "../src/errors.js";
import type { GpuProbe } from "../src/platform.js";

const gpu = (available: boolean): GpuProbe => ({ isAvailable: async () => available });

describe("resolveDevice", () => {
  it("prefers webgpu in auto mode when it is available", async () => {
    await expect(resolveDevice("auto", gpu(true))).resolves.toBe("webgpu");
  });

  it("falls back to wasm in auto mode when webgpu is unavailable", async () => {
    await expect(resolveDevice("auto", gpu(false))).resolves.toBe("wasm");
  });

  it("defaults to auto when no preference is given", async () => {
    await expect(resolveDevice(undefined, gpu(true))).resolves.toBe("webgpu");
  });

  it("honours an explicit webgpu request", async () => {
    await expect(resolveDevice("webgpu", gpu(true))).resolves.toBe("webgpu");
  });

  it("fails loudly when webgpu is requested but unavailable", async () => {
    await expect(resolveDevice("webgpu", gpu(false))).rejects.toBeInstanceOf(
      DeviceUnavailableError,
    );
  });

  it("never probes the gpu when wasm is requested", async () => {
    let probed = false;
    const probe: GpuProbe = {
      isAvailable: async () => {
        probed = true;
        return true;
      },
    };

    await expect(resolveDevice("wasm", probe)).resolves.toBe("wasm");
    expect(probed).toBe(false);
  });

  it("rejects an unknown preference", async () => {
    await expect(
      resolveDevice("cuda" as unknown as "auto", gpu(true)),
    ).rejects.toBeInstanceOf(InvalidInputError);
  });
});
