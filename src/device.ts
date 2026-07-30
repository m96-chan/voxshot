import { DeviceUnavailableError, InvalidInputError } from "./errors.js";
import type { GpuProbe } from "./platform.js";

/** What the caller asks for. */
export type DevicePreference = "auto" | "webgpu" | "wasm";

/** What VoxShot actually runs on after probing the environment. */
export type ResolvedDevice = "webgpu" | "wasm";

/**
 * Decide which backend to run inference on.
 *
 * - `auto` (default): WebGPU when present, WASM otherwise.
 * - `webgpu`: WebGPU or a {@link DeviceUnavailableError}; never silently slow.
 * - `wasm`: WASM without probing the GPU at all.
 */
export async function resolveDevice(
  preference: DevicePreference | undefined,
  gpu: GpuProbe,
): Promise<ResolvedDevice> {
  const requested = preference ?? "auto";

  switch (requested) {
    case "wasm":
      return "wasm";
    case "webgpu":
      if (await gpu.isAvailable()) {
        return "webgpu";
      }
      throw new DeviceUnavailableError("webgpu");
    case "auto":
      return (await gpu.isAvailable()) ? "webgpu" : "wasm";
    default:
      throw new InvalidInputError(
        `Unknown device preference "${String(requested)}". Expected "auto", "webgpu" or "wasm".`,
      );
  }
}
