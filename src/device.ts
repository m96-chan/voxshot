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
 * - `webgpu`: WebGPU or a {@link DeviceUnavailableError}.
 * - `wasm`: WASM without probing the GPU at all.
 *
 * This picks what to *try*, and nothing more. An engine may still degrade
 * internally — `ChatterboxEngine`'s fallback chain ends on WASM so a GPU that
 * cannot run the model still produces audio — so `webgpu` here does not
 * promise you will end up there.
 *
 * Two things follow. `VoxShot.device` reports where the engine actually
 * landed, so a degrade is observable rather than silent. And an engine that
 * should refuse rather than degrade says so itself, with `requiresGpu`:
 * whether CPU is usable depends on the model, not on the caller's preference.
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
