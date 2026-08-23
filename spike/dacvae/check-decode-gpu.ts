import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);

import { decodeGpu } from "./decode-gpu.js";
import { decoderConfig } from "./decoder.js";
import { ResidentGpu } from "./gpu-resident.js";

/**
 * The device-resident decode against the same golden `check.ts` uses.
 *
 *     cd spike/dacvae && npm run check:decode-gpu
 *
 * Only the waveform is compared, not each stage: this path keeps its
 * intermediates on the device by design, and reading them back would be
 * checking a path it does not take. Every operation is between the latent and
 * that waveform, so a fault in any of them lands here.
 */

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN = join(HERE, "golden");
const TOLERANCE = 1e-4;

function golden(name: string): Float32Array {
  const bytes = readFileSync(join(GOLDEN, `${name}.f32`));
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return new Float32Array(copy.buffer);
}

async function main(): Promise<void> {
  const instance = create([]);
  const adapter = await instance.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  const info = adapter.info?.description ?? "unknown adapter";
  const device = await ResidentGpu.requestDevice(adapter);
  const gpu = new ResidentGpu(device, info, [instance, adapter]);
  gpu.begin();

  const config = decoderConfig();
  const latent = golden("latent_mean");
  const frames = latent.length / config.latentDim;
  console.log(`decoding ${frames} frames on ${info} (max workgroups ${gpu.maxWorkgroups})`);

  // Warm the pipelines, then time the path a request would take.
  const uploaded = gpu.upload(latent);
  await gpu.read(decodeGpu(gpu, uploaded, frames).tensor);
  await gpu.check("warm-up");

  const started = Date.now();
  const mine = await gpu.read(decodeGpu(gpu, uploaded, frames).tensor);
  await gpu.check("decoding");
  const ms = Date.now() - started;

  const theirs = golden("tail_2_tanh");
  let worst = 0;
  let peak = 0;
  for (let i = 0; i < theirs.length; i += 1) {
    worst = Math.max(worst, Math.abs(mine[i]! - theirs[i]!));
    peak = Math.max(peak, Math.abs(theirs[i]!));
  }
  const relative = worst / peak;
  const ok = mine.length === theirs.length && relative <= TOLERANCE;
  console.log(
    `${ok ? "ok  " : "FAIL"} ${mine.length} samples in ${ms} ms  ` +
      `max |diff| ${worst.toExponential(2)} of peak ${peak.toFixed(3)} (${relative.toExponential(2)} relative)`,
  );
  console.log(`     ${gpu.stats.dispatches} dispatches, ${gpu.stats.submits} submits`);
  if (!ok) process.exitCode = 1;
  gpu.destroy();
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
