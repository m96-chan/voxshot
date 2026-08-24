import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);

import { loadBertWeights } from "./bert-weights.js";
import { Gpu } from "./gpu.js";
import { prepareBert, runBertGpu } from "./modernbert-gpu.js";
import { realLength } from "./modernbert.js";

/**
 * ModernBERT-ja on the device, against the goldens `check-bert.ts` uses.
 *
 *     cd spike/irodori && npm run check:bert-gpu
 *
 * Only `final_norm` is compared, not every layer: the device version keeps its
 * intermediates on the device by design, and reading them back would be
 * checking a path the port does not take. The 25 layers are between the input
 * and that number, so a fault in any of them lands here.
 *
 * The bound is looser than the CPU path's 5e-6 for the same reason
 * `check-dit-gpu.ts`'s is — the device tiles its matmuls and reduces in a tree
 * where the reference walks K in one loop — and is set from what this achieves.
 */

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN = join(HERE, "golden", "bert");
const TOLERANCE = 5e-5;

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
  const gpu = Gpu.fromDevice(await Gpu.requestDevice(adapter), info, [instance, adapter]);
  gpu.begin();

  const { weights } = loadBertWeights(GOLDEN);
  const index = JSON.parse(readFileSync(join(GOLDEN, "index.json"), "utf8")) as {
    cases: Record<string, { normalized: string; real_positions: number }>;
  };
  console.log(`ModernBERT-ja on ${info}: ${weights.config.numLayers} layers\n`);

  let failures = 0;
  for (const [name, entry] of Object.entries(index.cases)) {
    const paddedIds = Int32Array.from(golden(`${name}/input_ids`), (value) => value | 0);
    const keep = Array.from(golden(`${name}/mask`), (value) => value !== 0);
    const real = realLength(keep);

    const started = Date.now();
    const bert = prepareBert(gpu, weights, paddedIds.subarray(0, real), keep.slice(0, real));
    const mine = await gpu.read(runBertGpu(bert));
    await gpu.check(`${name}: running the encoder`);
    const ms = Date.now() - started;

    const theirs = golden(`${name}/final_norm`);
    let worst = 0;
    let peak = 0;
    for (let i = 0; i < mine.length; i += 1) {
      worst = Math.max(worst, Math.abs(mine[i]! - theirs[i]!));
      peak = Math.max(peak, Math.abs(theirs[i]!));
    }
    const relative = worst / peak;
    const ok = relative <= TOLERANCE;
    if (!ok) failures += 1;
    const preview = entry.normalized.length > 22 ? `${entry.normalized.slice(0, 22)}…` : entry.normalized;
    console.log(
      `${ok ? "ok  " : "FAIL"} ${name.padEnd(6)} ${real.toString().padStart(3)} positions  ` +
        `max |diff| ${worst.toExponential(2)} of peak ${peak.toFixed(2)} ` +
        `(${relative.toExponential(2)} relative, ${ms} ms)  ${JSON.stringify(preview)}`,
    );
  }

  console.log();
  console.log(
    failures > 0
      ? `${failures} cases disagree`
      : `both cases agree within ${TOLERANCE.toExponential(0)} of peak, on the device`,
  );
  if (failures > 0) process.exitCode = 1;
  gpu.destroy();
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
