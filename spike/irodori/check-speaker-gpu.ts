import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);

import { Gpu } from "./gpu.js";
import { loadModelWeights } from "./model-weights.js";
import { prepareSpeaker, runSpeakerGpu } from "./speaker-encoder-gpu.js";

/**
 * The speaker encoder on the device, against the golden `check-speaker.ts` uses.
 *
 *     cd spike/irodori && npm run check:speaker-gpu
 *
 * `speaker_norm` is the comparison point rather than the encoder's raw output:
 * it is what `encode_conditions` hands on, the norm is one more chance to be
 * wrong, and the golden for it exists.
 */

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN = join(HERE, "golden");
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

  const weights = loadModelWeights(join(GOLDEN, "model"));
  const latent = golden("speaker_encoder.in.latent");
  const keep = Array.from(golden("speaker_encoder.in.mask"), (value) => value !== 0);
  const patchedDim = latent.length / keep.length;
  console.log(
    `speaker encoder on ${info}: ${weights.speaker.layers} blocks, ` +
      `${keep.length} frames of ${patchedDim}\n`,
  );

  const uploaded = gpu.upload(latent);
  const speaker = prepareSpeaker(gpu, weights, keep);
  await gpu.read(runSpeakerGpu(speaker, uploaded, patchedDim)); // warm the pipelines
  await gpu.check("warm-up");

  const started = Date.now();
  const mine = await gpu.read(runSpeakerGpu(speaker, uploaded, patchedDim));
  await gpu.check("running the speaker encoder");
  const ms = Date.now() - started;

  const theirs = golden("speaker_norm");
  let worst = 0;
  let peak = 0;
  for (let i = 0; i < theirs.length; i += 1) {
    worst = Math.max(worst, Math.abs(mine[i]! - theirs[i]!));
    peak = Math.max(peak, Math.abs(theirs[i]!));
  }
  const relative = worst / peak;
  const ok = relative <= TOLERANCE;
  console.log(
    `${ok ? "ok  " : "FAIL"} speaker_norm  max |diff| ${worst.toExponential(2)} ` +
      `of peak ${peak.toFixed(3)} (${relative.toExponential(2)} relative, ${ms} ms)`,
  );
  if (!ok) process.exitCode = 1;
  gpu.destroy();
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
