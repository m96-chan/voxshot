import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);

import { ACTIVATION, activation } from "web-xpu-ops/ops/activation";

import { linear } from "./blocks.js";
import type { Context } from "./dit.js";
import { type Modulation, prepareGpu, velocityGpu } from "./dit-gpu.js";
import type { Tensor } from "./gpu.js";
import { Gpu } from "./gpu.js";
import { loadModelWeights } from "./model-weights.js";

/**
 * The GPU DiT against the same goldens `check-dit.ts` uses.
 *
 *     cd spike/irodori && npm run check:dit-gpu
 *
 * This checks the **whole velocity function** — `in_proj`, twelve blocks,
 * `out_norm`, `out_proj` — rather than one block, because that is the unit the
 * device version is built as: a block's intermediates never leave the device,
 * so there is nothing to compare in the middle without adding readbacks that
 * the real path does not do.
 *
 * The goldens are `out_proj`'s output at three recorded steps, which is exactly
 * that function's result.
 *
 * The tolerance is looser than the CPU path's 5e-6 and set from what this path
 * achieves — 8.9e-6 at worst across the four recorded steps. The device tiles
 * its matmuls and reduces in a tree where the reference walks K in one loop,
 * and the difference that produces is not an error in either.
 *
 * `--trace` compares the recorded intermediate stages as well. It is how the
 * two faults this had were found, and both were invisible from the end: the
 * context keys were concatenated in token-major order onto head-major self
 * keys, and RoPE ran once over a batch of three where its dispatch is sized for
 * one — leaving two thirds of the buffer never written. Neither threw.
 */

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN = join(HERE, "golden");
const MODEL = join(GOLDEN, "model");
const TOLERANCE = 5e-5;

function golden(name: string): Float32Array {
  const bytes = readFileSync(join(GOLDEN, `${name}.f32`));
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return new Float32Array(copy.buffer);
}

function flags(name: string): boolean[] {
  return Array.from(golden(name), (value) => value !== 0);
}

function report(label: string, mine: Float32Array, theirs: Float32Array): boolean {
  let worst = 0;
  let peak = 0;
  for (let index = 0; index < theirs.length; index += 1) {
    worst = Math.max(worst, Math.abs(mine[index]! - theirs[index]!));
    peak = Math.max(peak, Math.abs(theirs[index]!));
  }
  const relative = peak > 0 ? worst / peak : worst;
  const ok = relative <= TOLERANCE;
  console.log(
    `${ok ? "ok  " : "FAIL"} ${label.padEnd(24)} max |diff| ${worst.toExponential(2)} ` +
      `of peak ${peak.toFixed(2)}  (${relative.toExponential(2)} relative)`,
  );
  return ok;
}

/**
 * AdaLN's low-rank refinement, on the host.
 *
 * `[batch, dim]`-sized and about 2 MFLOP a block — small enough that moving it
 * to the device would add dispatches to buy nothing. What the device receives
 * is already `1 + scale` and `tanh(gate)`, so the graph there has no constants
 * in it.
 */
function modulationFor(
  gpu: Gpu,
  cond: Float32Array,
  w: { shiftDown: Float32Array; scaleDown: Float32Array; gateDown: Float32Array; shiftUp: Float32Array; scaleUp: Float32Array; gateUp: Float32Array; shiftBias: Float32Array; scaleBias: Float32Array; gateBias: Float32Array },
  dim: number,
  rank: number,
  batch: number,
): Modulation {
  const refine = (part: number, down: Float32Array, up: Float32Array, bias: Float32Array) => {
    const source = new Float32Array(batch * dim);
    for (let b = 0; b < batch; b += 1) {
      source.set(cond.subarray(b * 3 * dim + part * dim, b * 3 * dim + (part + 1) * dim), b * dim);
    }
    const out = linear(
      linear(activation({ input: source.slice(), kind: ACTIVATION.silu }), down, batch, dim, rank),
      up,
      batch,
      rank,
      dim,
      bias,
    );
    for (let i = 0; i < out.length; i += 1) out[i]! += source[i]!;
    return out;
  };
  const shift = refine(0, w.shiftDown, w.shiftUp, w.shiftBias);
  const scale = refine(1, w.scaleDown, w.scaleUp, w.scaleBias);
  const gate = refine(2, w.gateDown, w.gateUp, w.gateBias);
  for (let i = 0; i < scale.length; i += 1) scale[i]! += 1;
  for (let i = 0; i < gate.length; i += 1) gate[i] = Math.tanh(gate[i]!);
  return { scale: gpu.upload(scale), shift: gpu.upload(shift), gate: gpu.upload(gate) };
}

const trace = process.argv.includes("--trace");

async function main(): Promise<void> {
  const instance = create([]);
  const adapter = await instance.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  const info = adapter.info?.description ?? "unknown adapter";
  const gpu = Gpu.fromDevice(await Gpu.requestDevice(adapter), info, [instance, adapter]);
  gpu.begin();

  const weights = loadModelWeights(MODEL);
  const { dit } = weights;
  console.log(`DiT on ${info}: ${dit.blocks.length} blocks, dim ${dit.shape.dim}\n`);

  const index = JSON.parse(readFileSync(join(GOLDEN, "index.json"), "utf8")) as {
    tensors: Record<string, { shape: number[] }>;
  };
  // Every recorded step of the whole velocity function.
  const cases = Object.keys(index.tensors)
    .filter((name) => /^out_proj(__step\d+)?$/.test(name))
    .sort();

  let failures = 0;
  for (const name of cases) {
    const suffix = name.slice("out_proj".length);
    const at = (argument: string) => `blocks.0.in.${argument}${suffix}`;
    const [batch, tokens] = index.tensors[`in_proj.in.input${suffix}`]!.shape as [number, number];

    const contexts: Record<string, Context> = {
      text: { state: golden(at("text_state")), keep: flags(at("text_mask")) },
      speaker: { state: golden(at("speaker_state")), keep: flags(at("speaker_mask")) },
      caption: { state: golden(at("caption_state")), keep: flags(at("caption_mask")) },
    };

    const prepared = Date.now();
    const prep = prepareGpu({ gpu, weights, contexts, batch, tokens });
    await gpu.check(`${name}: preparing contexts`);
    const prepMs = Date.now() - prepared;

    const cond = golden(`cond_module${suffix}`);
    const modulation = dit.blocks.map((block) => ({
      attention: modulationFor(gpu, cond, block.attentionAdaLn, dit.shape.dim, dit.shape.rank, batch),
      mlp: modulationFor(gpu, cond, block.mlpAdaLn, dit.shape.dim, dit.shape.rank, batch),
    }));

    const started = Date.now();
    const x = gpu.upload(golden(`in_proj.in.input${suffix}`));
    // `--trace` copies each recorded stage into a buffer of its own before the
    // pooled slot it lives in is reused, then compares it. Off by default: it
    // is a fault-finding path, not part of what the port does.
    const traced: [string, Tensor][] = [];
    const wanted = new Set(["in_proj", "blocks.0", "blocks.5", "blocks.11", "out_norm"]);
    const out = velocityGpu(
      prep,
      x,
      modulation,
      trace
        ? (stage, tensor) => {
            if (!wanted.has(stage)) return;
            const copy = gpu.alloc(tensor.length);
            gpu.copy(tensor, 0, copy, 0, tensor.length * 4);
            traced.push([stage, copy]);
          }
        : undefined,
    );
    const mine = await gpu.read(out);
    await gpu.check(`${name}: running the velocity function`);
    const ms = Date.now() - started;

    for (const [stage, tensor] of traced) {
      report(`  ${stage}`, await gpu.read(tensor), golden(`${stage}${suffix}`));
    }

    if (!report(`${name} (batch ${batch})`, mine, golden(name))) failures += 1;
    console.log(
      `     ${prepMs} ms of context projection, ${ms} ms for the forward, ` +
        `${gpu.stats.dispatches} dispatches so far`,
    );
  }

  console.log();
  console.log(
    failures > 0
      ? `${failures} of ${cases.length} disagree`
      : `all ${cases.length} agree within ${TOLERANCE.toExponential(0)} of peak, on the device`,
  );
  if (failures > 0) process.exitCode = 1;
  gpu.destroy();
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
