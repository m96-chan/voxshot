import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { Safetensors } from "./safetensors.js";

/**
 * The checkpoint, from wherever `huggingface_hub` put it.
 *
 * Node-only, and only for the tests — a browser build would fetch from the CDN
 * instead. The resolution goes through `refs/main` to the snapshot directory,
 * exactly as `spike/miocodec/weights-cache.ts` does.
 */

export interface Tensor {
  data: Float32Array;
  shape: number[];
}

/**
 * Memoised access to the checkpoint's tensors.
 *
 * `Safetensors.tensor` copies (and, for this bf16 checkpoint, widens) on every
 * call, by design — a view would pin the whole file. That makes it the wrong
 * thing to call per layer per forward. Memoised by name, each tensor is
 * converted once and every later reader gets the same array — which is also
 * what lets `model.ts`'s transpose cache key on array identity.
 */
export class Weights {
  private readonly cache = new Map<string, Tensor>();

  constructor(private readonly file: Safetensors) {}

  get(name: string): Tensor {
    let tensor = this.cache.get(name);
    if (!tensor) {
      const view = this.file.tensor(name);
      tensor = { data: view.data, shape: [...view.shape] };
      this.cache.set(name, tensor);
    }
    return tensor;
  }

  maybe(name: string): Tensor | null {
    return this.file.has(name) ? this.get(name) : null;
  }
}

export function loadWeights(repoId: string): Weights {
  const hub = join(homedir(), ".cache", "huggingface", "hub");
  const repo = `models--${repoId.replace("/", "--")}`;
  let file: string;
  try {
    const revision = readFileSync(join(hub, repo, "refs", "main"), "utf8").trim();
    file = join(hub, repo, "snapshots", revision, "model.safetensors");
  } catch {
    throw new Error(
      `${repoId}'s checkpoint is not in the HF cache. It arrives with the golden:\n` +
        `  cd spike/miotts && python3 dump_golden.py`,
    );
  }
  const bytes = readFileSync(file);
  return new Weights(
    Safetensors.parse(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer,
    ),
  );
}
