import { CHATTERBOX_SAMPLE_RATE } from "../engine/chatterbox/chatterbox-engine.js";
import type { SynthesisEngine, SynthesisRequest } from "../engine/types.js";
import { VoxShotError } from "../errors.js";
import type { PcmAudio } from "../platform.js";
import type { EngineDescription, EngineRequest, ResponseMessage, RpcEndpoint } from "./protocol.js";
import { PROTOCOL_VERSION, isProgressMessage, isResponseMessage } from "./protocol.js";

export interface WorkerSynthesisEngineOptions {
  /**
   * Sample rate reported before `load()` has answered.
   *
   * @defaultValue 24000 (Chatterbox)
   */
  sampleRate?: number;
  /** Engine name reported before `load()` has answered. */
  name?: string;
  /** Receives progress events pushed by the worker (model download, ready). */
  onProgress?: (progress: Record<string, unknown>) => void;
  /**
   * How long to wait for a reply before failing. `0` waits forever.
   *
   * @defaultValue 0
   */
  timeoutMs?: number;
}

interface Pending {
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
  /**
   * Release everything attached to this call: its timeout and its abort
   * listener. The listener matters because one `AbortController` covers a
   * whole utterance, so without this the listeners grow with the chunk count.
   */
  settle: () => void;
}

/**
 * Main-thread {@link SynthesisEngine} that forwards every call to an engine
 * running inside a Web Worker.
 *
 * ```ts
 * const worker = new Worker(new URL("./tts.worker.ts", import.meta.url), { type: "module" });
 * const engine = new WorkerSynthesisEngine(worker);
 * const tts = await VoxShot.create({ engine });
 * ```
 *
 * Audio crosses the boundary as a transferable buffer, but always as a copy —
 * the caller's `Float32Array` is never detached.
 */
export class WorkerSynthesisEngine implements SynthesisEngine {
  #name: string;
  #sampleRate: number;

  readonly #endpoint: RpcEndpoint;
  readonly #onProgress: ((progress: Record<string, unknown>) => void) | undefined;
  readonly #timeoutMs: number;
  readonly #pending = new Map<number, Pending>();
  readonly #listener: (event: { data: unknown }) => void;

  #nextId = 1;
  #connected = true;

  constructor(endpoint: RpcEndpoint, options: WorkerSynthesisEngineOptions = {}) {
    this.#endpoint = endpoint;
    this.#name = options.name ?? "worker";
    this.#sampleRate = options.sampleRate ?? CHATTERBOX_SAMPLE_RATE;
    this.#onProgress = options.onProgress;
    this.#timeoutMs = options.timeoutMs ?? 0;

    this.#listener = (event) => this.#receive(event.data);
    endpoint.addEventListener("message", this.#listener);
    endpoint.start?.();
  }

  /** Name of the worker side engine; known accurately after `load()`. */
  get name(): string {
    return this.#name;
  }

  /** Sample rate of the worker side engine; known accurately after `load()`. */
  get sampleRate(): number {
    return this.#sampleRate;
  }

  async load(device: "webgpu" | "wasm"): Promise<void> {
    const description = (await this.#send({ method: "load", device })) as EngineDescription;
    this.#name = description.name;
    this.#sampleRate = description.sampleRate;
  }

  async embed(audio: PcmAudio): Promise<Float32Array> {
    // Copy first: transferring the caller's buffer would detach it.
    const samples = Float32Array.from(audio.samples);
    const vector = (await this.#send({ method: "embed", samples, sampleRate: audio.sampleRate }, [
      samples.buffer as ArrayBuffer,
    ])) as Float32Array;
    return vector;
  }

  async synthesize(request: SynthesisRequest): Promise<Float32Array> {
    return (await this.#send(
      {
        method: "synthesize",
        text: request.text,
        voice: request.voice,
        speed: request.speed,
        ...(request.expressiveness === undefined
          ? {}
          : { expressiveness: request.expressiveness }),
      },
      undefined,
      request.signal,
    )) as Float32Array;
  }

  async dispose(): Promise<void> {
    await this.#send({ method: "dispose" });
  }

  /**
   * Stop listening and fail every in-flight call. The worker itself is not
   * terminated — its lifetime belongs to whoever created it.
   */
  disconnect(): void {
    if (!this.#connected) {
      return;
    }
    this.#connected = false;
    this.#endpoint.removeEventListener("message", this.#listener);

    const error = new VoxShotError("The worker connection was closed.");
    for (const [id, pending] of this.#pending) {
      pending.settle();
      this.#pending.delete(id);
      pending.reject(error);
    }
  }

  /**
   * Tell the worker to drop a request we have stopped waiting for.
   *
   * Every path that abandons a request needs this, not just cancellation:
   * the worker runs one call at a time, so an abandoned render holds the only
   * slot and everything behind it waits for work nobody wants (#80).
   */
  #abandon(id: number): void {
    if (!this.#connected) {
      return;
    }
    this.#endpoint.postMessage({
      voxshot: PROTOCOL_VERSION,
      id: this.#nextId++,
      method: "cancel",
      target: id,
    });
  }

  async #send(
    request: EngineRequest,
    transfer?: Transferable[],
    signal?: AbortSignal,
  ): Promise<unknown> {
    if (!this.#connected) {
      throw new VoxShotError("This worker engine has been disconnected.");
    }

    const id = this.#nextId++;
    return new Promise<unknown>((resolve, reject) => {
      const onAbort = (): void => {
        const pending = this.#pending.get(id);
        if (!pending) {
          return;
        }
        pending.settle();
        this.#pending.delete(id);
        this.#abandon(id);
        reject(new VoxShotError("The request was cancelled."));
      };
      if (signal) {
        if (signal.aborted) {
          queueMicrotask(onAbort);
        } else {
          signal.addEventListener("abort", onAbort);
        }
      }
      const timer =
        this.#timeoutMs > 0
          ? setTimeout(() => {
              const pending = this.#pending.get(id);
              pending?.settle();
              this.#pending.delete(id);
              // The worker is still rendering; without this it keeps the only
              // execution slot and every later request queues behind it.
              this.#abandon(id);
              reject(
                new VoxShotError(
                  `The worker did not answer "${request.method}" within ${this.#timeoutMs}ms.`,
                ),
              );
            }, this.#timeoutMs)
          : undefined;

      const settle = (): void => {
        if (timer !== undefined) {
          clearTimeout(timer);
        }
        signal?.removeEventListener("abort", onAbort);
      };
      this.#pending.set(id, { resolve, reject, settle });
      this.#endpoint.postMessage({ voxshot: PROTOCOL_VERSION, id, ...request }, transfer);
    });
  }

  #receive(data: unknown): void {
    if (isProgressMessage(data)) {
      this.#onProgress?.(data.progress);
      return;
    }
    if (!isResponseMessage(data)) {
      return;
    }

    const pending = this.#pending.get(data.id);
    if (!pending) {
      return;
    }
    pending.settle();
    this.#pending.delete(data.id);

    if (data.ok) {
      pending.resolve(data.result);
    } else {
      pending.reject(deserializeError(data));
    }
  }
}

function deserializeError(response: Extract<ResponseMessage, { ok: false }>): VoxShotError {
  const error = new VoxShotError(response.error.message, response.error.code);
  error.name = response.error.name;
  return error;
}
