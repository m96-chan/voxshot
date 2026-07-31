import type { SynthesisEngine } from "../engine/types.js";
import { VoxShotError, isVoxShotError } from "../errors.js";
import type {
  EngineDescription,
  RequestMessage,
  ResponseMessage,
  RpcEndpoint,
  SerializedError,
} from "./protocol.js";
import { PROTOCOL_VERSION, isRequestMessage } from "./protocol.js";

/**
 * Serve `engine` over a message port, so inference runs off the UI thread.
 *
 * Typical worker entry point:
 *
 * ```ts
 * // tts.worker.ts
 * import { ChatterboxEngine, exposeEngine } from "voxshot";
 *
 * const engine = new ChatterboxEngine({
 *   onProgress: (progress) => emitProgress(progress),
 * });
 * const { emitProgress } = exposeEngine(engine, self as unknown as RpcEndpoint);
 * ```
 *
 * @returns a function that stops serving; it also carries `emitProgress` for
 * pushing model download progress to the main thread.
 */
export function exposeEngine(
  engine: SynthesisEngine,
  endpoint: RpcEndpoint,
): (() => void) & { emitProgress: (progress: Record<string, unknown>) => void } {
  const emitProgress = (progress: Record<string, unknown>): void => {
    endpoint.postMessage({ voxshot: PROTOCOL_VERSION, progress });
  };

  // One engine call at a time. ONNX Runtime sessions are not re-entrant, and
  // overlapping calls wedge them: an utterance cut mid-render used to leave a
  // synthesize running, and the next request re-entered the same session and
  // never came back (#67).
  const queue: Job[] = [];
  const jobs = new Map<number, Job>();
  let draining = false;

  const drain = async (): Promise<void> => {
    if (draining) {
      return;
    }
    draining = true;
    try {
      for (let job = queue.shift(); job !== undefined; job = queue.shift()) {
        if (job.cancelled) {
          // Never started, so there is nothing to interrupt — just answer.
          jobs.delete(job.request.id);
          fail(endpoint, job.request.id, new VoxShotError("The request was cancelled."));
          continue;
        }
        try {
          // Stays registered while it runs: a cancel arriving mid-render has
          // to be able to find it, which is the case that matters most.
          await handle(engine, job.request, endpoint, emitProgress, job.controller.signal);
        } finally {
          jobs.delete(job.request.id);
        }
      }
    } finally {
      draining = false;
    }
  };

  const cancel = (target: number): void => {
    const job = jobs.get(target);
    if (!job) {
      // Already finished, or never existed. A cancel racing its own reply is
      // expected, so this is deliberately not an error.
      return;
    }
    job.cancelled = true;
    job.controller.abort();
  };

  const listener = (event: { data: unknown }): void => {
    const request = event.data;
    if (!isRequestMessage(request)) {
      return;
    }

    if (request.method === "cancel") {
      cancel(request.target);
      reply(endpoint, request.id, undefined);
      return;
    }

    if (request.method === "dispose") {
      // Teardown jumps the queue. Waiting its turn behind a render is how a
      // wedged engine became impossible to shut down.
      for (const job of queue.splice(0)) {
        jobs.delete(job.request.id);
        job.controller.abort();
        fail(endpoint, job.request.id, new VoxShotError("The engine was disposed."));
      }
      for (const job of jobs.values()) {
        job.controller.abort();
      }
      void handle(engine, request, endpoint, emitProgress, undefined);
      return;
    }

    const job: Job = { request, controller: new AbortController(), cancelled: false };
    jobs.set(request.id, job);
    queue.push(job);
    void drain();
  };

  endpoint.addEventListener("message", listener);
  endpoint.start?.();

  const stop = (): void => {
    endpoint.removeEventListener("message", listener);
  };
  return Object.assign(stop, { emitProgress });
}

/** One queued request, with the handle used to abandon it. */
interface Job {
  readonly request: RequestMessage;
  readonly controller: AbortController;
  cancelled: boolean;
}

async function handle(
  engine: SynthesisEngine,
  request: RequestMessage,
  endpoint: RpcEndpoint,
  emitProgress: (progress: Record<string, unknown>) => void,
  signal: AbortSignal | undefined,
): Promise<void> {
  try {
    switch (request.method) {
      case "load": {
        await engine.load(request.device);
        const description: EngineDescription = {
          name: engine.name,
          sampleRate: engine.sampleRate,
        };
        // Tell the main thread the model is usable before the reply is read,
        // so a UI can drop its loading indicator as early as possible.
        emitProgress({ status: "ready", file: engine.name });
        reply(endpoint, request.id, description);
        return;
      }
      case "embed": {
        const embedded = await engine.embed({
          samples: request.samples,
          sampleRate: request.sampleRate,
        });
        // The vector travels as a transferable; any engine specific tensors go
        // through structured clone alongside it.
        const vector = embedded instanceof Float32Array ? embedded : embedded.vector;
        reply(endpoint, request.id, embedded, [vector.buffer as ArrayBuffer]);
        return;
      }
      case "synthesize": {
        const samples = await engine.synthesize({
          text: request.text,
          voice: request.voice,
          speed: request.speed,
          ...(request.expressiveness === undefined
            ? {}
            : { expressiveness: request.expressiveness }),
          // Engines that cannot interrupt a render simply ignore this; the
          // caller still stops waiting.
          ...(signal ? { signal } : {}),
        });
        reply(endpoint, request.id, samples, [samples.buffer as ArrayBuffer]);
        return;
      }
      case "dispose": {
        await engine.dispose();
        reply(endpoint, request.id, null);
        return;
      }
      default: {
        const { method } = request as { method: string };
        throw new VoxShotError(`Unknown worker method "${method}".`);
      }
    }
  } catch (cause) {
    fail(endpoint, request.id, cause);
  }
}

/** Answer a request with an error. */
function fail(endpoint: RpcEndpoint, id: number, cause: unknown): void {
  const message: ResponseMessage = {
    voxshot: PROTOCOL_VERSION,
    id,
    ok: false,
    error: serializeError(cause),
  };
  endpoint.postMessage(message);
}

function reply(
  endpoint: RpcEndpoint,
  id: number,
  result: unknown,
  transfer?: Transferable[],
): void {
  const message: ResponseMessage = { voxshot: PROTOCOL_VERSION, id, ok: true, result };
  endpoint.postMessage(message, transfer);
}

function serializeError(cause: unknown): SerializedError {
  if (isVoxShotError(cause)) {
    return { name: cause.name, message: cause.message, code: cause.code };
  }
  if (cause instanceof Error) {
    return { name: cause.name, message: cause.message, code: "UNKNOWN" };
  }
  return { name: "Error", message: String(cause), code: "UNKNOWN" };
}
