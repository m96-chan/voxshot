import type { SynthesisEngine } from "../engine/types.js";
import { VoxShotError, isVoxShotError } from "../errors.js";
import type {
  EngineDescription,
  RequestMessage,
  ResponseMessage,
  RpcEndpoint,
  SerializedError,
} from "./protocol.js";
import { CONTROL_METHODS, PROTOCOL_VERSION, isRequestMessage } from "./protocol.js";

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
  let running: Job | undefined;
  let draining = false;
  let disposed = false;
  let disposeWhenIdle = false;

  /** Answer a job that will never reach the engine. */
  const abandon = (job: Job, reason: string): void => {
    if (jobs.get(job.request.id) === job) {
      jobs.delete(job.request.id);
    }
    job.controller.abort();
    fail(endpoint, job.request.id, new VoxShotError(reason));
  };

  const drain = async (): Promise<void> => {
    if (draining) {
      return;
    }
    draining = true;
    try {
      for (let job = queue.shift(); job !== undefined; job = queue.shift()) {
        if (job.cancelled || disposed) {
          abandon(job, job.cancelled ? "The request was cancelled." : "The engine was disposed.");
          continue;
        }
        running = job;
        try {
          // `handle` answers its own failures and `fail` cannot throw, so this
          // is not expected to reject. Kept so that a future change there
          // cannot strand the rest of the queue, but deliberately silent: a
          // job that reached here has already been answered or is
          // unanswerable.
          await handle(engine, job.request, endpoint, emitProgress, job.controller.signal);
        } catch {
          // Nothing left to try; see above.
        } finally {
          running = undefined;
          if (disposeWhenIdle) {
            disposeWhenIdle = false;
            try {
              await engine.dispose();
            } catch {
              // Nobody is waiting for this: the caller was answered when the
              // teardown was deferred.
            }
          }
          // Compare by identity: a second engine sharing this endpoint starts
          // its ids at 1 too, so deleting by id alone can remove its entry.
          if (jobs.get(job.request.id) === job) {
            jobs.delete(job.request.id);
          }
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

  /**
   * Abandon everything and dispose the engine.
   *
   * Teardown jumps the queue — waiting its turn behind a wedged render is how
   * consumers were left with no way out.
   *
   * Disposing an ONNX session while a call is still inside it is the exact
   * overlap the queue exists to prevent, and abort is only advisory: an engine
   * that cannot interrupt keeps running whatever it started. So neither
   * waiting nor disposing underneath it is acceptable. Instead the engine is
   * marked unusable and answered immediately, and the disposal itself is left
   * for whenever the running call finishes. A caller that cannot wait for that
   * terminates the worker, which is the documented recovery.
   */
  const teardown = async (request: RequestMessage): Promise<void> => {
    if (disposed) {
      // Already torn down. Repeating the work would start a second
      // engine.dispose(), which is the overlap this queue exists to prevent.
      reply(endpoint, request.id, null);
      return;
    }
    disposed = true;
    for (const job of queue.splice(0)) {
      abandon(job, "The engine was disposed.");
    }

    if (running) {
      running.controller.abort();
      disposeWhenIdle = true;
      // Not inside `handle`, so a throw here would go unanswered and surface
      // as an unhandled rejection rather than becoming a failure notice.
      try {
        reply(endpoint, request.id, null);
      } catch (cause) {
        fail(endpoint, request.id, cause);
      }
      return;
    }
    await handle(engine, request, endpoint, emitProgress, undefined);
  };

  const listener = (event: { data: unknown }): void => {
    const request = event.data;
    if (!isRequestMessage(request)) {
      return;
    }

    if (CONTROL_METHODS.has(request.method)) {
      if (request.method === "cancel") {
        cancel(request.target);
        reply(endpoint, request.id, undefined);
      } else {
        void teardown(request);
      }
      return;
    }

    if (disposed) {
      // Silence here is what left callers waiting forever once teardown had
      // parked the drain loop.
      fail(endpoint, request.id, new VoxShotError("The engine was disposed."));
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
    // Leaving the queue to run would keep the engine working for a server
    // that has been told to stop, and those jobs are no longer reachable by
    // `cancel` because the listener is gone.
    disposed = true;
    for (const job of queue.splice(0)) {
      abandon(job, "The worker stopped serving.");
    }
    running?.controller.abort();
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
          // Only when the engine tracks it: reporting the requested device
          // would defeat the point, which is to surface a degrade.
          ...(engine.loadedDevice === undefined ? {} : { device: engine.loadedDevice }),
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

/**
 * Answer a request with an error.
 *
 * Delivery itself can fail — `postMessage` throws on a closed port and on a
 * payload that cannot be cloned — and this is the last chance to answer, so
 * there is nothing useful to do with that failure. Swallowing it keeps one
 * unanswerable request from taking the rest of the queue with it; the caller
 * is left waiting either way.
 */
function fail(endpoint: RpcEndpoint, id: number, cause: unknown): void {
  const message: ResponseMessage = {
    voxshot: PROTOCOL_VERSION,
    id,
    ok: false,
    error: serializeError(cause),
  };
  post(endpoint, message);
}

/** `postMessage` that cannot throw. See {@link fail}. */
function post(endpoint: RpcEndpoint, message: unknown, transfer?: Transferable[]): void {
  try {
    endpoint.postMessage(message, transfer);
  } catch {
    // Nothing left to try: reporting a delivery failure needs delivery.
  }
}

function reply(
  endpoint: RpcEndpoint,
  id: number,
  result: unknown,
  transfer?: Transferable[],
): void {
  const message: ResponseMessage = { voxshot: PROTOCOL_VERSION, id, ok: true, result };
  // Deliberately allowed to throw: `handle` turns a failed reply into a
  // failure notice, which is more useful to the caller than silence.
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
