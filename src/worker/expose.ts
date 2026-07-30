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

  const listener = (event: { data: unknown }): void => {
    const request = event.data;
    if (!isRequestMessage(request)) {
      return;
    }
    void handle(engine, request, endpoint, emitProgress);
  };

  endpoint.addEventListener("message", listener);
  endpoint.start?.();

  const stop = (): void => {
    endpoint.removeEventListener("message", listener);
  };
  return Object.assign(stop, { emitProgress });
}

async function handle(
  engine: SynthesisEngine,
  request: RequestMessage,
  endpoint: RpcEndpoint,
  emitProgress: (progress: Record<string, unknown>) => void,
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
    const message: ResponseMessage = {
      voxshot: PROTOCOL_VERSION,
      id: request.id,
      ok: false,
      error: serializeError(cause),
    };
    endpoint.postMessage(message);
  }
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
