import type { ResolvedDevice } from "../device.js";
import type { VoxShotErrorCode } from "../errors.js";
import type { VoiceEmbedding } from "../voice/types.js";

/**
 * Protocol version, carried on every message as the `voxshot` field.
 *
 * It doubles as a marker, so a VoxShot worker can share a port with an
 * application's own messages without either side misreading the other.
 */
export const PROTOCOL_VERSION = 1;

export type EngineRequest =
  | { readonly method: "load"; readonly device: ResolvedDevice }
  | { readonly method: "embed"; readonly samples: Float32Array; readonly sampleRate: number }
  | {
      readonly method: "synthesize";
      readonly text: string;
      readonly voice: VoiceEmbedding;
      readonly speed: number;
    }
  | { readonly method: "dispose" };

/** A request as it travels over the port. */
export type RequestMessage = EngineRequest & {
  readonly voxshot: number;
  readonly id: number;
};

/** Structured-cloneable stand-in for an Error. */
export interface SerializedError {
  readonly name: string;
  readonly message: string;
  readonly code: VoxShotErrorCode;
}

export type ResponseMessage =
  | { readonly voxshot: number; readonly id: number; readonly ok: true; readonly result: unknown }
  | {
      readonly voxshot: number;
      readonly id: number;
      readonly ok: false;
      readonly error: SerializedError;
    };

/** Progress emitted by the worker outside of any request / response pair. */
export interface ProgressMessage {
  readonly voxshot: number;
  readonly progress: Record<string, unknown>;
}

/** Reply to `load`, so the main thread learns the engine's identity. */
export interface EngineDescription {
  readonly name: string;
  readonly sampleRate: number;
}

/**
 * The subset of `MessagePort` / `Worker` this transport needs.
 *
 * Browser `Worker` and `MessagePort` both satisfy it, as does Node's
 * `MessagePort`, which is what the tests drive.
 */
export interface RpcEndpoint {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  addEventListener(type: "message", listener: (event: { data: unknown }) => void): void;
  removeEventListener(type: "message", listener: (event: { data: unknown }) => void): void;
  start?(): void;
}

export function isRequestMessage(value: unknown): value is RequestMessage {
  return (
    isProtocolMessage(value) &&
    typeof (value as RequestMessage).id === "number" &&
    typeof (value as RequestMessage).method === "string"
  );
}

export function isResponseMessage(value: unknown): value is ResponseMessage {
  return (
    isProtocolMessage(value) &&
    typeof (value as ResponseMessage).id === "number" &&
    typeof (value as ResponseMessage).ok === "boolean"
  );
}

export function isProgressMessage(value: unknown): value is ProgressMessage {
  return isProtocolMessage(value) && typeof (value as ProgressMessage).progress === "object";
}

function isProtocolMessage(value: unknown): value is { voxshot: number } {
  return (
    typeof value === "object" &&
    value !== null &&
    (value as { voxshot?: unknown }).voxshot === PROTOCOL_VERSION
  );
}
