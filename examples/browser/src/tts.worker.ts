import { ChatterboxEngine, exposeEngine, type RpcEndpoint } from "voxshot";

// Progress events fire only after the main thread requests load(), by which
// time exposeEngine below has already run, so `emitProgress` is initialised.
const engine = new ChatterboxEngine({
  onProgress: (progress) => emitProgress(progress as unknown as Record<string, unknown>),
});
const { emitProgress } = exposeEngine(engine, self as unknown as RpcEndpoint);
