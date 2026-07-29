import { env } from "@huggingface/transformers";
import { ChatterboxEngine, exposeEngine, type RpcEndpoint } from "zerovox";

// Serve the model from this origin: the Hub copy of the multilingual repo is
// missing the config files Transformers.js needs, so a locally assembled
// directory (scripts/download-multilingual.sh) is used instead.
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = "/models/";

const engine = new ChatterboxEngine({
  modelId: "chatterbox-multilingual",
  // The multilingual q4 weights mis-generate on WebGPU (the model emits STOP
  // after a handful of speech tokens; the same file is fine on native CPU),
  // so the language model runs at fp32. Local serving makes the bigger file
  // cheap, and the demo machine class that runs WebGPU can afford it.
  dtype: { language_model: "fp32" },
  onProgress: (progress) => emitProgress(progress as unknown as Record<string, unknown>),
});
const { emitProgress } = exposeEngine(engine, self as unknown as RpcEndpoint);
