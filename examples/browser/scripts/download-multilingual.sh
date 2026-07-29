#!/usr/bin/env bash
# Assemble a locally servable Chatterbox Multilingual model directory.
#
# The Hub repo onnx-community/chatterbox-multilingual-ONNX ships no
# config.json / preprocessor_config.json, so Transformers.js cannot load it
# remotely. Its ONNX graphs are architecture-identical to the English repo
# (the language_model graph is byte-identical), so the English configs are
# compatible: configs come from the English repo, everything else from the
# multilingual repo.
set -euo pipefail

EN="https://huggingface.co/onnx-community/chatterbox-ONNX/resolve/main"
MULTI="https://huggingface.co/onnx-community/chatterbox-multilingual-ONNX/resolve/main"
DEST="$(cd "$(dirname "$0")/.." && pwd)/public/models/chatterbox-multilingual"

mkdir -p "$DEST/onnx"

fetch() {
  local url="$1" out="$2"
  if [ -s "$out" ]; then
    echo "skip   $(basename "$out") (already downloaded)"
    return
  fi
  echo "fetch  $url"
  curl -fL --retry 3 --progress-bar -o "$out" "$url"
}

# Architecture configs: English repo (compatible, see header note).
fetch "$EN/config.json" "$DEST/config.json"
fetch "$EN/preprocessor_config.json" "$DEST/preprocessor_config.json"

# Tokenizer and generation config: multilingual repo.
fetch "$MULTI/tokenizer.json" "$DEST/tokenizer.json"
fetch "$MULTI/tokenizer_config.json" "$DEST/tokenizer_config.json"
fetch "$MULTI/generation_config.json" "$DEST/generation_config.json"
fetch "$MULTI/Cangjie5_TC.json" "$DEST/Cangjie5_TC.json"
fetch "$MULTI/default_voice.wav" "$DEST/default_voice.wav"

# Weights: multilingual repo. The fp32 language model is included because the
# q4 weights mis-generate on the WebGPU backend (near-instant STOP), while the
# same q4 file works on native CPU — fp32 avoids the affected kernels.
for f in \
  embed_tokens.onnx embed_tokens.onnx_data \
  speech_encoder.onnx speech_encoder.onnx_data \
  conditional_decoder.onnx conditional_decoder.onnx_data \
  language_model.onnx language_model.onnx_data \
  language_model_q4f16.onnx language_model_q4f16.onnx_data \
  language_model_q4.onnx language_model_q4.onnx_data; do
  fetch "$MULTI/onnx/$f" "$DEST/onnx/$f"
done

echo "done → $DEST"
