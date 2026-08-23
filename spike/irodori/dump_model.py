"""Dump everything in Irodori's checkpoint that is not the text backbone.

    cd spike/irodori && ../dacvae/.venv/bin/python dump_model.py

Writes `golden/model/` — one `.f32` per tensor plus `index.json`. Not in git.

562 tensors, 1.81 GB. With the backbone's 1.26 GB that is the whole 3.06 GB
checkpoint, which is the arithmetic behind the corrected download figure in the
README: ModernBERT-ja is inside this file, not beside it.

Where the size is:

    blocks               1433.8 MB   12 DiffusionBlocks, model_dim 1280
    speaker_encoder       242.0 MB   8 TextBlocks, dim 768
    duration_predictor     87.1 MB
    cond_module            28.8 MB
    caption_encoder         6.8 MB
    text_encoder            6.8 MB   the projector, not a second encoder
    the four norms etc.     0.4 MB

`config_json` travels in the safetensors metadata rather than a separate file,
and it is what says which of the model's many branches this checkpoint takes —
`residual_mlp` projector, `token_sum_dual_adarn_zero_no_aux` duration head,
caption conditioning on. Recording it here means the port reads the branch
rather than assuming one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "golden" / "model"
SKIP_PREFIX = "pretrained_text_backbone."


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Aratako/Irodori-TTS-v4.1-Small")
    parser.add_argument("--sha", action="store_true", help="hash every file (slow)")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    checkpoint = args.checkpoint
    if "/" in checkpoint and not Path(checkpoint).exists():
        checkpoint = hf_hub_download(checkpoint, "model.safetensors")

    OUT.mkdir(parents=True, exist_ok=True)
    tensors: dict = {}
    total = 0
    with safe_open(checkpoint, "pt") as handle:
        metadata = handle.metadata() or {}
        for key in handle.keys():
            if key.startswith(SKIP_PREFIX):
                continue
            tensor = handle.get_tensor(key).to(torch.float32).contiguous()
            path = OUT / f"{key}.f32"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tensor.numpy().tobytes())
            entry = {
                "shape": list(tensor.shape),
                "dtype": "float32",
                "bytes": path.stat().st_size,
            }
            if args.sha:
                entry["sha256"] = sha256_of(path)
            tensors[key] = entry
            total += entry["bytes"]

    payload = {
        "_rebuild": "cd spike/irodori && ../dacvae/.venv/bin/python dump_model.py",
        "checkpoint": args.checkpoint,
        # The branch the checkpoint takes, not the branches the code has.
        "config": json.loads(metadata["config_json"]),
        "tensors": tensors,
    }
    (OUT / "index.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(tensors)} tensors, {total / 1e6:.1f} MB to {OUT}/")


if __name__ == "__main__":
    main()
