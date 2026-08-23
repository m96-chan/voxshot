"""Dump the whole Irodori pipeline, stage by stage, from its own reference.

    cd spike/irodori && ../dacvae/.venv/bin/python dump_golden.py \\
        --text "……ん。来たんだ。" --ref-wav path/to/48k.wav

Writes `golden/index.json` plus one `.f32` per tensor. Not in git — regenerable,
and large.

## Hooks, not a re-implementation

`InferenceRuntime.synthesize` is five hundred lines branching over candidates,
CFG modes, KV scaling and four forms of reference input. Reading all of it to
decide what to record would mean deciding what the pipeline *is*, which is the
thing the port is supposed to learn from the reference rather than assert about
it. So this registers forward hooks on named submodules and records what
actually ran — the same way `spike/dacvae/dump_golden.py` does for the decoder.

Two consequences worth knowing.

**Only modules that fire get dumped.** A branch this request did not take leaves
no golden, which is correct: the port is checked against the path the reference
walked for these arguments, not against every path it has.

**The flow-matching loop fires the DiT once per step.** Recording all of them
would be 32x the bytes for very little — the steps differ only in the timestep
and in their own input. The first, the middle and the last are kept, which is
enough to catch a schedule that drifts and cheap enough to keep.

## Determinism

The seed is fixed and passed through, and the reference's own sampler runs. Two
runs with the same arguments produce identical bytes, which the sha256 in the
index pins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
OUT = HERE / "golden"

# The reference is a clone rather than a dependency: it is not on PyPI, and
# taking it as a path makes the revision something `index.json` can record.
REFERENCE = Path(os.environ.get("IRODORI_REPO", "")).expanduser()

# Which modules are worth a golden. Anything deeper is a detail of a block the
# port will reproduce as a whole; anything shallower says only "the model ran".
INTERESTING = (
    "pretrained_text_backbone.backbone",  # ModernBERT-ja, all 25 layers
    "text_encoder",                       # the projector onto the DiT's context
    "speaker_encoder",                    # the reference-latent encoder
    "duration_predictor",
    "blocks.0",                           # first, middle and last DiffusionBlock
    "blocks.5",
    "blocks.11",
    "out_norm",
    "out_proj",
)

# DiT steps to keep. The loop is 32 by default; these three bracket it.
KEEP_STEPS = (0, 16, 31)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(name: str, tensor: torch.Tensor, into: dict) -> None:
    array = tensor.detach().to(torch.float32).cpu().contiguous()
    path = OUT / f"{name}.f32"
    path.write_bytes(array.numpy().tobytes())
    into[name] = {
        "shape": list(array.shape),
        "dtype": "float32",
        "bytes": path.stat().st_size,
        "sha256": sha256_of(path),
    }
    print(f"  {name:44s} {tuple(array.shape)}")


def first_tensor(value: object) -> torch.Tensor | None:
    """The output of a module, whatever shape the module chose to return it in."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--ref-wav", required=True)
    parser.add_argument("--checkpoint", default="Aratako/Irodori-TTS-v4.1-Small")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--list-modules", action="store_true", help="print the tree and stop")
    args = parser.parse_args()

    if not REFERENCE.is_dir():
        raise SystemExit(
            "Set IRODORI_REPO to a clone of https://github.com/Aratako/Irodori-TTS\n"
            "  git clone https://github.com/Aratako/Irodori-TTS\n"
            "  IRODORI_REPO=$PWD/Irodori-TTS ../dacvae/.venv/bin/python dump_golden.py ..."
        )
    sys.path.insert(0, str(REFERENCE))

    from huggingface_hub import hf_hub_download  # noqa: E402
    from irodori_tts.inference_runtime import (  # noqa: E402
        InferenceRuntime,
        RuntimeKey,
        SamplingRequest,
    )

    checkpoint = args.checkpoint
    if "/" in checkpoint and not Path(checkpoint).exists():
        checkpoint = hf_hub_download(checkpoint, "model.safetensors")

    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=str(checkpoint),
            model_device="cuda",
            codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
            model_precision="fp32",
            codec_device="cuda",
            codec_precision="fp32",
            codec_deterministic_encode=True,
            codec_deterministic_decode=True,
            compile_model=False,
            compile_dynamic=False,
        )
    )

    if args.list_modules:
        for name, module in runtime.model.named_modules():
            if name.count(".") <= 2:
                print(f"{name or '<root>':52s} {type(module).__name__}")
        return

    OUT.mkdir(exist_ok=True)
    tensors: dict = {}
    fired: dict[str, int] = {}

    def hook(name: str):
        def record(_module, _inputs, output):
            index = fired.get(name, 0)
            fired[name] = index + 1
            # A module inside the flow loop fires once per step; keep three.
            if index > 0 and index not in KEEP_STEPS:
                return
            tensor = first_tensor(output)
            if tensor is None or tensor.numel() == 0:
                return
            suffix = "" if index == 0 else f"__step{index}"
            save(f"{name}{suffix}", tensor, tensors)

        return record

    handles = []
    for name, module in runtime.model.named_modules():
        if any(name == key or name.endswith("." + key) for key in INTERESTING):
            handles.append(module.register_forward_hook(hook(name)))
    print(f"hooked {len(handles)} modules\n")

    result = runtime.synthesize(
        SamplingRequest(
            text=args.text,
            ref_wav=args.ref_wav,
            seed=args.seed,
            num_steps=args.num_steps,
        )
    )
    for handle in handles:
        handle.remove()

    save("audio", result.audio, tensors)

    index = {
        "checkpoint": args.checkpoint,
        "text": args.text,
        "ref_wav": Path(args.ref_wav).name,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "sample_rate": int(result.sample_rate),
        "torch": torch.__version__,
        "timings": {name: ms for name, ms in result.stage_timings},
        "fired": fired,
        "tensors": tensors,
    }
    (OUT / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT}/index.json — {len(tensors)} tensors")


if __name__ == "__main__":
    main()
