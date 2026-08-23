"""Write the DACVAE decoder's weights as flat `.f32` files the port can read.

    cd spike/dacvae && .venv/bin/python dump_weights.py

`weights.pth` is a torch pickle; nothing on the TypeScript side is going to
parse one. This writes the same shape of artifact the goldens use — one file
per tensor plus an index carrying shapes and digests.

## Only what a real run reaches

14.7 M parameters sit in branches `forward()` walks past (see `dump_golden.py`'s
doc for which). None of them is written, so the port cannot accidentally depend
on something the browser would then have to download.

The encoder's 27 M *are* written now. They were not at first — decoding never
touches them — but `spike/irodori` needs them: cloning a voice means turning a
reference clip into a latent, and a text-to-speech pipeline that can only speak
in one recorded voice is a partial pipeline. It is the decoder mirrored and
needs no operation the decoder did not already need.

`weight_g`/`weight_v` pairs are written as they are, not folded — folding is
`weights.ts`' job, and doing it here would hide from the port the fact that the
checkpoint is weight-normalised at all.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

from huggingface_hub import hf_hub_download  # noqa: E402

REPO_ID = "Aratako/Semantic-DACVAE-Japanese-32dim"
HERE = Path(__file__).resolve().parent
OUT = HERE / "weights"

# `DecoderBlock.forward` walks chunks where j % 2 == 0, with chunk size 2.
MAIN_PATH_BLOCK_INDICES = {0, 1, 4, 5, 8, 9}
BLOCK_INDEX = re.compile(r"^decoder\.model\.\d+\.block\.(\d+)\.")
# The output tail: WatermarkEncoderBlock.pre[0..2], with pre[3] swapped for
# Identity by `forward_no_conv`.
TAIL_PREFIX = "decoder.wm_model.encoder_block.pre."
TAIL_KEPT = {0, 1, 2}


# The encode path, added after `spike/irodori` needed it: cloning a voice means
# turning a reference clip into a latent, and only the decode side was dumped
# the first time. It is the decoder mirrored — conv, snake, conv, over
# `encoder_rates` [2, 8, 10, 12] — so it needs no operation the decoder did not
# already need.
ENCODER_PREFIXES = ("encoder.", "quantizer.in_proj")


def wanted(key: str) -> bool:
    if key.startswith(ENCODER_PREFIXES):
        return True
    if key.startswith("quantizer.out_proj"):
        return True  # 32 -> 1024, the decode side of the VAE bottleneck
    if key.startswith(TAIL_PREFIX):
        return int(key[len(TAIL_PREFIX) :].split(".")[0]) in TAIL_KEPT
    if key.startswith("decoder.wm_model"):
        return False
    match = BLOCK_INDEX.match(key)
    if match:
        return int(match.group(1)) in MAIN_PATH_BLOCK_INDICES
    return key.startswith("decoder.")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    checkpoint = Path(hf_hub_download(REPO_ID, "weights.pth"))
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state, metadata = blob["state_dict"], blob["metadata"]["kwargs"]

    tensors: dict = {}
    kept = skipped = 0
    for key, value in state.items():
        if not wanted(key):
            skipped += value.numel()
            continue
        kept += value.numel()
        array = value.detach().to(torch.float32).cpu().contiguous()
        path = OUT / f"{key}.f32"
        path.write_bytes(array.numpy().tobytes())
        tensors[key] = {
            "shape": list(array.shape),
            "dtype": "float32",
            "bytes": path.stat().st_size,
            "sha256": sha256_of(path),
        }

    index = {
        "repo_id": REPO_ID,
        "torch": torch.__version__,
        "config": {
            "sample_rate": int(metadata["sample_rate"]),
            "hop_length": 1,
            "latent_dim": int(metadata["codebook_dim"]),
            "decoder_dim": int(metadata["decoder_dim"]),
            "decoder_rates": list(metadata["decoder_rates"]),
            "encoder_dim": int(metadata["encoder_dim"]),
            "encoder_rates": list(metadata["encoder_rates"]),
        },
        "checkpoint": {"file": "weights.pth", "sha256": sha256_of(checkpoint)},
        "tensors": tensors,
    }
    hop = 1
    for rate in metadata["decoder_rates"]:
        hop *= int(rate)
    index["config"]["hop_length"] = hop

    (OUT / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"wrote {len(tensors)} tensors to {OUT}")
    print(f"  kept    {kept / 1e6:7.2f} M parameters")
    print(f"  skipped {skipped / 1e6:7.2f} M (watermark branches)")


if __name__ == "__main__":
    main()
