"""Collect the encoder's global-path weights into one f32 safetensors file.

The reference encoder reads two checkpoints — the MioCodec safetensors
(`GlobalEncoder`, 50 tensors) and torchaudio's `wavlm_base_plus.pth` (the conv
frontend, feature projection, positional conv and transformer layers 1..2). The
TypeScript port should not reimplement torch's checkpoint loading, weight-norm
parametrisation or polyphase kernel synthesis just to read them, so this script
does those once, at export time:

- **pos_conv's weight-norm is folded.** The checkpoint stores `weight_g` /
  `weight_v`; reading `conv.weight` off the *loaded* module hands back the
  product torch actually convolves with.
- **The 24k -> 16k resample kernel is precomputed.** `transforms.Resample`
  synthesises a windowed-sinc polyphase kernel `[new, 1, width*2 + orig]` =
  `[2, 1, 23]` in `__init__`; exported as data, it is a conv1d like any other.

Output: `golden-encoder/encoder-weights.safetensors` (~117 MB, gitignored with
the golden it sits beside) plus `encoder-weights.json` recording every tensor's
shape and **both** source checkpoints' sha256. The digests are cross-checked
against `golden-encoder/index.json` rather than merely recorded — weights
exported from a different checkpoint than the golden was dumped from would make
every stage comparison chase a phantom.

Run with the same venv as the goldens:

    .venv/bin/python export_encoder_weights.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

from dump_golden import REPO_ID, load_model

OUT = Path(__file__).parent / "golden-encoder"
WEIGHTS = OUT / "encoder-weights.safetensors"
MANIFEST = OUT / "encoder-weights.json"

# The two WavLM layers the global branch reads (`global_ssl_layers = [1, 2]`,
# 1-based). Everything above layer 2 never runs in the port.
WAVLM_LAYERS = (0, 1)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_checkpoints() -> dict:
    """Both source files, digested, and checked against the golden's index.

    The golden's `index.json` already records what the reference ran with;
    recomputing here and *diffing* is what catches an exported-weights file
    taken after either checkpoint silently changed.
    """
    from huggingface_hub import hf_hub_download
    import torchaudio.pipelines as pipelines

    miocodec_path = Path(hf_hub_download(REPO_ID, "model.safetensors"))
    wavlm_path = Path(torch.hub.get_dir()) / "checkpoints" / pipelines.WAVLM_BASE_PLUS._path

    sources = {
        "miocodec": {"file": miocodec_path.name, "bytes": miocodec_path.stat().st_size, "sha256": sha256_of(miocodec_path)},
        "wavlm": {"file": wavlm_path.name, "bytes": wavlm_path.stat().st_size, "sha256": sha256_of(wavlm_path)},
    }

    index_path = OUT / "index.json"
    if not index_path.exists():
        raise SystemExit(
            f"{index_path} is missing — dump the golden first (.venv/bin/python "
            f"dump_encoder_golden.py); weights without a golden to check them "
            f"against verify nothing."
        )
    index = json.loads(index_path.read_text())
    for name, recorded in [("miocodec", index["checkpoint"]), ("wavlm", index["ssl_checkpoint"])]:
        if sources[name]["sha256"] != recorded["sha256"]:
            raise SystemExit(
                f"{name} checkpoint sha256 {sources[name]['sha256'][:12]} does not match "
                f"the golden's {recorded['sha256'][:12]} — the checkpoint changed since "
                f"the golden was dumped. Regenerate the golden, then re-export."
            )
    return sources


def collect(model) -> dict[str, torch.Tensor]:
    """Every tensor the global path needs, under flat dotted names.

    WavLM tensors keep torchaudio's own names under a `wavlm.` prefix; the
    MioCodec tensors keep the checkpoint's names verbatim. A reader holding
    either source open can grep a name straight back to its origin.
    """
    tensors: dict[str, torch.Tensor] = {}

    def put(name: str, tensor: torch.Tensor) -> None:
        tensors[name] = tensor.detach().to(torch.float32).contiguous().cpu()

    extractor = model.ssl_feature_extractor
    wavlm = extractor.model

    # -- resample: the precomputed polyphase kernel, [new=2, 1, 23], stride 3.
    put("resample.kernel", extractor.resampler.kernel)

    # -- conv frontend: 7 bias-free convs, GroupNorm(512) after block 0 only.
    for i, block in enumerate(wavlm.feature_extractor.conv_layers):
        put(f"wavlm.feature_extractor.conv_layers.{i}.conv.weight", block.conv.weight)
        if block.layer_norm is not None:
            put(f"wavlm.feature_extractor.conv_layers.{i}.layer_norm.weight", block.layer_norm.weight)
            put(f"wavlm.feature_extractor.conv_layers.{i}.layer_norm.bias", block.layer_norm.bias)

    encoder = wavlm.encoder
    put("wavlm.encoder.feature_projection.layer_norm.weight", encoder.feature_projection.layer_norm.weight)
    put("wavlm.encoder.feature_projection.layer_norm.bias", encoder.feature_projection.layer_norm.bias)
    put("wavlm.encoder.feature_projection.projection.weight", encoder.feature_projection.projection.weight)
    put("wavlm.encoder.feature_projection.projection.bias", encoder.feature_projection.projection.bias)

    # -- pos_conv, weight-norm FOLDED: `.weight` on the parametrised module is
    # the recomputed g * v / ||v|| product, not the raw `weight_v`.
    pos_conv = encoder.transformer.pos_conv_embed.conv
    put("wavlm.encoder.transformer.pos_conv_embed.conv.weight", pos_conv.weight)
    put("wavlm.encoder.transformer.pos_conv_embed.conv.bias", pos_conv.bias)

    # The encoder-level LayerNorm. `_get_wavlm_encoder` builds the Transformer
    # with `layer_norm_first = not encoder_layer_norm_first`, and base+ is
    # post-norm — so this norm runs *before* layer 1, right after the pos_conv
    # residual add. Easy to miss, impossible to leave out.
    put("wavlm.encoder.transformer.layer_norm.weight", encoder.transformer.layer_norm.weight)
    put("wavlm.encoder.transformer.layer_norm.bias", encoder.transformer.layer_norm.bias)

    for i in WAVLM_LAYERS:
        layer = encoder.transformer.layers[i]
        prefix = f"wavlm.encoder.transformer.layers.{i}"
        attn = layer.attention
        put(f"{prefix}.attention.attention.in_proj_weight", attn.attention.in_proj_weight)
        put(f"{prefix}.attention.attention.in_proj_bias", attn.attention.in_proj_bias)
        put(f"{prefix}.attention.attention.out_proj.weight", attn.attention.out_proj.weight)
        put(f"{prefix}.attention.attention.out_proj.bias", attn.attention.out_proj.bias)
        put(f"{prefix}.attention.gru_rel_pos_linear.weight", attn.gru_rel_pos_linear.weight)
        put(f"{prefix}.attention.gru_rel_pos_linear.bias", attn.gru_rel_pos_linear.bias)
        put(f"{prefix}.attention.gru_rel_pos_const", attn.gru_rel_pos_const)
        if attn.rel_attn_embed is not None:
            put(f"{prefix}.attention.rel_attn_embed.weight", attn.rel_attn_embed.weight)
        put(f"{prefix}.layer_norm.weight", layer.layer_norm.weight)
        put(f"{prefix}.layer_norm.bias", layer.layer_norm.bias)
        put(f"{prefix}.final_layer_norm.weight", layer.final_layer_norm.weight)
        put(f"{prefix}.final_layer_norm.bias", layer.final_layer_norm.bias)
        put(f"{prefix}.feed_forward.intermediate_dense.weight", layer.feed_forward.intermediate_dense.weight)
        put(f"{prefix}.feed_forward.intermediate_dense.bias", layer.feed_forward.intermediate_dense.bias)
        put(f"{prefix}.feed_forward.output_dense.weight", layer.feed_forward.output_dense.weight)
        put(f"{prefix}.feed_forward.output_dense.bias", layer.feed_forward.output_dense.bias)

    # -- the GlobalEncoder itself, straight from the loaded module.
    count_before = len(tensors)
    for name, parameter in model.global_encoder.named_parameters():
        put(f"global_encoder.{name}", parameter)
    global_count = len(tensors) - count_before
    assert global_count == 50, f"expected 50 GlobalEncoder tensors, collected {global_count}"

    return tensors


def main() -> int:
    sources = source_checkpoints()

    model = load_model()
    assert not model.training

    tensors = collect(model)

    from safetensors.torch import save_file

    OUT.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(WEIGHTS))

    resampler = model.ssl_feature_extractor.resampler
    manifest = {
        "repo_id": REPO_ID,
        "torch": torch.__version__,
        "sources": sources,
        "resample": {
            # `_apply_sinc_resample_kernel`'s constants, so the port pads
            # (width, width + orig) and strides by `orig` without rederiving.
            "orig": int(resampler.orig_freq // resampler.gcd),
            "new": int(resampler.new_freq // resampler.gcd),
            "width": int(resampler.width),
        },
        "tensors": {name: list(tensor.shape) for name, tensor in tensors.items()},
        "bytes": WEIGHTS.stat().st_size,
        "sha256": sha256_of(WEIGHTS),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    total = 0
    for name, tensor in tensors.items():
        total += tensor.numel()
        print(f"  {name:64s} {str(list(tensor.shape)):16s}")
    print(f"\n{len(tensors)} tensors, {total:,} parameters, "
          f"{WEIGHTS.stat().st_size / 1e6:.1f} MB -> {WEIGHTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
