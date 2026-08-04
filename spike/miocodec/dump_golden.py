"""Dump per-stage golden tensors from the reference MioCodec decoder.

The goldens themselves are **not in git** — they are ~10 MB of raw floats that
would say nothing in a diff, and they are reproducible from here. This script is
in git; its output is ignored. Regenerate with:

    .venv/bin/python dump_golden.py

Why per stage, and not just the waveform: a port checked only at the output
learns that something is wrong and nothing about where. Every error in every
stage arrives at the waveform together, and the transformer stages are the ones
most likely to be subtly wrong (RoPE pairing, AdaLN ordering, window masking) in
ways that still sound like speech.

Two things are pinned deliberately.

**CPU, not CUDA.** `miocodec.model._get_autocast_context` turns on bfloat16
autocast for CUDA, so a golden taken there would carry bf16 rounding into every
comparison and the port would be measured against a blurred target. On CPU the
context is a no-op and the reference runs in fp32.

**torch 2.10.** The same version every convention in web-xpu-ops was verified
against. The venv pulls a newer torch transitively through `julius`; it is
uninstalled again so the system 2.10 shows through.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import torch

from miocodec.model import MioCodecModel

REPO_ID = "Aratako/MioCodec-25Hz-24kHz"

# The AdaLN-Zero conditioning width, and therefore the speaker embedding width.
GLOBAL_DIM = 128
OUT = Path(__file__).parent / "golden"

SEED = 20260804

# Two cases, because one of the stages is invisible in the natural one.
#
# `aligned` is what `decode` does on its own: 25 tokens is one second at 25 Hz,
# the conv upsample doubles that to 50 frames, and the STFT length derived from
# the audio length is also 50 — so `F.interpolate` between them resamples 50 to
# 50 and is the **identity**. A port that dropped the interpolate entirely would
# pass every comparison in that case.
#
# `resampled` asks for a shorter audio length, so the interpolate genuinely runs
# (50 frames in, 41 out) and a port that skips it cannot agree.
#
# Token counts stay above 65 nowhere — the wave decoder's attention window is 65
# and both cases sit under it, so neither observes the window mask. `windowed`
# exists for that: 80 tokens puts 160 frames through a 65-wide window, and a port
# that attends to everything is then wrong in the middle of the sequence rather
# than nowhere.
CASES = [
    {"name": "aligned", "tokens": 25, "target_audio_length": None},
    {"name": "resampled", "tokens": 25, "target_audio_length": 20000},
    {"name": "windowed", "tokens": 80, "target_audio_length": None},
]


def write(directory: Path, name: str, tensor: torch.Tensor, manifest: dict) -> None:
    """One tensor to one file of raw little-endian f32, shape in the manifest."""
    array = tensor.detach().to(torch.float32).contiguous().cpu().numpy()
    path = directory / f"{name}.f32"
    path.write_bytes(array.tobytes())
    manifest["tensors"][name] = {
        "shape": list(array.shape),
        "dtype": "float32",
        "bytes": path.stat().st_size,
        # Cheap enough to compute and the only way a half-written file is
        # distinguishable from a correct one at load time.
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    model = load_model()
    config = model.config

    # Dropout is 0.1 in the resnet stacks. eval() in load_model turns it off;
    # asserted rather than trusted, because a golden taken in train mode is
    # noise that looks like data and every later comparison would chase it.
    assert not model.training, "model must be in eval mode"

    levels = 1
    for level in config_levels(model):
        levels *= level

    index = {
        "repo_id": REPO_ID,
        "seed": SEED,
        "torch": torch.__version__,
        "codebook_size": levels,
        "config": {
            "sample_rate": config.sample_rate,
            "n_fft": config.n_fft,
            "hop_length": config.hop_length,
            "istft_padding": config.istft_padding,
            "wave_decoder_dim": config.wave_decoder_dim,
            "wave_resnet_num_blocks": config.wave_resnet_num_blocks,
            "wave_resnet_kernel_size": config.wave_resnet_kernel_size,
            "wave_resnet_num_groups": config.wave_resnet_num_groups,
            "wave_upsample_factor": config.wave_upsample_factor,
            "wave_interpolation_mode": config.wave_interpolation_mode,
            "wave_upsampler_factors": config.wave_upsampler_factors,
            "wave_decoder_window": window_size(model.wave_decoder),
            "wave_prenet_window": window_size(model.wave_prenet),
        },
        "checkpoint": checkpoint_provenance(REPO_ID),
        "cases": {},
    }

    for case in CASES:
        index["cases"][case["name"]] = dump_case(model, case, levels)

    (OUT / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"\nwrote {OUT}")
    return 0


def dump_case(model, case: dict, levels: int) -> dict:
    """One decode, every stage of it, into `golden/<name>/`."""
    # Seeded per case rather than once for the run, so adding or reordering a
    # case does not silently change the inputs of the others -- a golden whose
    # numbers move for an unrelated reason is a golden nobody trusts.
    torch.manual_seed(SEED + sum(ord(c) for c in case["name"]))

    num_tokens = case["tokens"]
    directory = OUT / case["name"]
    directory.mkdir(parents=True, exist_ok=True)

    tokens = torch.randint(0, levels, (num_tokens,), dtype=torch.long)
    global_embedding = torch.randn(GLOBAL_DIM)

    audio_length = case["target_audio_length"]
    if audio_length is None:
        audio_length = model._calculate_original_audio_length(num_tokens)
    stft_length = model._calculate_target_stft_length(audio_length)

    manifest: dict = {
        "name": case["name"],
        "num_tokens": num_tokens,
        "audio_length": audio_length,
        "stft_length": stft_length,
        "tensors": {},
    }

    captured: dict[str, torch.Tensor] = {}

    def capture_output(name: str):
        def hook(_module, _args, output):
            captured[name] = output.detach().clone()
        return hook

    def capture_input(name: str):
        def hook(_module, args):
            captured[name] = args[0].detach().clone()
        return hook

    handles = [
        model.wave_prenet.register_forward_hook(capture_output("after_prenet")),
        model.wave_conv_upsample.register_forward_hook(capture_output("after_conv_upsample")),
        # The functional `F.interpolate` between the conv upsample and the prior
        # net has no module to hang a hook on, so it is taken as the prior net's
        # input instead -- the same tensor, one step later.
        model.wave_prior_net.register_forward_pre_hook(capture_input("after_interpolate")),
        model.wave_prior_net.register_forward_hook(capture_output("after_prior_net")),
        model.wave_decoder.register_forward_hook(capture_output("after_decoder")),
        model.wave_post_net.register_forward_hook(capture_output("after_post_net")),
        # The 1922-wide projection, before it is split into log-magnitude and
        # phase, so the linear layer can be checked apart from the trigonometry.
        model.istft_head.out.register_forward_hook(capture_output("istft_linear")),
        # The complex spectrogram entering the inverse transform.
        model.istft_head.istft.register_forward_pre_hook(capture_input("spec")),
    ]

    try:
        with torch.inference_mode():
            content_embedding = model.decode_token_indices(tokens)
            waveform = model.decode(
                global_embedding,
                content_embedding=content_embedding,
                target_audio_length=audio_length,
            )
    finally:
        for handle in handles:
            handle.remove()

    write(directory, "tokens", tokens.to(torch.float32), manifest)
    write(directory, "global_embedding", global_embedding, manifest)
    write(directory, "content_embedding", content_embedding, manifest)
    for name in [
        "after_prenet",
        "after_conv_upsample",
        "after_interpolate",
        "after_prior_net",
        "after_decoder",
        "after_post_net",
        "istft_linear",
    ]:
        write(directory, name, captured[name], manifest)
    spec = captured["spec"]
    write(directory, "spec_real", spec.real, manifest)
    write(directory, "spec_imag", spec.imag, manifest)
    write(directory, "waveform", waveform, manifest)

    # Whether this case observes the interpolate at all. Recorded rather than
    # left to be worked out from the shapes, because "identity" is exactly the
    # condition under which a port that omits the stage still passes.
    before = captured["after_conv_upsample"]
    after = captured["after_interpolate"]
    manifest["interpolate_is_identity"] = bool(
        before.shape == after.shape and torch.equal(before, after)
    )

    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\n[{case['name']}] {num_tokens} tokens -> audio_length {audio_length}, "
          f"stft_length {stft_length}, waveform {tuple(waveform.shape)}"
          f"  interpolate_identity={manifest['interpolate_is_identity']}")
    for name, meta in manifest["tensors"].items():
        print(f"  {name:22s} {str(meta['shape']):22s} {meta['bytes']:>10,d} B")
    return manifest


def window_size(transformer) -> int | None:
    return getattr(transformer, "window_size", None)


def load_model() -> "MioCodecModel":
    """The model on its own, without `MioCodec.from_pretrained`.

    That wrapper expects a combined checkpoint carrying vocoder weights under a
    `vocoder.` prefix and raises without them. This rung has no vocoder — the
    iSTFT head *is* the vocoder — so it is loaded directly from the config and
    the state dict.

    `strict=False` because the WavLM SSL extractor's weights come from its own
    repository and are not in this file. Every missing and unexpected key is
    printed rather than swallowed: a decoder tensor quietly absent would leave a
    randomly initialised layer in the middle of the golden, which is noise that
    looks exactly like data.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    config_path = hf_hub_download(REPO_ID, "config.yaml")
    weights_path = hf_hub_download(REPO_ID, "model.safetensors")

    model = MioCodecModel.from_hparams(config_path)
    state = load_file(weights_path, device="cpu")
    result = model.load_state_dict(state, strict=False)

    decoder_prefixes = (
        "local_quantizer",
        "wave_prenet",
        "wave_conv_upsample",
        "wave_prior_net",
        "wave_decoder",
        "wave_post_net",
        "istft_head",
    )
    # Only *parameters* matter here. `istft_head.istft.window` is a registered
    # buffer built in `__init__` — a Hann window is a function of `n_fft`, not
    # something a checkpoint needs to carry — so its absence is correct and
    # counting it as missing would make this check cry wolf on every run.
    trained = set(dict(model.named_parameters()))
    missing = [
        k for k in result.missing_keys if k.startswith(decoder_prefixes) and k in trained
    ]
    if missing:
        raise SystemExit(f"decoder weights missing from the checkpoint: {missing[:10]}")
    buffers = [k for k in result.missing_keys if k not in trained]
    print(f"loaded {len(state)} tensors; {len(result.missing_keys)} missing "
          f"({len(buffers)} of them buffers, no decoder parameters), "
          f"{len(result.unexpected_keys)} unexpected")

    return model.to("cpu").eval()


def config_levels(model) -> list[int]:
    """FSQ levels, from the quantizer rather than from the yaml.

    `FiniteScalarQuantizer` wraps an `FSQ` that holds them; taking them from the
    loaded module rather than re-reading the yaml means the token range in the
    golden is the range the model actually has.
    """
    for holder in (model.local_quantizer, getattr(model.local_quantizer, "fsq", None)):
        value = getattr(holder, "levels", None)
        if value is not None:
            return [int(v) for v in (value.tolist() if torch.is_tensor(value) else value)]
    raise AttributeError("could not read FSQ levels from the quantizer")


def checkpoint_provenance(repo_id: str) -> dict:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo_id, "model.safetensors"))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    header_length = struct.unpack("<Q", path.read_bytes()[:8])[0]
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "header_bytes": header_length,
    }


if __name__ == "__main__":
    sys.exit(main())
