"""Dump per-stage golden tensors from the reference MioCodec encoder's global path.

Issue #119: the TTS page's speaker identity is a 128-dim `global_embedding`
that today comes from a committed fixture. Zero-shot cloning needs the encoder
half — reference audio in, embedding out — ported to the browser, and a port
checked only at the embedding learns that something is wrong and nothing about
where. So: per-stage tensors, exactly as `dump_golden.py` does for the decoder.

Only the **global** path is captured. `codec.encode(w, return_content=False,
return_global=True)` runs WavLM's conv frontend, the feature projection, the
positional conv, and transformer layers 1..9 — but the global branch reads only
layers 1 and 2 (averaged), so a port needs just those two, and the goldens here
stop there. The content branch (layers 6+9, local encoder, FSQ) is out of
scope; its tokens come from the LM, not from reference audio.

The same two pins as the decoder golden, for the same reasons:

**CPU, not CUDA.** `_get_autocast_context` turns on bfloat16 autocast on CUDA;
a golden taken there is a blurred target.

**torch 2.10.** The version every convention in web-xpu-ops was verified
against. `torchaudio` 2.10 supplies both the WavLM implementation and the
24 kHz -> 16 kHz resampler, so its version is pinned by the same install.

Regenerate with:

    .venv/bin/python dump_encoder_golden.py

Output goes to `golden-encoder/`, ignored by the same `spike/**/golden*`
pattern as `golden/`; `index.json` records both checkpoints' sha256 — the
MioCodec safetensors **and** torchaudio's `wavlm_base_plus.pth`, because the
encoder's weights are split across the two files and either drifting silently
would poison every comparison.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
import torchaudio

from dump_golden import REPO_ID, SEED, checkpoint_provenance, load_model

OUT = Path(__file__).parent / "golden-encoder"

# The reference clips shipped with MioTTS-0.6B — real speech, 44.1 kHz, the
# exact inputs a user would hand the cloning page.
SAMPLES_REPO = "Aratako/MioTTS-0.6B"

# One synthetic case so the golden does not depend on any audio file at all.
# 2 seconds; scaled to a plausible speech amplitude — randn's unit variance is
# far outside [-1, 1] often enough to be unlike any waveform.
SYNTHETIC_SECONDS = 2.0
SYNTHETIC_SCALE = 0.1

CASES = [
    {"name": "jp_ref1", "wav": "samples/jp_ref1.wav"},
    {"name": "en_ref1", "wav": "samples/en_ref1.wav"},
    {"name": "synthetic", "wav": None},
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
    assert not model.training, "model must be in eval mode"

    # `load_model` checks the *decoder* prefixes; this golden dies instead on a
    # missing global-encoder tensor, for the same reason — a randomly
    # initialised layer mid-graph is noise that looks exactly like data.
    check_global_weights_loaded(model)

    extractor = model.ssl_feature_extractor
    index = {
        "repo_id": REPO_ID,
        "seed": SEED,
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "config": {
            "sample_rate": config.sample_rate,
            "ssl_sample_rate": int(extractor.ssl_sample_rate),
            "ssl_model": "wavlm_base_plus",
            "ssl_hop_size": extractor.hop_size,
            "ssl_conv_config": [list(t) for t in extractor.conv_config],
            "global_ssl_layers": list(model.global_ssl_layers),
            "global_dim": model.global_encoder.output_dim,
            "backbone_dim": model.global_encoder.backbone.dim,
            "backbone_layers": len(model.global_encoder.backbone.convnext),
        },
        "checkpoint": checkpoint_provenance(REPO_ID),
        "ssl_checkpoint": wavlm_provenance(),
        "cases": {},
    }

    for case in CASES:
        index["cases"][case["name"]] = dump_case(model, case)

    (OUT / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"\nwrote {OUT}")
    return 0


def load_case_waveform(model, case: dict) -> tuple[torch.Tensor, dict]:
    """The 24 kHz mono waveform `encode` will see, plus its provenance."""
    sample_rate = model.config.sample_rate

    if case["wav"] is None:
        # Seeded per case rather than once for the run, same formula as the
        # decoder golden, so adding a case cannot move another case's input.
        torch.manual_seed(SEED + sum(ord(c) for c in case["name"]))
        waveform = torch.randn(int(SYNTHETIC_SECONDS * sample_rate)) * SYNTHETIC_SCALE
        return waveform, {"source": "synthetic", "source_rate": sample_rate}

    from huggingface_hub import hf_hub_download

    # `soundfile` rather than `torchaudio.load`, which now routes through
    # TorchCodec and raises without it. Decoding a PCM wav needs neither.
    import soundfile

    path = hf_hub_download(SAMPLES_REPO, case["wav"])
    samples, source_rate = soundfile.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples).T.mean(dim=0)  # to mono
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform.unsqueeze(0), source_rate, sample_rate
        ).squeeze(0)
    return waveform, {
        "source": f"{SAMPLES_REPO}/{case['wav']}",
        "source_rate": int(source_rate),
        # The source file's own digest. `hf_hub_download` resolves through a
        # mutable `refs/main`, so the clip a consumer feeds a port can differ
        # from the one this golden was built from with nothing to say so —
        # and the failure would look like a kernel or resampler bug. Recorded
        # here so `check-tts.mjs` can hash the wav it feeds the page and name
        # "the wav drifted" instead.
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }


def dump_case(model, case: dict) -> dict:
    """One encode, every global-path stage of it, into `golden-encoder/<name>/`."""
    directory = OUT / case["name"]
    directory.mkdir(parents=True, exist_ok=True)

    waveform, provenance = load_case_waveform(model, case)
    audio_length = waveform.numel()
    padding = model._calculate_waveform_padding(audio_length)

    manifest: dict = {
        "name": case["name"],
        **provenance,
        "sample_rate": model.config.sample_rate,
        "audio_length": audio_length,
        # `encode` pads this many zero samples on *both* sides at 24 kHz before
        # resampling to 16 kHz. Recorded so the port reproduces it rather than
        # rediscovering the formula from `_calculate_waveform_padding`.
        "waveform_padding": padding,
        "tensors": {},
    }

    captured: dict[str, torch.Tensor] = {}

    def capture_output(name: str, pick=lambda output: output):
        def hook(_module, _args, output):
            captured[name] = pick(output).detach().clone()
        return hook

    def capture_input(name: str):
        def hook(_module, args):
            captured[name] = args[0].detach().clone()
        return hook

    wavlm = model.ssl_feature_extractor.model
    pooling = model.global_encoder.pooling
    handles = [
        # The 16 kHz waveform entering WavLM: padding at 24 kHz has already
        # been applied, so this one tensor checks pad + resample together.
        model.ssl_feature_extractor.resampler.register_forward_hook(
            capture_output("after_resample")
        ),
        # WavLM's conv frontend returns (features, lengths).
        wavlm.feature_extractor.register_forward_hook(
            capture_output("after_feature_extractor", pick=lambda out: out[0])
        ),
        wavlm.encoder.feature_projection.register_forward_hook(
            capture_output("after_feature_projection")
        ),
        # The additive positional embedding on its own; the transformer adds it
        # to its input, so `layer` inputs = projection + this.
        wavlm.encoder.transformer.pos_conv_embed.register_forward_hook(
            capture_output("after_pos_conv")
        ),
        # Each encoder layer returns (x, position_bias). Only layers 1 and 2
        # feed the global branch (`global_ssl_layers = [1, 2]`).
        wavlm.encoder.transformer.layers[0].register_forward_hook(
            capture_output("ssl_layer1", pick=lambda out: out[0])
        ),
        wavlm.encoder.transformer.layers[1].register_forward_hook(
            capture_output("ssl_layer2", pick=lambda out: out[0])
        ),
        # What the global encoder actually receives: mean of layers 1 and 2,
        # taken at the module boundary rather than recomputed here.
        model.global_encoder.register_forward_pre_hook(
            capture_input("global_input")
        ),
        # The ConvNeXt backbone, block by block (B, 384, T).
        *[
            block.register_forward_hook(capture_output(f"convnext_block{i + 1}"))
            for i, block in enumerate(model.global_encoder.backbone.convnext)
        ],
        # Backbone output after its final LayerNorm (B, T, 384) — the
        # pre-pooling tensor.
        model.global_encoder.backbone.register_forward_hook(
            capture_output("after_backbone")
        ),
        # The attention weights of the stats pooling (B, 384, T): softmax over
        # time, the stage a port gets wrong by normalising the wrong axis.
        pooling.attn.register_forward_hook(capture_output("attn_weights")),
        # cat(mean, std) entering the output projection (B, 768).
        pooling.proj.register_forward_pre_hook(capture_input("pooled_stats")),
    ]

    try:
        features = model.encode(waveform, return_content=False, return_global=True)
    finally:
        for handle in handles:
            handle.remove()

    assert features.content_embedding is None
    global_embedding = features.global_embedding

    # A second, hook-free call: the hooks must observe the computation, not
    # perturb it, and the reference must be deterministic — both are claims,
    # so both are checked.
    features_again = model.encode(waveform, return_content=False, return_global=True)
    assert torch.equal(global_embedding, features_again.global_embedding), (
        f"[{case['name']}] encode() is not deterministic or hooks perturbed it"
    )

    write(directory, "waveform_24k", waveform, manifest)
    for name in [
        "after_resample",
        "after_feature_extractor",
        "after_feature_projection",
        "after_pos_conv",
        "ssl_layer1",
        "ssl_layer2",
        "global_input",
        *[f"convnext_block{i + 1}" for i in range(len(model.global_encoder.backbone.convnext))],
        "after_backbone",
        "attn_weights",
        "pooled_stats",
    ]:
        write(directory, name, captured[name], manifest)
    write(directory, "global_embedding", global_embedding, manifest)

    # The captured input to the global encoder must be the mean of the two
    # captured layers — asserted, because "layers [1, 2], averaged" is exactly
    # the kind of claim a config change would silently invalidate.
    expected = (captured["ssl_layer1"] + captured["ssl_layer2"]) / 2
    assert torch.equal(captured["global_input"], expected), (
        f"[{case['name']}] global input is not mean(layer1, layer2)"
    )

    print(f"\n[{case['name']}] {audio_length} samples @ {manifest['sample_rate']} Hz "
          f"(pad {padding} each side) -> global_embedding {tuple(global_embedding.shape)}")
    for name, meta in manifest["tensors"].items():
        print(f"  {name:26s} {str(meta['shape']):22s} {meta['bytes']:>12,d} B")

    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def check_global_weights_loaded(model) -> None:
    """Die if any global-encoder parameter still carries its random init.

    `load_model` loads with `strict=False` and verifies only the decoder
    prefixes. The global encoder's weights are in the same safetensors file;
    re-loading it here and diffing the key sets is cheaper than trusting.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    weights_path = hf_hub_download(REPO_ID, "model.safetensors")
    with safe_open(weights_path, framework="pt") as f:
        checkpoint_keys = set(f.keys())

    wanted = [k for k in dict(model.named_parameters()) if k.startswith("global_encoder.")]
    absent = [k for k in wanted if k not in checkpoint_keys]
    if absent:
        raise SystemExit(f"global encoder weights missing from the checkpoint: {absent[:10]}")
    print(f"global encoder: {len(wanted)} parameters, all present in the checkpoint")


def wavlm_provenance() -> dict:
    """sha256 of torchaudio's WavLM checkpoint — the encoder's *other* weights file.

    `bundle.get_model()` reads it from torch hub's cache; the safetensors
    provenance alone would let the SSL half drift without the manifest noticing.
    """
    import torchaudio.pipelines as pipelines

    bundle = pipelines.WAVLM_BASE_PLUS
    path = Path(torch.hub.get_dir()) / "checkpoints" / bundle._path
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


if __name__ == "__main__":
    sys.exit(main())
