"""Dump what the DACVAE encoder was actually given, and what it produced.

    cd spike/dacvae && IRODORI_REPO=... .venv/bin/python dump_encode.py \\
        --input ../irodori/samples/reference-voice.wav

Writes `golden/encode/` — the waveform, the per-block activations, the latent.
Not in git.

## The input is recorded, not reconstructed

`encode_waveform` does three things before the encoder sees a sample: it
downmixes to mono, resamples to 48 kHz, and normalises to **-16 dB LUFS**. The
last is a K-weighted, gated loudness measurement — a filter-design problem
rather than a tensor one, and not ported.

Recording the waveform *after* that step is what lets the encoder be checked on
its own. The alternative is to check it on a clip that skipped normalisation,
which compares two different signals and calls the difference a porting error.
This is the same reason `spike/irodori`'s `dump_golden.py` grew pre-hooks.

## Deterministic

`deterministic_encode=True` is what Irodori asks for and what a reference clip
should get: the latent is the VAE's **mean**, not a sample from it. Two runs on
the same clip produce identical bytes, which the sha256 in the index pins.
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
OUT = HERE / "golden" / "encode"
REFERENCE = Path(os.environ.get("IRODORI_REPO", "")).expanduser()

# Which submodules are worth an activation golden: the first convolution, each
# downsampling block, and the two tail steps. Anything deeper is inside a
# residual unit the port reproduces whole.
INTERESTING = ("block.0", "block.1", "block.2", "block.3", "block.4", "block.5", "block.6")


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
    print(f"  {name:26s} {tuple(array.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="a reference clip, any rate")
    parser.add_argument("--codec", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    args = parser.parse_args()

    if not REFERENCE.is_dir():
        raise SystemExit("Set IRODORI_REPO to a clone of Aratako/Irodori-TTS")
    sys.path.insert(0, str(REFERENCE))

    import torchaudio  # noqa: E402

    from irodori_tts.codec import DACVAECodec  # noqa: E402

    codec = DACVAECodec.load(
        repo_id=args.codec,
        device="cpu",
        deterministic_encode=True,
        deterministic_decode=True,
    )
    model = codec.model

    waveform, rate = torchaudio.load(args.input)
    print(f"input {args.input}: {waveform.shape[1]} samples at {rate} Hz")

    # The measured loudness, recorded separately from the normalised waveform.
    #
    # For this clip the two say different things. The gain takes the peak past
    # full scale, so `ensure_max_of_audio` scales it back to exactly 1.0 and the
    # output becomes `raw / peak` — **independent of what the meter measured**.
    # A port with a broken gate, a broken block overlap or a missing -0.691
    # offset reproduces that waveform exactly. Only this number can tell them
    # apart.
    from audiotools import AudioSignal  # noqa: E402

    measured = float(AudioSignal(waveform[:1].unsqueeze(0), int(rate)).loudness().item())
    print(f"  integrated loudness {measured:.4f} LUFS")

    OUT.mkdir(parents=True, exist_ok=True)
    tensors: dict = {}

    # The waveform as `encode_waveform` prepares it — mono, 48 kHz, normalised.
    # A pre-hook on the encoder is the only place that is observable, since
    # `encode_waveform` does not return it.
    captured: dict[str, torch.Tensor] = {}

    def pre(_module, inputs):
        captured["encoder_input"] = inputs[0]

    handles = [model.encoder.register_forward_pre_hook(pre)]
    for name, module in model.encoder.block.named_children():
        if f"block.{name}" in INTERESTING:
            handles.append(
                module.register_forward_hook(
                    lambda _m, _i, out, n=name: captured.__setitem__(f"block.{n}", out)
                )
            )
    handles.append(
        model.quantizer.in_proj.register_forward_hook(
            lambda _m, _i, out: captured.__setitem__("quantizer.in_proj", out)
        )
    )

    with torch.inference_mode():
        latent = codec.encode_waveform(waveform, rate)  # (B, T, D)
    for handle in handles:
        handle.remove()

    # The pre-hook fires on `self.encoder(self._pad(audio_data))`, so what it
    # caught is *after* `_pad` — 950400 samples for a 950323-sample clip. Both
    # are saved: trimming back to the original length gives the encoder's real
    # input, and keeping the padded one lets the port's reflect padding be
    # checked rather than assumed. A golden of only the padded signal would hide
    # a padding bug instead of catching it.
    padded = captured["encoder_input"]
    save("waveform_padded", padded, tensors)
    save("waveform", padded[..., : int(waveform.shape[1])], tensors)
    for name in INTERESTING:
        if name in captured:
            save(name, captured[name], tensors)
    save("quantizer_in_proj", captured["quantizer.in_proj"], tensors)
    save("latent", latent, tensors)

    payload = {
        "_rebuild": (
            "cd spike/dacvae && IRODORI_REPO=... .venv/bin/python dump_encode.py "
            "--input ../irodori/samples/reference-voice.wav"
        ),
        "codec": args.codec,
        "input": args.input,
        "input_samples": int(waveform.shape[1]),
        "input_rate": int(rate),
        "normalize_db": codec.normalize_db,
        "measured_lufs": measured,
        "deterministic_encode": True,
        "torch": torch.__version__,
        "tensors": tensors,
    }
    (OUT / "index.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT}/ — normalised to {codec.normalize_db} dB LUFS before the encoder")


if __name__ == "__main__":
    main()
