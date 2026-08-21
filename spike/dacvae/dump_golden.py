"""Dump stage-by-stage goldens from Meta's DACVAE reference, for the port to check against.

    cd spike/dacvae && .venv/bin/python dump_golden.py

Writes `golden/index.json` plus one `.f32` per tensor. The goldens are NOT in
git — they are ~10 MB and regenerable — so every reader has to fail loudly when
they are absent rather than letting a suite go green over nothing.

## What is dumped, and why those stages

The decoder is four upsampling blocks over `decoder_rates = [12, 10, 8, 2]`,
and each block is `Snake -> ConvTranspose1d -> ResUnit(d=1) -> ResUnit(d=3) ->
ResUnit(d=9)`. A whole-decoder comparison would say "wrong" without saying
where, and at 1920x upsampling an error in the first block is unrecognisable by
the last. So every block boundary is a checkpoint, and so is the input
projection that feeds them.

## Irodori's watermark bypass is applied, and it has to be

The checkpoint ships with `decoder.alpha = 0.25`, so the unmodified
`Decoder.forward` embeds a watermark — and `watermark()` draws a **random
message** when none is given. A golden taken from the unpatched model is
therefore both wrong for this port and not reproducible.

`irodori_tts/codec.py` replaces the whole method with

    decoder.watermark = lambda x, _m=None: decoder.wm_model.encoder_block.forward_no_conv(x)

which is not "skip the watermark" — it is "keep the output tail, drop the
message". `WatermarkEncoderBlock.pre` is `[Snake1d(96), NormConv1d(96->1),
Tanh, NormConv1d]` and `forward_no_conv` swaps the last conv for `Identity`,
so the tail that actually produces the waveform is **Snake -> Conv -> Tanh**,
and it lives inside `wm_model`. Everything else there — the message processor,
the LSTM, the decoder block, the up/down samplers — is unreachable once this
override is in place.

Within each `DecoderBlock` the picture is different and simpler: `forward()`
walks only the even chunks (block indices 0,1,4,5,8,9), and the odd ones are
reached solely by `upsample_group()`/`downsample_group()`, which only the real
`watermark()` calls. With this override they are dead.

The same override is applied here, verbatim. Reproducing the reference means
reproducing the reference Irodori actually calls.

## Determinism

`deterministic_encode` in Irodori's `codec.py` takes the VAE's mean rather than
sampling it, so encode is a function of its input. This mirrors that: the
latent is `quantizer.in_proj(z).chunk(2)[0]`, never a draw. Two runs of this
script produce identical bytes, which the sha256 in the index pins.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import torch
import torchaudio

warnings.filterwarnings("ignore")

from dacvae import DACVAE  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

REPO_ID = "Aratako/Semantic-DACVAE-Japanese-32dim"
SEED = 20260821
HERE = Path(__file__).resolve().parent
OUT = HERE / "golden"

# One second, and a bit. Long enough that every block has several frames to
# work with, short enough that the goldens stay small and a mismatch is
# readable when printed.
SECONDS = 1.2

# Irodori's own preprocessing: `codec.py` normalises loudness to -16 dB before
# encoding, so a golden taken without it would be measuring a different input
# than the one the model is used with.
NORMALIZE_DB = -16.0


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
    print(f"  {name:28s} {tuple(array.shape)}")


def load_reference_audio(sample_rate: int) -> torch.Tensor:
    """A deterministic pseudo-random waveform, not a recording.

    A real clip would be better for judging *quality*, but this dump is for
    judging *arithmetic*, and for that a synthetic signal is strictly better:
    it is reproducible anywhere, it carries no licence, and it excites every
    frequency instead of whatever a particular voice happens to contain. The
    round-trip listening check is a separate exercise on real audio.
    """
    generator = torch.Generator().manual_seed(SEED)
    length = int(SECONDS * sample_rate)
    noise = torch.randn(length, generator=generator) * 0.1
    # Plus a few tones, so the spectrum is not flat and a channel mix-up shows.
    time = torch.arange(length, dtype=torch.float32) / sample_rate
    tones = sum(0.15 * torch.sin(2 * torch.pi * f * time) for f in (110.0, 440.0, 1730.0))
    return (noise + tones).clamp(-1.0, 1.0).reshape(1, 1, -1)


def main() -> None:
    torch.manual_seed(SEED)
    OUT.mkdir(exist_ok=True)

    checkpoint = Path(hf_hub_download(REPO_ID, "weights.pth"))
    model = DACVAE.load(str(checkpoint)).eval()
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)["metadata"]["kwargs"]

    tensors: dict = {}
    audio = load_reference_audio(model.sample_rate)

    # -- loudness normalisation, mirroring irodori_tts/codec.py
    from audiotools import AudioSignal

    signal = AudioSignal(audio.squeeze(0), model.sample_rate)
    signal.normalize(NORMALIZE_DB)
    signal.ensure_max_of_audio()
    audio = signal.audio_data.reshape(1, 1, -1)
    save("input_48k", audio, tensors)

    with torch.no_grad():
        padded = model._pad(audio)
        save("after_pad", padded, tensors)

        encoded = model.encoder(padded)
        save("encoder_out", encoded, tensors)

        mean, _scale = model.quantizer.in_proj(encoded).chunk(2, dim=1)
        save("latent_mean", mean, tensors)

        projected = model.quantizer.out_proj(mean)
        save("after_out_proj", projected, tensors)

        # -- the decoder, one stage at a time.
        #
        # Hooks rather than a re-implementation: the point of a golden is to
        # record what the reference does, and anything re-derived here could
        # be wrong in the same way the port is about to be.
        stages: dict[str, torch.Tensor] = {}
        handles = []
        for index, module in enumerate(model.decoder.model):
            handles.append(
                module.register_forward_hook(
                    lambda _m, _inp, out, i=index: stages.__setitem__(f"decoder_model_{i}", out)
                )
            )
        # Irodori's bypass, copied from irodori_tts/codec.py rather than
        # re-derived: alpha to zero, and `watermark` replaced by the tail.
        model.decoder.alpha = 0.0
        model.decoder.watermark = (
            lambda x, message=None, _d=model.decoder: _d.wm_model.encoder_block.forward_no_conv(x)
        )

        waveform = model.decoder(projected)
        for handle in handles:
            handle.remove()

        # The tail, one stage at a time: Snake -> Conv -> Tanh -> (Identity).
        tail = model.decoder.wm_model.encoder_block.pre
        h = stages["decoder_model_4"]
        for i, layer in enumerate(tail[:-1]):
            h = layer(h)
            save(f"tail_{i}_{type(layer).__name__.lower()}", h, tensors)

        for name in sorted(stages, key=lambda k: int(k.rsplit("_", 1)[1])):
            save(name, stages[name], tensors)
        save("waveform", waveform, tensors)

    index = {
        "repo_id": REPO_ID,
        "seed": SEED,
        "torch": torch.__version__,
        "normalize_db": NORMALIZE_DB,
        "config": {
            "sample_rate": int(model.sample_rate),
            "hop_length": int(model.hop_length),
            "latent_dim": int(metadata["codebook_dim"]),
            "encoder_dim": int(metadata["encoder_dim"]),
            "encoder_rates": list(metadata["encoder_rates"]),
            "decoder_dim": int(metadata["decoder_dim"]),
            "decoder_rates": list(metadata["decoder_rates"]),
        },
        "checkpoint": {
            "file": "weights.pth",
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_of(checkpoint),
        },
        "tensors": tensors,
    }
    (OUT / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"\nwrote {OUT}/index.json ({len(tensors)} tensors)")
    print(f"  {model.sample_rate} Hz, hop {model.hop_length}, "
          f"{model.sample_rate / model.hop_length:.0f} latent frames/s")


if __name__ == "__main__":
    main()
