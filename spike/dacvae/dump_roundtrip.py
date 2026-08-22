"""Encode real speech with the reference, so the port can decode it and be listened to.

    cd spike/dacvae && .venv/bin/python dump_roundtrip.py --input path/to/48k.wav

Writes `roundtrip/` — the input as the codec saw it, the latent, and the
reference's own decode of that latent. `roundtrip.ts` then decodes the same
latent through the port and compares the two.

## Why this exists when `check.ts` already passes

`dump_golden.py`'s input is a synthetic signal, chosen so the stage-by-stage
comparison is reproducible and carries no licence. That is the right input for
checking arithmetic and the wrong one for checking reconstruction: its
round-trip correlation is 0.70 because a codec trained on speech does not
reconstruct broadband noise, and no amount of correct arithmetic would change
that.

So the two questions are separated. `check.ts` asks "does the port compute what
the reference computes", stage by stage, on a signal chosen for that. This asks
"does real speech survive the round trip", on speech.

## Only the decoder is ported

The encoder is not, and is not going to be — decoding never touches it, and
Irodori generates latents from its DiT rather than encoding at synthesis time.
It is used here purely to obtain a real latent to decode. That is also why the
reference's decode is dumped alongside: the port is compared against it, so a
difference is attributable to the decoder rather than to the encoder both sides
share.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torchaudio

warnings.filterwarnings("ignore")

from dacvae import DACVAE  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

REPO_ID = "Aratako/Semantic-DACVAE-Japanese-32dim"
HERE = Path(__file__).resolve().parent
OUT = HERE / "roundtrip"
NORMALIZE_DB = -16.0


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name: str, tensor: torch.Tensor, into: dict) -> None:
    array = tensor.detach().to(torch.float32).cpu().contiguous()
    path = OUT / f"{name}.f32"
    path.write_bytes(array.numpy().tobytes())
    into[name] = {"shape": list(array.shape), "dtype": "float32", "sha256": sha256_of(path)}
    print(f"  {name:16s} {tuple(array.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="a mono WAV; resampled if it is not 48 kHz")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--start", type=float, default=0.0)
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    model = DACVAE.load(str(hf_hub_download(REPO_ID, "weights.pth"))).eval()
    rate = int(model.sample_rate)

    # `torchaudio.load` now routes through TorchCodec, which is another
    # dependency for something the standard library already does. These are
    # 16-bit PCM WAVs; anything else is refused rather than guessed at.
    import wave

    with wave.open(args.input, "rb") as handle:
        if handle.getsampwidth() != 2:
            raise SystemExit(f"{args.input} is {handle.getsampwidth() * 8}-bit; only 16-bit PCM is read here")
        channels = handle.getnchannels()
        source_rate = handle.getframerate()
        frames = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    waveform = torch.from_numpy(frames.astype(np.float32) / 32768.0).reshape(-1, channels).T
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, rate)
    start = int(args.start * rate)
    waveform = waveform[:, start : start + int(args.seconds * rate)]
    audio = waveform.reshape(1, 1, -1)

    from audiotools import AudioSignal

    signal = AudioSignal(audio.squeeze(0), rate)
    signal.normalize(NORMALIZE_DB)
    signal.ensure_max_of_audio()
    audio = signal.audio_data.reshape(1, 1, -1)

    tensors: dict = {}
    save("input", audio, tensors)

    with torch.no_grad():
        encoded = model.encoder(model._pad(audio))
        mean, _scale = model.quantizer.in_proj(encoded).chunk(2, dim=1)
        save("latent", mean, tensors)

        # Irodori's bypass, so the reference decode is the one it actually uses.
        model.decoder.alpha = 0.0
        model.decoder.watermark = (
            lambda x, message=None, _d=model.decoder: _d.wm_model.encoder_block.forward_no_conv(x)
        )
        save("reference_decode", model.decoder(model.quantizer.out_proj(mean)), tensors)

    index = {
        "repo_id": REPO_ID,
        "source": {"file": str(Path(args.input).name), "start": args.start, "seconds": args.seconds},
        "config": {"sample_rate": rate, "hop_length": int(model.hop_length)},
        "normalize_db": NORMALIZE_DB,
        "tensors": tensors,
    }
    (OUT / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"\nwrote {OUT}/index.json — {audio.shape[-1] / rate:.2f} s at {rate} Hz")


if __name__ == "__main__":
    main()
