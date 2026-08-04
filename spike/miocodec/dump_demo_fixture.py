"""Encode a real speech clip to MioCodec tokens, for the browser demo to decode.

Unlike the goldens, this output **is** small enough to commit — a couple of
kilobytes of token indices and a 128-dim speaker vector — and the demo needs it
at load time, so it goes to `examples/` rather than to the ignored `golden/`.

Why real speech rather than the random tokens the goldens use: the goldens exist
to compare two implementations, where being in distribution buys nothing. A demo
has to produce something a listener recognises as a voice.

    .venv/bin/python dump_demo_fixture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torchaudio

from dump_golden import GLOBAL_DIM, REPO_ID, load_model

HERE = Path(__file__).parent
SOURCE = HERE.parent.parent / "examples" / "browser" / "public" / "models" / "chatterbox-multilingual" / "default_voice.wav"
OUT = HERE.parent.parent / "examples" / "mio-tts-fixture.json"
# Long enough to be recognisably speech, short enough that the reference decode
# below stays quick and the fixture stays a couple of kilobytes.
MAX_SECONDS = 6.0


def main() -> int:
    model = load_model()
    config = model.config
    sample_rate = config.sample_rate

    if not SOURCE.exists():
        raise SystemExit(f"no source audio at {SOURCE}")

    # `soundfile` rather than `torchaudio.load`, which now routes through
    # TorchCodec and raises without it. Decoding a PCM wav needs neither.
    import soundfile

    samples, source_rate = soundfile.read(str(SOURCE), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples).T.mean(dim=0, keepdim=True)  # to mono
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    waveform = waveform[:, : int(MAX_SECONDS * sample_rate)]

    with torch.inference_mode():
        features = model.encode(waveform.squeeze(0))
        tokens = features.content_token_indices.reshape(-1)
        global_embedding = features.global_embedding.reshape(-1)
        reference = model.decode(global_embedding, content_token_indices=tokens)

    assert global_embedding.numel() == GLOBAL_DIM, global_embedding.shape

    audio_length = model._calculate_original_audio_length(tokens.numel())
    fixture = {
        "repo_id": REPO_ID,
        "source": str(SOURCE.relative_to(HERE.parent.parent)),
        "sample_rate": sample_rate,
        "n_fft": config.n_fft,
        "hop_length": config.hop_length,
        "stft_length": model._calculate_target_stft_length(audio_length),
        # Integers, so the JSON stays small and exact. The decoder reads them
        # back as floats -- an FSQ index is a small integer and f32 carries it
        # without rounding well past this codebook's 12800.
        "tokens": [int(v) for v in tokens.tolist()],
        # Full precision: this is a speaker identity, and a rounded one is a
        # different voice.
        "global_embedding": [float(v) for v in global_embedding.tolist()],
    }
    OUT.write_text(json.dumps(fixture) + "\n")

    reference_path = HERE / "golden" / "demo_reference.f32"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_bytes(reference.to(torch.float32).cpu().numpy().tobytes())

    print(f"source        {SOURCE.name}, {source_rate} Hz -> {sample_rate} Hz, "
          f"{waveform.shape[1] / sample_rate:.2f} s")
    print(f"tokens        {tokens.numel()} @ 25 Hz, range {tokens.min().item()}..{tokens.max().item()}")
    print(f"stft_length   {fixture['stft_length']}")
    print(f"reference     {reference.numel()} samples "
          f"({reference.numel() / sample_rate:.2f} s), peak {reference.abs().max():.4f}")
    print(f"fixture       {OUT} ({OUT.stat().st_size:,} B)")
    print(f"reference pcm {reference_path} (not in git)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
