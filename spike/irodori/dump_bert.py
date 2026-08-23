"""Dump ModernBERT-ja's weights and per-layer activations from Irodori's checkpoint.

    cd spike/irodori && IRODORI_REPO=... ../dacvae/.venv/bin/python dump_bert.py

Writes `golden/bert/` — one `.f32` per tensor plus `index.json`. Not in git.

## The weights come from Irodori, not from Hugging Face

`sbintuitions/modernbert-ja-310m` is the *architecture*. Irodori fine-tunes it
(`requires_grad_(True)` on every parameter) and ships the result inside its own
`model.safetensors` — 152 tensors under `pretrained_text_backbone.backbone.`,
which is exactly `5 + 24*6 + 3`: layer 0 has no `attn_norm` because
`ModernBertEncoderLayer` makes it `nn.Identity` there, the other 24 layers have
six tensors each, and embeddings contribute `tok_embeddings` + `norm` with
`final_norm` at the end.

Loading HF's weights instead would produce a model that runs, embeds Japanese
sensibly, and conditions the DiT on the wrong thing. The reference does not load
them either: `load_pretrained_backbone_weights=not use_pretrained_text_encoder`,
so at inference the 1.26 GB download supplies `config.json` and `tokenizer.json`
and nothing else.

## Why per-layer goldens

There are 25 layers and the pipeline golden records only the last one. A single
end-to-end comparison can say "wrong" but not "wrong from layer 7", and with two
attention patterns, two RoPE thetas and a GeGLU whose halves are easy to swap,
"wrong" on its own is most of a day.

## The input is reconstructed, and that reconstruction is checked

Irodori pads every text to `max_text_len = 256` — `[BOS, body..., PAD...]` with a
bool mask — so the backbone always runs 256 positions regardless of the
sentence. This script rebuilds that input with the reference's own
`PretrainedTextTokenizer.batch_encode` rather than re-running the 4 GB pipeline,
then compares against the pipeline run's own dump. If the reconstruction were
wrong the comparison would say so, which is why it is worth doing here rather
than trusting it.

The comparison is against **`final_norm`, not `backbone`**. `dump_golden.py`
hooks the module named `pretrained_text_backbone.backbone`, which is the inner
`ModernBertModel` — it returns `last_hidden_state` with every one of the 256
positions filled. The zeroing of padded positions happens one level up, in
`PretrainedTextBackbone.forward`, and is visible in the `text_encoder` golden:
7 non-zero rows there against 256 here.
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
OUT = HERE / "golden" / "bert"
REFERENCE = Path(os.environ.get("IRODORI_REPO", "")).expanduser()

PREFIX = "pretrained_text_backbone.backbone."
TOKENIZER_REPO = "sbintuitions/modernbert-ja-310m"
TOKENIZER_REVISION = "77675fc96a7e445e982e2ba90246b816efc74ec6"

# Which layers get an activation golden. The first is where a wrong embedding
# norm or a swapped GeGLU half shows up, the last is where everything that
# drifted has accumulated, and the ones between bracket the two attention
# patterns: layers 0, 3, 6 ... are full attention, the rest slide over 128.
KEEP_LAYERS = (0, 1, 2, 3, 12, 23, 24)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(name: str, tensor: torch.Tensor, into: dict, quiet: bool = False) -> None:
    array = tensor.detach().to(torch.float32).cpu().contiguous()
    path = OUT / f"{name}.f32"
    path.write_bytes(array.numpy().tobytes())
    into[name] = {
        "shape": list(array.shape),
        "dtype": "float32",
        "bytes": path.stat().st_size,
        "sha256": sha256_of(path),
    }
    if not quiet:
        print(f"  {name:52s} {tuple(array.shape)}")


def run_case(case, text, tokenizer, normalize_text, model, max_text_len, here) -> dict:
    """One input, its per-layer activations, and — for `short` — the pipeline check."""
    normalized = normalize_text(text).strip()
    input_ids, mask = tokenizer.batch_encode([normalized], max_length=max_text_len)
    real = int(mask.sum())
    print(f"\n{case}: {normalized[:28]!r} -> {real} real of {input_ids.shape[1]} positions")
    if case == "long" and real <= 65:
        raise SystemExit(
            f"the long case has only {real} tokens, so the 64-position sliding window "
            "never binds and every sliding layer behaves like a full one"
        )

    activations: dict = {}
    save(f"{case}/input_ids", input_ids.to(torch.float32), activations)
    save(f"{case}/mask", mask.to(torch.float32), activations)

    captured: dict[str, torch.Tensor] = {}
    handles = [
        model.embeddings.register_forward_hook(
            lambda _m, _i, out: captured.__setitem__("embeddings", out)
        )
    ]
    for index in KEEP_LAYERS:
        handles.append(
            model.layers[index].register_forward_hook(
                lambda _m, _i, out, i=index: captured.__setitem__(f"layers.{i}", out)
            )
        )
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=mask, return_dict=True)
    for handle in handles:
        handle.remove()

    for name in ["embeddings"] + [f"layers.{i}" for i in KEEP_LAYERS]:
        tensor = captured[name]
        if isinstance(tensor, tuple):
            tensor = tensor[0]
        save(f"{case}/{name}", tensor, activations)

    state_out = outputs.last_hidden_state
    save(f"{case}/final_norm", state_out, activations)
    # What `PretrainedTextBackbone.forward` returns: the final state zeroed
    # wherever the mask is false.
    masked = state_out * mask.unsqueeze(-1).to(dtype=state_out.dtype)
    save(f"{case}/backbone", masked, activations)

    result = {
        "text": text,
        "normalized": normalized,
        "real_positions": real,
        "activations": activations,
    }

    # Only the short case corresponds to a pipeline run, and only its
    # reconstruction can be checked against one.
    pipeline = here / "golden" / "pretrained_text_backbone.backbone.f32"
    if case == "short" and pipeline.exists():
        import numpy as np

        theirs = torch.from_numpy(
            np.frombuffer(pipeline.read_bytes(), dtype=np.float32).copy()
        ).view_as(state_out)
        # `state_out`, not `masked` — see the module note at the top.
        gap = (state_out.cpu() - theirs).abs().max().item()
        scale = theirs.abs().max().item()
        print(f"  against the pipeline's own dump: max |diff| {gap:.3e} of peak {scale:.3f}")
        if gap > 1e-3 * scale:
            raise SystemExit("reconstructed input does not reproduce the pipeline's activation")
        print("  the reconstructed input reproduces it — these goldens are the real path")
        result["matches_pipeline"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="……ん。来たんだ。")
    parser.add_argument(
        "--long-text",
        default=(
            "こんにちは、今日はいい天気ですね。散歩でもしませんか。"
            "えっと、それはちょっと、わたしには難しいかもしれません。"
            "全部追うの、無理だよ？　でもね、ひとつずつなら、たぶん、なんとかなる気がするの。"
            "朝から雨が降っていて、洗濯物も干せないし、出かける気にもなれなくて、"
            "結局ずっと家の中で本を読んでいました。"
            "そういう日があってもいいと思うんです。"
            "明日は晴れるといいですね。そうしたら、今度こそ散歩に行きましょう。"
        ),
        help="a text long enough that the 64-position sliding window actually binds",
    )
    parser.add_argument("--checkpoint", default="Aratako/Irodori-TTS-v4.1-Small")
    parser.add_argument("--max-text-len", type=int, default=256)
    args = parser.parse_args()

    if not REFERENCE.is_dir():
        raise SystemExit("Set IRODORI_REPO to a clone of Aratako/Irodori-TTS")
    sys.path.insert(0, str(REFERENCE))

    from huggingface_hub import hf_hub_download  # noqa: E402
    from safetensors.torch import load_file  # noqa: E402
    from transformers import AutoConfig  # noqa: E402
    from transformers.models.modernbert.modeling_modernbert import ModernBertModel  # noqa: E402

    from irodori_tts.text_normalization import normalize_text  # noqa: E402
    from irodori_tts.tokenizer import PretrainedTextTokenizer  # noqa: E402

    checkpoint = args.checkpoint
    if "/" in checkpoint and not Path(checkpoint).exists():
        checkpoint = hf_hub_download(checkpoint, "model.safetensors")

    state = load_file(checkpoint)
    backbone_state = {
        key[len(PREFIX) :]: value for key, value in state.items() if key.startswith(PREFIX)
    }
    if not backbone_state:
        raise SystemExit(f"no {PREFIX}* tensors in {checkpoint}")

    config = AutoConfig.from_pretrained(TOKENIZER_REPO, revision=TOKENIZER_REVISION)
    config._attn_implementation = "sdpa"
    model = ModernBertModel(config)
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    # `strict=False` only to tolerate buffers; a missing *parameter* means the
    # port would be reading weights the reference never used.
    real_missing = [k for k in missing if not k.endswith(".inv_freq")]
    if real_missing or unexpected:
        raise SystemExit(f"state_dict mismatch\n  missing {real_missing}\n  unexpected {unexpected}")
    model.eval()

    OUT.mkdir(parents=True, exist_ok=True)
    weights: dict = {}
    print(f"weights ({len(backbone_state)} tensors)")
    for key in sorted(backbone_state):
        save(key, backbone_state[key], weights, quiet=True)
    total = sum(v["bytes"] for v in weights.values())
    print(f"  {len(weights)} tensors, {total / 1e6:.1f} MB")

    tokenizer = PretrainedTextTokenizer.from_pretrained(
        TOKENIZER_REPO, add_bos=True, revision=TOKENIZER_REVISION
    )

    cases = {}
    for case, text in (("short", args.text), ("long", args.long_text)):
        cases[case] = run_case(
            case, text, tokenizer, normalize_text, model, args.max_text_len, HERE
        )

    payload = {
        "_rebuild": "cd spike/irodori && IRODORI_REPO=... ../dacvae/.venv/bin/python dump_bert.py",
        "checkpoint": args.checkpoint,
        "config_repo": TOKENIZER_REPO,
        "config_revision": TOKENIZER_REVISION,
        "max_text_len": args.max_text_len,
        "torch": torch.__version__,
        "config": {
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "intermediate_size": config.intermediate_size,
            "norm_eps": config.norm_eps,
            "layer_types": list(config.layer_types),
            "sliding_window": config.sliding_window,
            # Read from `rope_parameters` rather than the flat
            # `local_rope_theta` / `global_rope_theta` in `config.json`:
            # transformers resolves those into a per-layer-type mapping, and the
            # mapping is what the model actually indexes.
            "rope_theta": {
                kind: float(params["rope_theta"])
                for kind, params in config.rope_parameters.items()
            },
            "vocab_size": config.vocab_size,
        },
        "keep_layers": list(KEEP_LAYERS),
        "weights": weights,
        "cases": cases,
    }
    (OUT / "index.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    print(f"\nwrote {OUT}/")


if __name__ == "__main__":
    main()
