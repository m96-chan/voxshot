"""Dump per-stage golden tensors from the reference MioTTS-0.6B LM.

The goldens themselves are **not in git** — raw floats say nothing in a diff and
they are reproducible from here. This script is in git; its output is ignored.
Regenerate with:

    python3 dump_golden.py

Why per stage, and not just the generated ids: a port checked only at the
sampler learns that something is wrong and nothing about where. Every error in
every layer arrives at the logits together, and the attention internals are the
ones most likely to be subtly wrong (RoPE half-rotation pairing, the per-head
q/k RMSNorm that Qwen3 applies after projection, GQA head repetition) in ways
that still produce plausible-looking token ids. Layer 0's attention is therefore
opened up further than the rest: q_proj output, q after q_norm, q after RoPE,
and the o_proj output, so a RoPE or qk-norm bug is localized to the exact op.

Two things are pinned deliberately.

**CPU float32, eager attention.** The checkpoint is bf16; loading with
`dtype=torch.float32` and running on CPU keeps the golden sharp — a golden taken
under bf16 or SDPA fusion would carry rounding into every comparison and the
port would be measured against a blurred target. `attn_implementation="eager"`
keeps the attention path on the plain matmul/softmax code that the hooks and the
monkeypatch below were verified against (transformers 5.3.0's
`modeling_qwen3.py` was read, not guessed).

**Greedy, cross-checked.** `generate(do_sample=False)` is the convenient oracle
but it reads `generation_config.json` (which says `do_sample: true`,
`max_new_tokens: 2048`) and its quirks can drift silently. So a manual
argmax-over-logits loop runs alongside it; the first 8 ids must agree, and if
they ever diverge the manual loop is the one written as the golden and the
disagreement is recorded in the manifest.

Capture mechanics, module hook vs functional capture:

- module forward hooks: `embed_tokens`, every decoder layer (their forward
  returns a plain tensor in transformers 5.3.0), `model.norm`,
  `layers[0].self_attn.q_proj`, `layers[0].self_attn.q_norm`,
  `layers[0].self_attn.o_proj`.
- functional capture: q **after RoPE** has no module boundary —
  `apply_rotary_pos_emb` is a module-level function called from
  `Qwen3Attention.forward` via a runtime global lookup — so it is monkeypatched
  for the duration of one prompt forward and the layer-0 call's return value is
  kept. (The `@use_kernelized_func` decorator only sets an unused `rotary_fn`
  attribute; the generated forward calls the global, so the patch does
  intercept. Asserted, not trusted.)
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import torch

REPO_ID = "Aratako/MioTTS-0.6B"
OUT = Path(__file__).parent / "golden"

# Greedy at f32 on CPU is deterministic — no randomness anywhere in this dump.
# Kept for any future sampled case so it lands next to the tensors it seeded.
SEED = 20260821

EOS_IDS = (151645, 151643)  # <|im_end|>, <|endoftext|>
SPEECH_TOKEN_BASE = 151669  # <|s_0|>
SPEECH_TOKEN_LAST = 164468  # <|s_12799|>
GREEDY_STEPS = 64
GREEDY_FULL_CAP = 512

CASES = [
    {"name": "ja", "content": "こんにちは、今日はいい天気ですね", "full": True},
    {"name": "en", "content": "Hello! How are you today?", "full": False},
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
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    model = AutoModelForCausalLM.from_pretrained(
        REPO_ID, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    assert not model.training, "model must be in eval mode"

    config = model.config
    # The facts every downstream comparison leans on, asserted rather than
    # assumed — a checkpoint swap that changed any of these must fail here,
    # not as a mysterious shape error in a test.
    assert config.num_hidden_layers == 28
    assert config.hidden_size == 1024
    assert config.vocab_size == 164480
    assert config.tie_word_embeddings
    assert tokenizer.convert_tokens_to_ids("<|s_0|>") == SPEECH_TOKEN_BASE
    assert tokenizer.convert_tokens_to_ids("<|s_12799|>") == SPEECH_TOKEN_LAST
    # Tied embeddings: no separate lm_head tensor in the checkpoint. If the
    # tie ever silently broke, logits goldens would encode a random matrix.
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()

    rope_theta = config.rope_parameters["rope_theta"]

    index = {
        "repo_id": REPO_ID,
        "seed": SEED,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "checkpoint": checkpoint_provenance(REPO_ID),
        "config": {
            "num_layers": config.num_hidden_layers,
            "hidden": config.hidden_size,
            "heads": config.num_attention_heads,
            "kv_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "ffn": config.intermediate_size,
            "vocab": config.vocab_size,
            "rope_theta": rope_theta,
            "rms_eps": config.rms_norm_eps,
            "tie_word_embeddings": config.tie_word_embeddings,
            "eos_ids": list(EOS_IDS),
            "speech_token_base": SPEECH_TOKEN_BASE,
        },
        "cases": {},
    }

    for case in CASES:
        index["cases"][case["name"]] = dump_case(model, tokenizer, case)

    (OUT / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"\nwrote {OUT}  ({time.monotonic() - started:.1f}s)")
    return 0


def prompt_ids_for(tokenizer, content: str) -> torch.Tensor:
    """The chat-templated prompt, pinned against the raw template string.

    The jinja template is trivial ChatML with no default system message, so the
    exact rendered string is written out here and tokenized independently; if
    the template file ever changes shape, the two id sequences disagree and the
    golden refuses to regenerate instead of silently moving.
    """
    messages = [{"role": "user", "content": content}]
    # transformers 5.3 returns a BatchEncoding here, not a bare id list.
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)["input_ids"]
    raw = f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
    raw_ids = tokenizer(raw).input_ids
    assert ids == raw_ids, (
        f"chat template drifted from the pinned ChatML string:\n"
        f"  template: {ids}\n  raw:      {raw_ids}"
    )
    return torch.tensor([ids], dtype=torch.long)


def greedy_manual(model, prompt: torch.Tensor, steps: int) -> list[int]:
    """Argmax continuation with an explicit KV cache, no `generate()` between
    the model and the ids. Does not stop at eos — the caller decides."""
    ids: list[int] = []
    with torch.inference_mode():
        out = model(input_ids=prompt, use_cache=True)
        past = out.past_key_values
        next_id = int(out.logits[0, -1].argmax())
        ids.append(next_id)
        for _ in range(steps - 1):
            out = model(
                input_ids=torch.tensor([[next_id]], dtype=torch.long),
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            next_id = int(out.logits[0, -1].argmax())
            ids.append(next_id)
    return ids


def greedy_generate(model, prompt: torch.Tensor, max_new_tokens: int) -> list[int]:
    with torch.inference_mode():
        out = model.generate(
            prompt,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=list(EOS_IDS),
            pad_token_id=151643,
        )
    return out[0, prompt.shape[1]:].tolist()


def dump_case(model, tokenizer, case: dict) -> dict:
    """One prompt forward with every stage hooked, then the greedy oracles,
    into `golden/<name>/`."""
    started = time.monotonic()
    name = case["name"]
    directory = OUT / name
    directory.mkdir(parents=True, exist_ok=True)

    prompt = prompt_ids_for(tokenizer, case["content"])
    num_tokens = prompt.shape[1]

    manifest: dict = {
        "name": name,
        "content": case["content"],
        "prompt_tokens": num_tokens,
        "tensors": {},
    }

    captured: dict[str, torch.Tensor] = {}

    def capture_output(key: str):
        def hook(_module, _args, output):
            # Decoder layers return a plain tensor in transformers 5.3.0, but
            # older versions returned a tuple; take [0] if one ever appears.
            tensor = output[0] if isinstance(output, tuple) else output
            captured[key] = tensor.detach().clone()
        return hook

    layer0 = model.model.layers[0].self_attn
    handles = [
        model.model.embed_tokens.register_forward_hook(capture_output("embedding")),
        model.model.norm.register_forward_hook(capture_output("final_norm")),
        # Layer 0 attention internals, module hooks. q_norm's output is the
        # [1, T, 16, 128] per-head view *before* the transpose to head-major,
        # so flattening its last two dims restores the q_proj [T, 2048] layout.
        layer0.q_proj.register_forward_hook(capture_output("l0_q_proj")),
        layer0.q_norm.register_forward_hook(capture_output("l0_q_normed")),
        layer0.o_proj.register_forward_hook(capture_output("l0_attn_out")),
    ]
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(capture_output(f"layer{i:02d}")))

    # q after RoPE: functional capture. `apply_rotary_pos_emb` has no module to
    # hang a hook on; `Qwen3Attention.forward` reaches it through a runtime
    # global lookup in modeling_qwen3, so replacing that global intercepts every
    # layer's call and the first one per forward is layer 0's. The tensor there
    # is head-major [1, 16, T, 128]; it is transposed back so every l0_q_*
    # golden shares the [T, 2048] layout.
    from transformers.models.qwen3 import modeling_qwen3

    original_rope = modeling_qwen3.apply_rotary_pos_emb

    def capturing_rope(q, k, cos, sin, unsqueeze_dim=1):
        q_embed, k_embed = original_rope(q, k, cos, sin, unsqueeze_dim)
        if "l0_q_roped" not in captured:
            captured["l0_q_roped"] = (
                q_embed.detach().clone().transpose(1, 2).reshape(q.shape[2], -1)
            )
        return q_embed, k_embed

    modeling_qwen3.apply_rotary_pos_emb = capturing_rope
    try:
        with torch.inference_mode():
            outputs = model(input_ids=prompt, use_cache=False)
    finally:
        modeling_qwen3.apply_rotary_pos_emb = original_rope
        for handle in handles:
            handle.remove()

    assert "l0_q_roped" in captured, (
        "the apply_rotary_pos_emb monkeypatch never fired — the attention "
        "forward no longer calls the module global; re-read modeling_qwen3.py"
    )

    write(directory, "prompt_ids", prompt[0].to(torch.float32), manifest)
    write(directory, "embedding", captured["embedding"][0], manifest)
    for i in range(len(model.model.layers)):
        write(directory, f"layer{i:02d}", captured[f"layer{i:02d}"][0], manifest)
    write(directory, "l0_q_proj", captured["l0_q_proj"][0], manifest)
    write(directory, "l0_q_normed", captured["l0_q_normed"][0].reshape(num_tokens, -1), manifest)
    write(directory, "l0_q_roped", captured["l0_q_roped"], manifest)
    write(directory, "l0_attn_out", captured["l0_attn_out"][0], manifest)
    write(directory, "final_norm", captured["final_norm"][0], manifest)
    write(directory, "logits_last", outputs.logits[0, -1], manifest)

    # Greedy oracle, twice: generate() for convenience, a manual argmax loop
    # as the tie-breaker. See the module docstring for why both.
    manual_ids = greedy_manual(model, prompt, GREEDY_STEPS)
    generate_ids = greedy_generate(model, prompt, GREEDY_STEPS)
    first8_agree = manual_ids[:8] == generate_ids[:8]
    full_agree = generate_ids == manual_ids[: len(generate_ids)]
    manifest["greedy_generate_matches_manual"] = full_agree
    manifest["greedy_generate_len"] = len(generate_ids)
    if not first8_agree:
        # generate() config quirks drifted the oracle; the manual loop is the
        # golden and the drift is on record for whoever regenerates next.
        manifest["greedy_generate_ids"] = generate_ids
        print(f"  WARNING [{name}] generate() != manual argmax; wrote the manual loop:")
        print(f"    manual:   {manual_ids[:8]}")
        print(f"    generate: {generate_ids[:8]}")
    greedy64 = manual_ids if not (first8_agree and len(generate_ids) == GREEDY_STEPS and full_agree) else generate_ids
    write(directory, "greedy64", torch.tensor(greedy64, dtype=torch.float32), manifest)

    if case["full"]:
        full_ids = greedy_generate(model, prompt, GREEDY_FULL_CAP)
        eos_reached = len(full_ids) > 0 and full_ids[-1] in EOS_IDS
        speech = [t for t in full_ids if SPEECH_TOKEN_BASE <= t <= SPEECH_TOKEN_LAST]
        ratio = len(speech) / len(full_ids)
        assert ratio > 0.9, (
            f"only {len(speech)}/{len(full_ids)} generated ids are speech tokens "
            f"({ratio:.1%}); the model is not doing TTS on this prompt"
        )
        manifest["greedy_full_len"] = len(full_ids)
        manifest["speech_token_count"] = len(speech)
        manifest["eos_reached"] = eos_reached
        write(directory, "greedy_full", torch.tensor(full_ids, dtype=torch.float32), manifest)

    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\n[{name}] {num_tokens} prompt tokens, "
          f"generate==manual over first {len(generate_ids)}: {full_agree}"
          + (f", greedy_full {manifest['greedy_full_len']} ids "
             f"({manifest['speech_token_count']} speech, eos={manifest['eos_reached']})"
             if case["full"] else "")
          + f"  ({time.monotonic() - started:.1f}s)")
    for tensor_name, meta in manifest["tensors"].items():
        print(f"  {tensor_name:14s} {str(meta['shape']):14s} {meta['bytes']:>10,d} B")
    return manifest


def checkpoint_provenance(repo_id: str) -> dict:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo_id, "model.safetensors"))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    with path.open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "header_bytes": header_length,
    }


if __name__ == "__main__":
    sys.exit(main())
