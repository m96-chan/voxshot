#!/usr/bin/env python3
"""Convert MioTTS-0.6B (Qwen3, bf16 safetensors) into the per-row int8 format
the GPU engine consumes, written to spike/miotts/q8/ (gitignored, rebuildable).

Format follows web-xpu-ops' upstream q8 conventions exactly:

- Every Linear (q/k/v/o, gate/up/down per layer) and the embedding table
  (tied — it is also the lm_head matvec matrix; the checkpoint has no lm_head
  tensor) is quantized per-row: symmetric int8 [-127, 127], one f32 scale per
  output row, `scale = absmax/127` (1.0 for an all-zero row). The rounding is
  bit-for-bit `ops/quantize/reference.ts#quantize` (JS `Math.round`, ties
  toward +Infinity) via `quant_common.quantize_per_row`, imported from
  web-xpu-ops rather than re-derived — that module's docstring explains why
  neither `np.round` nor `floor(x + 0.5)` matches.
- q_proj / k_proj ROWS are relabeled from HF's rotate-half channel order into
  ops/rope's adjacent-pair order BEFORE quantization (`permute_rope_channels`
  below, transcribed from `llm/tools/gen_fixture.py` /
  `llm/weights.ts#permuteRopeChannels`): within each head, HF channel j moves
  to 2j for j < headDim/2 and to 2(j - headDim/2) + 1 otherwise. The per-head
  q_norm / k_norm gammas get the SAME relabeling; both the permuted (qNorm /
  kNorm) and the checkpoint-order (qNormRaw / kNormRaw) gammas are written to
  norms.bin so a test can verify the permutation without re-reading the
  checkpoint.
- Norm vectors stay f32, unquantized.

bf16 is read via torch (`safe_open(framework="pt")` + `.to(torch.float32)`),
not numpy — numpy has no native bf16 and would need the u16 << 16 bit trick;
torch is already installed for dump_golden.py and widens exactly.

Config numbers and the checkpoint sha256 are taken from golden/index.json (the
same source the tests compare against) and the checkpoint file is re-hashed
here and required to match, so the artifacts can never silently come from a
different snapshot than the goldens.

Usage:  cd spike/miotts && python3 convert_weights.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
# Same sibling layout package.json's `file:../../../web-xpu-ops` relies on.
WEB_XPU_TOOLS = HERE.parents[2] / "web-xpu-ops" / "llm" / "tools"
if not (WEB_XPU_TOOLS / "quant_common.py").is_file():
    sys.exit(f"quant_common.py not found at {WEB_XPU_TOOLS}; is web-xpu-ops checked out beside zerovox?")
sys.path.insert(0, str(WEB_XPU_TOOLS))
from quant_common import quantize_per_row  # noqa: E402

REPO_ID = "Aratako/MioTTS-0.6B"
OUT_DIR = HERE / "q8"
GOLDEN_INDEX = HERE / "golden" / "index.json"
# Rows per quantization chunk: only embed_tokens (164480 rows) exceeds this;
# quantize_per_row is per-row independent, so chunking changes nothing but the
# peak of its f64 temporaries (~7x the tensor otherwise, >5 GB for the table).
CHUNK_ROWS = 16384


def permute_rope_channels(w: np.ndarray, heads: int, head_dim: int) -> np.ndarray:
    """`[heads*headDim, inFeatures]` -> same shape, rows relabeled per head.

    New row 2i is old row i; new row 2i+1 is old row i + headDim/2, for i in
    [0, headDim/2). Transcribed from `llm/tools/gen_fixture.py` (itself a numpy
    restatement of `llm/weights.ts#permuteRopeChannels`, whose module doc holds
    the derivation); weights-q8.test.ts re-verifies the mapping bit-for-bit
    against an independent restatement.
    """
    in_features = w.shape[1]
    half = head_dim // 2
    grouped = w.reshape(heads, head_dim, in_features)
    out = np.empty_like(grouped)
    out[:, 0::2, :] = grouped[:, :half, :]
    out[:, 1::2, :] = grouped[:, half:, :]
    return out.reshape(heads * head_dim, in_features)


def permute_gamma(g: np.ndarray, head_dim: int) -> np.ndarray:
    """The same relabeling applied to a `[headDim]` per-head norm gamma."""
    return permute_rope_channels(g.reshape(-1, 1), 1, head_dim).reshape(-1)


class BinWriter:
    """Appends to a file while hashing, so each bin's sha256 costs no re-read."""

    def __init__(self, path: Path):
        self.path = path
        self.f = path.open("wb")
        self.h = hashlib.sha256()
        self.offset = 0

    def write(self, data: bytes) -> None:
        self.f.write(data)
        self.h.update(data)
        self.offset += len(data)

    def close(self) -> dict:
        self.f.close()
        return {"name": self.path.name, "bytes": self.offset, "sha256": self.h.hexdigest()}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def locate_checkpoint() -> Path:
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(REPO_ID, "model.safetensors", local_files_only=True))
    except Exception:
        return Path(hf_hub_download(REPO_ID, "model.safetensors"))


def main() -> None:
    started = time.monotonic()
    if not GOLDEN_INDEX.is_file():
        sys.exit("golden/index.json is missing; run `python3 dump_golden.py` first (it pins config + checkpoint sha)")
    golden = json.loads(GOLDEN_INDEX.read_text())
    gc = golden["config"]
    config = {
        "numLayers": gc["num_layers"],
        "hiddenSize": gc["hidden"],
        "numHeads": gc["heads"],
        "numKvHeads": gc["kv_heads"],
        "headDim": gc["head_dim"],
        "ffnHidden": gc["ffn"],
        "vocabSize": gc["vocab"],
        "ropeTheta": gc["rope_theta"],
        "rmsNormEps": gc["rms_eps"],
        "tieEmbeddings": gc["tie_word_embeddings"],
        "speechTokenBase": gc["speech_token_base"],
        "eosIds": gc["eos_ids"],
    }
    head_dim = config["headDim"]

    st_path = locate_checkpoint()
    print(f"[convert] checkpoint {st_path}", flush=True)
    actual_sha = sha256_file(st_path)
    wanted_sha = golden["checkpoint"]["sha256"]
    if actual_sha != wanted_sha:
        sys.exit(f"checkpoint sha256 {actual_sha} != golden's {wanted_sha}; wrong snapshot")

    OUT_DIR.mkdir(exist_ok=True)
    codes_w = BinWriter(OUT_DIR / "weights.codes.bin")
    scales_w = BinWriter(OUT_DIR / "weights.scales.bin")
    norms_w = BinWriter(OUT_DIR / "weights.norms.bin")
    tensors: list[dict] = []
    # Per tensor-family worst quantization error, relative to the tensor's own
    # absmax: max_elements |dequant - src| / max_elements |src|.
    family_err: dict[str, float] = {}

    with safe_open(st_path, framework="pt") as f:

        def read_f32(key: str) -> np.ndarray:
            t = f.get_tensor(key)
            return t.to(torch.float32).numpy()

        def write_quant(name: str, family: str, key: str, permute_heads: int | None = None) -> None:
            w = read_f32(key)
            if permute_heads is not None:
                w = permute_rope_channels(w, permute_heads, head_dim)
            rows, cols = w.shape
            entry = {
                "name": name,
                "kind": "quant",
                "rows": rows,
                "cols": cols,
                "codesOffset": codes_w.offset,
                "codesBytes": rows * cols,
                "scaleOffset": scales_w.offset,
                "scaleBytes": rows * 4,
            }
            absmax = float(np.max(np.abs(w))) or 1.0
            worst = 0.0
            for at in range(0, rows, CHUNK_ROWS):
                chunk = w[at : at + CHUNK_ROWS]
                codes, scale = quantize_per_row(chunk)
                err = np.max(np.abs(codes.astype(np.float64) * scale.astype(np.float64)[:, None] - chunk))
                worst = max(worst, float(err))
                codes_w.write(np.ascontiguousarray(codes, dtype="<i1").tobytes())
                scales_w.write(np.ascontiguousarray(scale, dtype="<f4").tobytes())
            family_err[family] = max(family_err.get(family, 0.0), worst / absmax)
            tensors.append(entry)

        def write_norm(name: str, vec: np.ndarray) -> None:
            tensors.append({
                "name": name,
                "kind": "norm",
                "cols": int(vec.size),
                "offset": norms_w.offset,
                "bytes": int(vec.size) * 4,
            })
            norms_w.write(np.ascontiguousarray(vec, dtype="<f4").tobytes())

        write_quant("embedTokens", "embedTokens", "model.embed_tokens.weight")
        for i in range(config["numLayers"]):
            a = f"model.layers.{i}.self_attn"
            m = f"model.layers.{i}.mlp"
            write_norm(f"layers.{i}.attnNorm", read_f32(f"model.layers.{i}.input_layernorm.weight"))
            write_quant(f"layers.{i}.wq", "wq", f"{a}.q_proj.weight", permute_heads=config["numHeads"])
            write_quant(f"layers.{i}.wk", "wk", f"{a}.k_proj.weight", permute_heads=config["numKvHeads"])
            write_quant(f"layers.{i}.wv", "wv", f"{a}.v_proj.weight")
            write_quant(f"layers.{i}.wo", "wo", f"{a}.o_proj.weight")
            write_norm(f"layers.{i}.ffnNorm", read_f32(f"model.layers.{i}.post_attention_layernorm.weight"))
            write_quant(f"layers.{i}.wGate", "wGate", f"{m}.gate_proj.weight")
            write_quant(f"layers.{i}.wUp", "wUp", f"{m}.up_proj.weight")
            write_quant(f"layers.{i}.wDown", "wDown", f"{m}.down_proj.weight")
            q_gamma = read_f32(f"{a}.q_norm.weight")
            k_gamma = read_f32(f"{a}.k_norm.weight")
            write_norm(f"layers.{i}.qNorm", permute_gamma(q_gamma, head_dim))
            write_norm(f"layers.{i}.kNorm", permute_gamma(k_gamma, head_dim))
            write_norm(f"layers.{i}.qNormRaw", q_gamma)
            write_norm(f"layers.{i}.kNormRaw", k_gamma)
            print(f"[convert] layer {i + 1}/{config['numLayers']} done", flush=True)
        write_norm("finalNorm", read_f32("model.norm.weight"))

    files = {"codes": codes_w.close(), "scales": scales_w.close(), "norms": norms_w.close()}

    manifest = {
        "generatedBy": "spike/miotts/convert_weights.py",
        "source": {"path": str(st_path), "bytes": st_path.stat().st_size, "sha256": actual_sha},
        "config": config,
        "ropePermuted": True,
        "ropePermutedNote": (
            "layers.*.wq / layers.*.wk rows and the layers.*.qNorm / layers.*.kNorm gammas are "
            "relabeled from HF rotate-half channel order to ops/rope adjacent-pair order "
            "(within each head: HF channel j -> 2j for j < headDim/2, else 2(j-headDim/2)+1; "
            "llm/weights.ts#permuteRopeChannels), applied BEFORE quantization. "
            "layers.*.qNormRaw / layers.*.kNormRaw keep the checkpoint's original gamma order."
        ),
        "quantStats": {
            "description": "per tensor-family worst |dequant - src| / absmax(src)",
            "maxRelError": {k: family_err[k] for k in sorted(family_err)},
        },
        "files": files,
        "tensors": tensors,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    elapsed = time.monotonic() - started
    total = sum(m["bytes"] for m in files.values())
    print(f"[convert] wrote {OUT_DIR / 'manifest.json'}")
    for label, meta in files.items():
        print(f"[convert] {label}: {meta['bytes']:,} bytes")
    print(f"[convert] total: {total:,} bytes ({total / 1024 / 1024:.1f} MiB)")
    print("[convert] max |dequant - src| / absmax per family:")
    for fam in sorted(family_err):
        print(f"[convert]   {fam}: {family_err[fam]:.6f}")
    print(f"[convert] done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
