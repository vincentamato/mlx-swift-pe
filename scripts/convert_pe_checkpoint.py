#!/usr/bin/env python3
"""Convert PE Core checkpoints into MLXPE model directory layout.

Supported input forms:
1) Local model directory containing `config.json` and model weights.
2) Local checkpoint file (`.pt`, `.pth`, `.bin`, `.safetensors`).
3) Hugging Face repo id (for example `facebook/PE-Core-L14-336`).

The output directory contains:
- `config.json`
- `model.safetensors`
- `bpe_simple_vocab_16e6.txt`
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import torch
except Exception:  # pragma: no cover - runtime dependency
    torch = None

try:
    from safetensors.torch import load_file as load_safetensors
    from safetensors.torch import save_file as save_safetensors
except Exception:  # pragma: no cover - runtime dependency
    load_safetensors = None
    save_safetensors = None

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover - runtime dependency
    snapshot_download = None


PRESET_CONFIGS: Dict[str, Dict] = {
    "PE-Core-G14-448": {
        "model_name": "PE-Core-G14-448",
        "vision": {
            "image_size": 448,
            "patch_size": 14,
            "width": 1536,
            "layers": 50,
            "heads": 16,
            "mlp_ratio": 8960.0 / 1536.0,
            "output_dim": 1280,
            "use_class_token": False,
            "use_abs_posemb": True,
            "use_rope2d": True,
            "use_ln_pre": True,
            "use_ln_post": True,
            "pool_type": "attn",
            "attn_pooler_heads": 8,
        },
        "text": {
            "context_length": 72,
            "vocab_size": 49408,
            "width": 1280,
            "heads": 20,
            "layers": 24,
            "mlp_ratio": 4.0,
            "output_dim": 1280,
            "pool_type": "argmax",
            "pad_id": 0,
        },
        "initial_logit_scale": math.log(1.0 / 0.07),
    },
    "PE-Core-L14-336": {
        "model_name": "PE-Core-L14-336",
        "vision": {
            "image_size": 336,
            "patch_size": 14,
            "width": 1024,
            "layers": 24,
            "heads": 16,
            "mlp_ratio": 4.0,
            "output_dim": 1024,
            "use_class_token": True,
            "use_abs_posemb": True,
            "use_rope2d": True,
            "use_ln_pre": True,
            "use_ln_post": True,
            "pool_type": "attn",
            "attn_pooler_heads": 8,
        },
        "text": {
            "context_length": 32,
            "vocab_size": 49408,
            "width": 1024,
            "heads": 16,
            "layers": 24,
            "mlp_ratio": 4.0,
            "output_dim": 1024,
            "pool_type": "argmax",
            "pad_id": 0,
        },
        "initial_logit_scale": math.log(1.0 / 0.07),
    },
    "PE-Core-B16-224": {
        "model_name": "PE-Core-B16-224",
        "vision": {
            "image_size": 224,
            "patch_size": 16,
            "width": 768,
            "layers": 12,
            "heads": 12,
            "mlp_ratio": 4.0,
            "output_dim": 1024,
            "use_class_token": True,
            "use_abs_posemb": True,
            "use_rope2d": True,
            "use_ln_pre": True,
            "use_ln_post": True,
            "pool_type": "attn",
            "attn_pooler_heads": 8,
        },
        "text": {
            "context_length": 32,
            "vocab_size": 49408,
            "width": 1024,
            "heads": 16,
            "layers": 24,
            "mlp_ratio": 4.0,
            "output_dim": 1024,
            "pool_type": "argmax",
            "pad_id": 0,
        },
        "initial_logit_scale": math.log(1.0 / 0.07),
    },
    "PE-Core-S16-384": {
        "model_name": "PE-Core-S16-384",
        "vision": {
            "image_size": 384,
            "patch_size": 16,
            "width": 384,
            "layers": 12,
            "heads": 6,
            "mlp_ratio": 4.0,
            "output_dim": 512,
            "use_class_token": True,
            "use_abs_posemb": True,
            "use_rope2d": True,
            "use_ln_pre": True,
            "use_ln_post": True,
            "pool_type": "attn",
            "attn_pooler_heads": 8,
        },
        "text": {
            "context_length": 32,
            "vocab_size": 49408,
            "width": 512,
            "heads": 8,
            "layers": 12,
            "mlp_ratio": 4.0,
            "output_dim": 512,
            "pool_type": "argmax",
            "pad_id": 0,
        },
        "initial_logit_scale": math.log(1.0 / 0.07),
    },
    "PE-Core-T16-384": {
        "model_name": "PE-Core-T16-384",
        "vision": {
            "image_size": 384,
            "patch_size": 16,
            "width": 192,
            "layers": 12,
            "heads": 3,
            "mlp_ratio": 4.0,
            "output_dim": 512,
            "use_class_token": True,
            "use_abs_posemb": True,
            "use_rope2d": True,
            "use_ln_pre": True,
            "use_ln_post": True,
            "pool_type": "attn",
            "attn_pooler_heads": 8,
        },
        "text": {
            "context_length": 32,
            "vocab_size": 49408,
            "width": 512,
            "heads": 8,
            "layers": 12,
            "mlp_ratio": 4.0,
            "output_dim": 512,
            "pool_type": "argmax",
            "pad_id": 0,
        },
        "initial_logit_scale": math.log(1.0 / 0.07),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PE Core checkpoints to MLXPE format.")
    parser.add_argument("source", nargs="?", help="Model source path or Hugging Face repo id")
    parser.add_argument("destination", nargs="?", help="Output model directory")
    parser.add_argument("--input", dest="input_path", help="Source path or Hugging Face repo id")
    parser.add_argument("--output", dest="output_path", help="Output model directory")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Tuple[str, Path]:
    src = args.input_path or args.source
    dst = args.output_path or args.destination
    if not src or not dst:
        raise SystemExit("error: provide source and destination (positional or --input/--output)")
    return src, Path(dst)


def infer_variant(name_hint: str) -> str:
    candidates = [
        "PE-Core-G14-448",
        "PE-Core-L14-336",
        "PE-Core-B16-224",
        "PE-Core-S16-384",
        "PE-Core-T16-384",
    ]
    for variant in candidates:
        if variant.lower() in name_hint.lower():
            return variant
    if "g14" in name_hint.lower():
        return "PE-Core-G14-448"
    if "l14" in name_hint.lower():
        return "PE-Core-L14-336"
    if "b16" in name_hint.lower():
        return "PE-Core-B16-224"
    if "s16" in name_hint.lower():
        return "PE-Core-S16-384"
    if "t16" in name_hint.lower():
        return "PE-Core-T16-384"
    return "PE-Core-L14-336"


def extract_state_dict(obj):
    if isinstance(obj, dict):
        if all(hasattr(v, "shape") for v in obj.values()):
            return obj
        for key in ("state_dict", "weights", "model"):
            nested = obj.get(key)
            if isinstance(nested, dict) and all(hasattr(v, "shape") for v in nested.values()):
                return nested
    raise RuntimeError("Unable to find tensor state_dict in checkpoint")


def write_default_config(output_dir: Path, variant_hint: str) -> None:
    variant = infer_variant(variant_hint)
    config = PRESET_CONFIGS[variant]
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    print(f"[config] wrote default config for {variant}")


BPE_VOCAB_URL = (
    "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
)


def ensure_tokenizer(src_dir: Path, output_dir: Path) -> None:
    dst = output_dir / "bpe_simple_vocab_16e6.txt"
    if dst.exists():
        return

    # Try to copy from source directory first
    for name in ("bpe_simple_vocab_16e6.txt", "bpe_simple_vocab_16e6.txt.gz"):
        src = src_dir / name
        if src.exists():
            if name.endswith(".gz"):
                import gzip

                with gzip.open(src, "rb") as fin, dst.open("wb") as fout:
                    fout.write(fin.read())
            else:
                shutil.copy2(src, dst)
            print(f"[tokenizer] copied {src} -> {dst}")
            return

    # Download from OpenAI CLIP repo as fallback
    import gzip
    import urllib.request

    print(f"[tokenizer] downloading BPE vocab from {BPE_VOCAB_URL}...")
    data = urllib.request.urlopen(BPE_VOCAB_URL).read()
    with dst.open("wb") as f:
        f.write(gzip.decompress(data))
    print(f"[tokenizer] wrote {dst}")


def remap_state_dict(state_dict: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
    """Remap PyTorch CLIP state dict keys to match the Swift MLX model structure.

    Transformations applied:
    1. Text encoder keys (top-level, not visual.*) get a ``text.`` prefix.
    2. Fused ``attn.in_proj_weight`` / ``attn.in_proj_bias`` are split into
       separate ``q_proj``, ``k_proj``, ``v_proj`` tensors.
    3. Conv2d weights are transposed from PyTorch OIHW to MLX OHWI layout.
    """
    remapped: Dict[str, torch.Tensor] = {}

    # Keys that belong to the text encoder (not visual, not global logit_scale)
    visual_prefix = "visual."
    global_keys = {"logit_scale"}

    for key, tensor in state_dict.items():
        # Determine the output key prefix for text encoder keys
        if key.startswith(visual_prefix) or key in global_keys:
            out_key = key
        else:
            out_key = f"text.{key}"

        # Split fused in_proj into separate q/k/v projections
        if "attn.in_proj_weight" in out_key:
            prefix = out_key.replace("attn.in_proj_weight", "")
            dim = tensor.shape[0] // 3
            remapped[f"{prefix}attn.q_proj.weight"] = tensor[:dim].contiguous()
            remapped[f"{prefix}attn.k_proj.weight"] = tensor[dim : 2 * dim].contiguous()
            remapped[f"{prefix}attn.v_proj.weight"] = tensor[2 * dim :].contiguous()
            continue

        if "attn.in_proj_bias" in out_key:
            prefix = out_key.replace("attn.in_proj_bias", "")
            dim = tensor.shape[0] // 3
            remapped[f"{prefix}attn.q_proj.bias"] = tensor[:dim].contiguous()
            remapped[f"{prefix}attn.k_proj.bias"] = tensor[dim : 2 * dim].contiguous()
            remapped[f"{prefix}attn.v_proj.bias"] = tensor[2 * dim :].contiguous()
            continue

        # Transpose conv2d weights from PyTorch OIHW to MLX OHWI
        if out_key.endswith("conv1.weight") and tensor.ndim == 4:
            tensor = tensor.permute(0, 2, 3, 1).contiguous()

        remapped[out_key] = tensor

    return remapped


def ensure_safetensors_from_checkpoint(checkpoint_path: Path, output_path: Path) -> None:
    if torch is None or save_safetensors is None:
        if checkpoint_path.suffix == ".safetensors":
            shutil.copy2(checkpoint_path, output_path)
            print(f"[weights] copied {checkpoint_path} -> {output_path} (no remapping – torch unavailable)")
            return
        raise RuntimeError(
            "Converting .pt/.pth/.bin checkpoints requires torch and safetensors. "
            "Install with: pip install torch safetensors"
        )

    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file as _load_sf

        state_dict = _load_sf(str(checkpoint_path))
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = extract_state_dict(checkpoint)

    state_dict = {k: v.detach().cpu() for k, v in state_dict.items() if hasattr(v, "shape")}
    tensors = remap_state_dict(state_dict)
    save_safetensors(tensors, str(output_path))
    print(f"[weights] converted {checkpoint_path} -> {output_path} ({len(tensors)} tensors)")


def pick_weight_file(src_dir: Path) -> Optional[Path]:
    preferred = [
        src_dir / "model.safetensors",
        src_dir / "pytorch_model.bin",
        src_dir / "model.bin",
    ]
    for path in preferred:
        if path.exists():
            return path

    for pattern in ("*.safetensors", "*.pt", "*.pth", "*.bin"):
        found = sorted(src_dir.glob(pattern))
        if found:
            return found[0]
    return None


def prepare_source_directory(source: str) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    source_path = Path(source)
    if source_path.exists():
        if source_path.is_dir():
            return source_path, None
        return source_path.parent, None

    if snapshot_download is None:
        raise RuntimeError(
            "Hugging Face source requires huggingface_hub. Install with: pip install huggingface_hub"
        )

    temp_dir = tempfile.TemporaryDirectory(prefix="mlxpe_hf_")
    local_dir = Path(
        snapshot_download(repo_id=source, local_dir=temp_dir.name, local_dir_use_symlinks=False)
    )
    print(f"[download] snapshot downloaded to {local_dir}")
    return local_dir, temp_dir


def maybe_copy_existing_config(src_dir: Path, output_dir: Path) -> bool:
    config_path = src_dir / "config.json"
    if config_path.exists():
        shutil.copy2(config_path, output_dir / "config.json")
        print(f"[config] copied {config_path}")
        return True
    return False


def main() -> None:
    args = parse_args()
    source, output_dir = resolve_paths(args)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_weights = output_dir / "model.safetensors"

    source_path = Path(source)

    temp_ctx: Optional[tempfile.TemporaryDirectory] = None
    try:
        if source_path.exists() and source_path.is_file():
            ensure_safetensors_from_checkpoint(source_path, output_weights)
            write_default_config(output_dir, source_path.name)
            ensure_tokenizer(source_path.parent, output_dir)
            return

        src_dir, temp_ctx = prepare_source_directory(source)
        weight_file = pick_weight_file(src_dir)
        if weight_file is None:
            raise RuntimeError(f"No weight file found in {src_dir}")

        ensure_safetensors_from_checkpoint(weight_file, output_weights)

        copied_config = maybe_copy_existing_config(src_dir, output_dir)
        if not copied_config:
            write_default_config(output_dir, source)

        ensure_tokenizer(src_dir, output_dir)
        print(f"[done] wrote MLXPE model directory to {output_dir}")
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI entry
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
