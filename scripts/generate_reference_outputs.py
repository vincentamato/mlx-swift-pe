#!/usr/bin/env python3
"""Generate deterministic PE Core reference outputs from the PyTorch implementation.

This script reads the Python reference implementation from `.reference/perception_models`
and writes layer-wise `.npy` outputs for Swift parity testing.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PE Core reference outputs.")
    parser.add_argument("--model-name", default="PE-Core-L14-336", help="PE Core config name")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional local .pt checkpoint path. If omitted, downloads from Hugging Face.",
    )
    parser.add_argument(
        "--output",
        default="Tests/MLXPETests/Resources/ReferenceOutputs",
        help="Output directory for .npy files",
    )
    parser.add_argument(
        "--image-url",
        default="https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
        help="Image URL used for deterministic test input",
    )
    parser.add_argument(
        "--prompt",
        nargs="+",
        default=["a photo of a cat", "a photo of a dog", "a diagram"],
        help="Text prompts",
    )
    return parser.parse_args()


def setup_imports(repo_root: Path) -> None:
    reference_root = repo_root / ".reference" / "perception_models"
    if not reference_root.exists():
        raise FileNotFoundError(
            f"Missing reference source at {reference_root}. Move the Python reference repo there first."
        )
    sys.path.insert(0, str(reference_root))


def load_image(url: str) -> Image.Image:
    import requests

    headers = {"User-Agent": "mlxpe-reference/1.0"}
    response = requests.get(url, timeout=60, headers=headers)
    response.raise_for_status()
    return Image.open(Path("/dev/null") if False else __import__("io").BytesIO(response.content)).convert("RGB")


def save_array(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().numpy()
    np.save(path, array)


def main() -> None:
    args = parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    repo_root = Path(__file__).resolve().parents[1]
    setup_imports(repo_root)

    from core.vision_encoder.config import PE_TEXT_CONFIG, PE_VISION_CONFIG
    from core.vision_encoder.pe import CLIP
    from core.vision_encoder.transforms import get_image_transform, get_text_tokenizer

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = CLIP.from_config(args.model_name, pretrained=True, checkpoint_path=args.checkpoint)
    model.eval()

    image = load_image(args.image_url)
    image_transform = get_image_transform(model.image_size, center_crop=False)
    image_tensor = image_transform(image).unsqueeze(0)

    tokenizer = get_text_tokenizer(model.context_length)
    text_tokens = tokenizer(args.prompt)

    with torch.no_grad():
        visual = model.visual

        batch, _, height, width = image_tensor.shape
        grid_h = height // visual.patch_size
        grid_w = width // visual.patch_size

        patch = visual.conv1(image_tensor)
        patch = patch.permute(0, 2, 3, 1).reshape(batch, -1, visual.width)

        hidden = patch
        if visual.use_cls_token:
            cls = visual.class_embedding.view(1, 1, -1).expand(batch, -1, -1)
            hidden = torch.cat([cls, hidden], dim=1)

        if visual.use_abs_posemb:
            hidden = hidden + visual._sample_abs_posemb(grid_h, grid_w)

        if visual.use_rope2d:
            visual.rope.update_grid(hidden.device, grid_h, grid_w)

        hidden = visual.ln_pre(hidden)

        vision_block_0 = None
        for index, block in enumerate(visual.transformer.resblocks):
            hidden = block(hidden)
            if index == 0:
                vision_block_0 = hidden.clone()

        hidden = visual.ln_post(hidden)
        vision_final = hidden
        vision_pooled = visual._pool(hidden)
        if visual.proj_dim is not None:
            vision_projected = vision_pooled @ visual.proj
        else:
            vision_projected = vision_pooled
        image_features = F.normalize(vision_projected, dim=-1)

        seq_len = text_tokens.shape[1]
        text_embed = model.token_embedding(text_tokens)
        text_embed = text_embed + model.positional_embedding[:seq_len]

        attn_mask = model.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask[:seq_len, :seq_len]

        text_hidden = text_embed
        text_block_0 = None
        for index, block in enumerate(model.transformer.resblocks):
            text_hidden = block(text_hidden, attn_mask=attn_mask)
            if index == 0:
                text_block_0 = text_hidden.clone()

        text_hidden = model.ln_final(text_hidden)
        text_pooled, _ = model.text_global_pool(text_hidden, text_tokens, pool_type=model.pool_type)
        if isinstance(model.text_projection, torch.nn.Linear):
            text_projected = model.text_projection(text_pooled)
        else:
            text_projected = text_pooled @ model.text_projection

        text_features = F.normalize(text_projected, dim=-1)
        logit_scale = model.logit_scale.exp()
        similarity = logit_scale * (image_features @ text_features.t())

    save_array(output_dir / "test_image.npy", image_tensor)
    save_array(output_dir / "test_tokens.npy", text_tokens)
    save_array(output_dir / "vision_patch_embed_output.npy", patch)
    if vision_block_0 is not None:
        save_array(output_dir / "vision_block_0_output.npy", vision_block_0)
    save_array(output_dir / "vision_block_final_output.npy", vision_final)
    save_array(output_dir / "vision_pooled_output.npy", vision_pooled)
    save_array(output_dir / "vision_projected_output.npy", vision_projected)
    save_array(output_dir / "image_features.npy", image_features)
    save_array(output_dir / "text_embed_output.npy", text_embed)
    if text_block_0 is not None:
        save_array(output_dir / "text_block_0_output.npy", text_block_0)
    save_array(output_dir / "text_features.npy", text_features)
    save_array(output_dir / "logit_scale.npy", logit_scale)
    save_array(output_dir / "similarity.npy", similarity)

    config = {
        "model_name": args.model_name,
        "vision": asdict(PE_VISION_CONFIG[args.model_name]),
        "text": asdict(PE_TEXT_CONFIG[args.model_name]),
        "prompts": args.prompt,
        "image_url": args.image_url,
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"Saved reference outputs to {output_dir}")


if __name__ == "__main__":
    main()
