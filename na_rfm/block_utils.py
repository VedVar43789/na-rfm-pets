"""Utilities for resolving hookable blocks across model types."""

from __future__ import annotations

from typing import Tuple


def resolve_block(model, block_name: str):
    """
    Resolve a block/module from an OpenAI or HuggingFace UNet.

    Supported block names:
      - OpenAI: input_blocks_9, input_blocks.9, output_blocks_3, middle_block
      - HF: down_blocks_4, down_blocks.4, up_blocks_1, mid_block
    """
    if block_name is None:
        raise ValueError("block_name is required")

    name = block_name.replace(".", "_")

    # Diffusers pipelines wrap unet
    unet = model.unet if hasattr(model, "unet") else model

    if name.startswith("input_blocks_"):
        idx = int(name.split("_")[-1])
        if not hasattr(unet, "input_blocks"):
            raise ValueError("Model has no input_blocks")
        return unet.input_blocks[idx]

    if name.startswith("output_blocks_"):
        idx = int(name.split("_")[-1])
        if not hasattr(unet, "output_blocks"):
            raise ValueError("Model has no output_blocks")
        return unet.output_blocks[idx]

    if name in ("middle_block", "mid_block"):
        if hasattr(unet, "middle_block"):
            return unet.middle_block
        if hasattr(unet, "mid_block"):
            return unet.mid_block
        raise ValueError("Model has no middle_block or mid_block")

    if name.startswith("down_blocks_"):
        idx = int(name.split("_")[-1])
        if not hasattr(unet, "down_blocks"):
            raise ValueError("Model has no down_blocks")
        return unet.down_blocks[idx]

    if name.startswith("up_blocks_"):
        idx = int(name.split("_")[-1])
        if not hasattr(unet, "up_blocks"):
            raise ValueError("Model has no up_blocks")
        return unet.up_blocks[idx]

    raise ValueError(f"Unknown block name: {block_name}")


def normalize_block_name(block_name: str) -> str:
    """Normalize block names to underscore style for file naming."""
    return block_name.replace(".", "_")
