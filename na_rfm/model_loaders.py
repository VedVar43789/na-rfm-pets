"""Model loading helpers for OpenAI and HuggingFace DDPMs."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch


def _resolve_checkpoint(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute() and path.exists():
        return path

    try:
        from baselines.tfg.utils.env_utils import MODEL_PATH
        candidate = Path(MODEL_PATH) / path_str
        if candidate.exists():
            return candidate
    except Exception:
        pass

    return path


def load_openai_unet(
    model_name_or_path: str,
    image_size: int,
    device: str = "cuda",
) -> torch.nn.Module:
    """Load OpenAI DDPM UNet (TFG baseline-compatible)."""
    from baselines.tfg.diffusion.unet.openai import create_model

    if "cifar10" in model_name_or_path:
        num_res_blocks = 3
        num_channels = 128
        num_heads = 4
        num_heads_upsample = -1
        num_head_channels = -1
        attention_resolutions = "16,8"
        resblock_updown = False
        learn_sigma = False
    elif "imagenet" in model_name_or_path:
        num_res_blocks = 2
        num_channels = 256
        num_heads = 4
        num_heads_upsample = -1
        num_head_channels = 64
        attention_resolutions = "32,16,8"
        resblock_updown = True
        learn_sigma = True
    else:
        raise ValueError(f"Unsupported OpenAI model path: {model_name_or_path}")

    unet = create_model(
        image_size=image_size,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        num_head_channels=num_head_channels,
        attention_resolutions=attention_resolutions,
        resblock_updown=resblock_updown,
        learn_sigma=learn_sigma,
    )

    ckpt_path = _resolve_checkpoint(model_name_or_path)
    unet.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    unet.to(device).eval()
    return unet


def load_hf_unet_and_scheduler(
    model_id: str,
    device: str = "cuda",
) -> Tuple[torch.nn.Module, object]:
    """Load HuggingFace DDPM UNet and its scheduler."""
    from diffusers import DDIMPipeline

    pipe = DDIMPipeline.from_pretrained(model_id)
    pipe.unet.to(device).eval()
    return pipe.unet, pipe.scheduler
