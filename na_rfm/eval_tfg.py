"""TFG-compatible evaluation wrapper for image guidance tasks."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Sequence

import torch
from PIL import Image


def _ensure_tfg_on_path():
    tfg_path = Path(__file__).resolve().parents[1] / "baselines" / "tfg"
    if str(tfg_path) not in sys.path:
        sys.path.insert(0, str(tfg_path))


def tensor_to_pil(samples: torch.Tensor) -> List[Image.Image]:
    """Convert [-1, 1] tensor batch to list of PIL images."""
    samples = samples.clamp(-1, 1)
    samples = (samples + 1.0) / 2.0
    samples = (samples * 255).round().byte()
    pil_images = []
    for img in samples:
        pil = Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
        pil_images.append(pil)
    return pil_images


def build_tfg_args(
    dataset: str,
    tasks: Sequence[str],
    guide_networks: Sequence[str],
    targets: Sequence[int | str],
    device: str,
    eval_batch_size: int,
):
    """Create a minimal args object compatible with TFG ImageEvaluator."""
    _ensure_tfg_on_path()
    from utils.configs import Arguments

    task_str = "+".join(tasks)
    guide_network_str = "+".join(guide_networks)
    target_str = "+".join(str(t) for t in targets)

    args = Arguments(
        data_type="image",
        dataset=dataset,
        task=task_str,
        guide_network=guide_network_str,
        target=target_str,
        eval_batch_size=eval_batch_size,
        device=device,
    )

    args.tasks = list(tasks)
    args.guide_networks = list(guide_networks)
    args.targets = list(targets)
    return args


def evaluate_samples(
    samples: torch.Tensor | List[Image.Image],
    dataset: str,
    tasks: Sequence[str],
    guide_networks: Sequence[str],
    targets: Sequence[int | str],
    device: str = "cuda",
    eval_batch_size: int = 32,
):
    """Run TFG ImageEvaluator on generated samples."""
    _ensure_tfg_on_path()
    from evaluations.image import ImageEvaluator

    if torch.is_tensor(samples):
        images = tensor_to_pil(samples)
    else:
        images = samples

    args = build_tfg_args(
        dataset=dataset,
        tasks=tasks,
        guide_networks=guide_networks,
        targets=targets,
        device=device,
        eval_batch_size=eval_batch_size,
    )

    evaluator = ImageEvaluator(args)
    return evaluator.evaluate(images)
