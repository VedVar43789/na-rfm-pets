"""Activation collection utilities for RFM training."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch

from .block_utils import resolve_block


class ActivationCollector:
    def __init__(self, block_name: str, flatten: bool = True, dtype: torch.dtype = torch.float16):
        self.block_name = block_name
        self.flatten = flatten
        self.dtype = dtype
        self._handle = None
        self._captured = None

    def _hook_fn(self, module, inputs, output):
        if torch.is_tensor(output):
            act = output
        elif isinstance(output, tuple) and torch.is_tensor(output[0]):
            act = output[0]
        else:
            return output

        if self.flatten:
            act = act.reshape(act.shape[0], -1)
        self._captured = act.detach().to(self.dtype)
        return output

    def register(self, model):
        block = resolve_block(model, self.block_name)
        self._handle = block.register_forward_hook(self._hook_fn)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def capture(self) -> torch.Tensor:
        if self._captured is None:
            raise RuntimeError("No activation captured for this batch")
        return self._captured


@torch.no_grad()
def collect_activations(
    model,
    dataloader: Iterable,
    block_name: str,
    alpha_prod_t: float | torch.Tensor,
    timestep: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect activations at a fixed timestep from a dataloader.
    """
    unet = model.unet if hasattr(model, "unet") else model
    unet.eval()

    collector = ActivationCollector(block_name=block_name, flatten=True, dtype=dtype)
    collector.register(unet)

    alpha_prod_t = torch.tensor(alpha_prod_t, device=device, dtype=torch.float32)
    sqrt_alpha = torch.sqrt(alpha_prod_t)
    sqrt_one_minus = torch.sqrt(1.0 - alpha_prod_t)

    all_acts = []
    all_labels = []

    for batch in dataloader:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            images, labels = batch[0], batch[1]
        else:
            images, labels = batch, None

        images = images.to(device, dtype=torch.float32)
        noise = torch.randn_like(images)
        x_t = sqrt_alpha * images + sqrt_one_minus * noise

        t_batch = torch.full((images.shape[0],), int(timestep), device=device, dtype=torch.long)
        _ = unet(x_t, t_batch)
        acts = collector.capture()
        all_acts.append(acts.cpu())

        if labels is not None:
            all_labels.append(labels.cpu())

    collector.remove()

    activations = torch.cat(all_acts, dim=0)
    labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0)
    return activations, labels
