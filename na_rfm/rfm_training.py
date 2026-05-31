"""Helpers for training RFM directions from activations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from na_rfm.dual_rfm import DualRFM, DualRFMResult


@dataclass
class RFMTrainConfig:
    bandwidth: float = 100.0
    reg: float = 1e-3
    center_grads: bool = False
    center_data: bool = True
    normalize: bool = True
    n_iters: int = 5
    top_k: Optional[int] = 5
    early_stop: bool = True
    early_stop_threshold: float = 1.1
    device: str = "auto"
    verbose: bool = False


def _train_val_split(
    X: torch.Tensor,
    y: torch.Tensor,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    split = int((1.0 - val_ratio) * len(idx))
    train_idx = idx[:split]
    val_idx = idx[split:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def train_binary_rfm(
    activations: torch.Tensor,
    labels: torch.Tensor,
    target_value: int,
    config: Optional[RFMTrainConfig] = None,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> DualRFMResult:
    """
    Train a binary RFM direction (target vs non-target).
    """
    config = config or RFMTrainConfig()

    X = activations.reshape(activations.shape[0], -1).float()
    y = (labels == target_value).float()

    X_train, y_train, X_val, y_val = _train_val_split(X, y, val_ratio=val_ratio, seed=seed)

    rfm = DualRFM(
        bandwidth=config.bandwidth,
        reg=config.reg,
        center_grads=config.center_grads,
        center_data=config.center_data,
        normalize=config.normalize,
        device=config.device,
        verbose=config.verbose,
    )

    result = rfm.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        n_iters=config.n_iters,
        top_k=config.top_k,
        early_stop=config.early_stop,
        early_stop_threshold=config.early_stop_threshold,
    )
    return result


def get_direction(
    result: DualRFMResult,
    topk: int = 1,
) -> torch.Tensor:
    """Compute a weighted direction from top-k eigenvectors with sign correction."""
    k = min(topk, result.eigenvalues.shape[0])
    eigenvalues = result.eigenvalues[:k]
    weights = eigenvalues / (eigenvalues.sum() + 1e-12)
    vecs = result.eigenvectors[:, :k]
    direction = (vecs * weights.unsqueeze(0)).sum(dim=1)
    return direction * result.sign
