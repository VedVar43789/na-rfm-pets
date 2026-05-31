"""Gaussian epsilon-space guidance for DDIM sampling."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from pca.gaussian_denoiser import GaussianDenoiser, load_gaussian_filter
from .hooks import GuidanceWindow


class GaussianEpsGuidance:
    """
    Gaussian guidance in epsilon space.

    Modes:
      - "full_diff": eps_model + gamma * (eps_class - eps_full)
      - "class_only": eps_model + gamma * (eps_class - eps_model)

    Progressive schedule:
      - If schedule is provided, strength is modulated across steps
      - Format: list of (end_step, multiplier) tuples
      - Example: [(20, 0.2), (60, 1.0), (100, 0.3)]
        - Steps 0-20: strength * 0.2
        - Steps 20-60: strength * 1.0
        - Steps 60-100: strength * 0.3
    """

    def __init__(
        self,
        class_filter_path: str,
        full_filter_path: Optional[str] = None,
        mode: str = "full_diff",
        strength: float = 1.0,
        window: Optional[GuidanceWindow] = None,
        num_components: Optional[int] = None,
        score_scaling: bool = False,
        device: str = "cuda",
        schedule: Optional[List[Tuple[int, float]]] = None,
        total_steps: int = 100,
    ):
        self.mode = mode
        self.strength = strength
        self.window = window or GuidanceWindow()
        self.score_scaling = score_scaling
        self.device = device
        self.schedule = schedule
        self.total_steps = total_steps

        class_filter = load_gaussian_filter(class_filter_path, device=device)
        self.class_denoiser = GaussianDenoiser(
            class_filter,
            num_components=num_components,
            denoising_mode="wiener",
        ).to(device)

        self.full_denoiser = None
        if full_filter_path is not None:
            full_filter = load_gaussian_filter(full_filter_path, device=device)
            self.full_denoiser = GaussianDenoiser(
                full_filter,
                num_components=num_components,
                denoising_mode="wiener",
            ).to(device)

        if self.mode == "full_diff" and self.full_denoiser is None:
            raise ValueError("full_diff mode requires full_filter_path")

    def _get_schedule_multiplier(self, step_idx: int) -> float:
        """Get strength multiplier based on progressive schedule."""
        if self.schedule is None:
            return 1.0

        for end_step, multiplier in self.schedule:
            if step_idx < end_step:
                return multiplier
        # Default to last multiplier if step exceeds all boundaries
        return self.schedule[-1][1] if self.schedule else 1.0

    def apply(
        self,
        x: torch.Tensor,
        eps_model: torch.Tensor,
        alpha_prod_t: torch.Tensor,
        step_idx: int,
        sigma: float,
    ) -> torch.Tensor:
        if not self.window.contains(step_idx, sigma):
            return eps_model

        dtype = x.dtype
        alpha_prod_t = alpha_prod_t.to(dtype)

        with torch.no_grad():
            x_class = self.class_denoiser.denoise(x, sigma)
            eps_class = (x - torch.sqrt(alpha_prod_t) * x_class) / torch.sqrt(
                1.0 - alpha_prod_t
            )

            if self.mode == "full_diff":
                x_full = self.full_denoiser.denoise(x, sigma)
                eps_full = (x - torch.sqrt(alpha_prod_t) * x_full) / torch.sqrt(
                    1.0 - alpha_prod_t
                )
                delta = eps_class - eps_full
            else:
                delta = eps_class - eps_model

        if self.score_scaling:
            delta = delta / (sigma + 1e-3)

        # Apply progressive schedule multiplier
        multiplier = self._get_schedule_multiplier(step_idx)
        effective_strength = self.strength * multiplier

        return eps_model + effective_strength * delta


EpsilonSpaceGuidance = GaussianEpsGuidance
