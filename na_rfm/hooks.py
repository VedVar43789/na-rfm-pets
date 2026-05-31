"""Activation-space guidance hooks with step or sigma windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .block_utils import resolve_block


def _extract_eps(model_output: torch.Tensor, channels: int) -> torch.Tensor:
    """Handle learn-sigma outputs by taking the epsilon channels."""
    if model_output.shape[1] == 2 * channels:
        return model_output[:, :channels]
    return model_output


@dataclass
class GuidanceWindow:
    start_step: int = 0
    end_step: int = 10**9
    start_sigma: Optional[float] = None
    end_sigma: Optional[float] = None

    def contains(self, step: int, sigma: Optional[float]) -> bool:
        if self.start_sigma is not None and self.end_sigma is not None and sigma is not None:
            lo = min(self.start_sigma, self.end_sigma)
            hi = max(self.start_sigma, self.end_sigma)
            return lo <= sigma <= hi
        return self.start_step <= step <= self.end_step


class ActivationGuidanceHook:
    """
    Apply an activation-space direction at a target block.

    Uses an adaptive scaling by activation norm for stable magnitude.
    """

    def __init__(
        self,
        direction: torch.Tensor,
        strength: float = 1.0,
        window: Optional[GuidanceWindow] = None,
        guidance_type: str = "additive",
        adaptive_strength: bool = True,
    ):
        self.direction = direction
        self.strength = strength
        self.window = window or GuidanceWindow()
        self.guidance_type = guidance_type
        self.adaptive_strength = adaptive_strength
        self.enabled = True
        self.current_step = 0
        self.current_sigma: Optional[float] = None
        self._handle = None

    def set_step(self, step: int, sigma: Optional[float]):
        self.current_step = step
        self.current_sigma = sigma

    def should_apply(self) -> bool:
        return self.enabled and self.window.contains(self.current_step, self.current_sigma)

    def _apply_guidance(self, activation: torch.Tensor) -> torch.Tensor:
        direction = self.direction
        if direction.dim() > 1:
            direction = direction.reshape(-1)

        act_flat = activation.reshape(activation.shape[0], -1)
        if act_flat.shape[1] != direction.shape[0]:
            # Skip if mismatch (e.g., wrong block)
            return activation

        if self.guidance_type == "projection":
            unit = direction / (direction.norm() + 1e-8)
            proj = (act_flat * unit.unsqueeze(0)).sum(dim=1, keepdim=True)
            guided = act_flat + self.strength * proj * unit.unsqueeze(0)
        elif self.guidance_type == "amplify":
            unit = direction / (direction.norm() + 1e-8)
            proj = (act_flat * unit.unsqueeze(0)).sum(dim=1, keepdim=True)
            guided = act_flat + self.strength * proj.abs() * unit.unsqueeze(0)
        else:
            if self.adaptive_strength:
                act_norm = act_flat.norm(dim=1, keepdim=True)
                dir_norm = direction.norm() + 1e-8
                scale = self.strength * act_norm / dir_norm
                guided = act_flat + scale * direction.unsqueeze(0)
            else:
                guided = act_flat + self.strength * direction.unsqueeze(0)

        return guided.reshape_as(activation).to(activation.dtype)

    def _hook_fn(self, module, inputs, output):
        if not self.should_apply():
            return output

        if isinstance(output, tuple):
            main = output[0]
            rest = output[1:]
            if torch.is_tensor(main):
                main = self._apply_guidance(main)
            return (main, *rest)

        if torch.is_tensor(output):
            return self._apply_guidance(output)

        return output

    def register(self, model, block_name: str):
        """Register the hook on a specific block."""
        block = resolve_block(model, block_name)
        self._handle = block.register_forward_hook(self._hook_fn)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class RFMCFGWrapper:
    """Wraps activation hook for x0-level CFG-style guidance.

    This implements CFG (classifier-free guidance) at the x0 (denoised) level
    rather than the epsilon level, which has been shown to work better for
    activation-space steering.

    The formula applied is:
        x0 = x0_uncond + cfg_scale * (x0_cond - x0_uncond)

    where x0_cond comes from the model with the activation hook enabled,
    and x0_uncond comes from the model with the hook disabled.

    Args:
        hook: The ActivationGuidanceHook to control
        cfg_scale: CFG scale (typically 2.0-5.0)
        window: Optional GuidanceWindow to control when CFG is applied
    """

    def __init__(
        self,
        hook: ActivationGuidanceHook,
        cfg_scale: float = 3.0,
        window: Optional[GuidanceWindow] = None,
        clip_before_combine: bool = True,
    ):
        self.hook = hook
        self.cfg_scale = cfg_scale
        self.window = window or GuidanceWindow()
        self.clip_before_combine = clip_before_combine
        self.current_step = 0
        self.current_sigma: Optional[float] = None

    def should_apply(self, step: int, sigma: Optional[float]) -> bool:
        """Check if CFG should be applied at this step/sigma."""
        return self.window.contains(step, sigma)

    def compute_x0_cfg(
        self,
        model,
        x: torch.Tensor,
        t_batch: torch.Tensor,
        alpha_prod_t: torch.Tensor,
    ) -> torch.Tensor:
        """Run model twice and combine at x0 level using CFG.

        Args:
            model: The diffusion model
            x: Current noisy sample x_t [B, C, H, W]
            t_batch: Timestep tensor [B]
            alpha_prod_t: Alpha bar at current timestep

        Returns:
            Combined x0 prediction using CFG at x0 level
        """
        dtype = x.dtype
        channels = x.shape[1]

        # Convert alpha to scalars for computation
        sqrt_alpha = torch.sqrt(alpha_prod_t).to(dtype)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_prod_t).to(dtype)

        # Get conditioned prediction (with hook)
        self.hook.enabled = True
        out_cond = model(x, t_batch)
        if hasattr(out_cond, "sample"):
            out_cond = out_cond.sample
        eps_cond = _extract_eps(out_cond, channels)
        x0_cond = (x - sqrt_one_minus_alpha * eps_cond) / sqrt_alpha
        if self.clip_before_combine:
            x0_cond = x0_cond.clamp(-1.0, 1.0)

        # Get unconditioned prediction (without hook)
        self.hook.enabled = False
        out_uncond = model(x, t_batch)
        if hasattr(out_uncond, "sample"):
            out_uncond = out_uncond.sample
        eps_uncond = _extract_eps(out_uncond, channels)
        x0_uncond = (x - sqrt_one_minus_alpha * eps_uncond) / sqrt_alpha
        if self.clip_before_combine:
            x0_uncond = x0_uncond.clamp(-1.0, 1.0)

        # Re-enable hook for subsequent steps
        self.hook.enabled = True

        # CFG combination at x0 level
        x0_cfg = x0_uncond + self.cfg_scale * (x0_cond - x0_uncond)

        return x0_cfg
