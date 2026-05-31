"""DDIM sampling helpers with CFG-style hook support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import torch


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between tensors (per-sample).

    For batched inputs [B, C, H, W], performs slerp independently per sample.
    For small t values (< 0.1), approximates to linear interpolation for efficiency.
    """
    if t < 0.1:
        # For small t, slerp ≈ lerp (and lerp is simpler/safer)
        return (1 - t) * a + t * b.expand_as(a)

    # Per-sample slerp for larger t values
    batch_size = a.shape[0]
    result = torch.zeros_like(a)

    for i in range(batch_size):
        a_i = a[i].reshape(-1)
        b_i = b[0].reshape(-1) if b.shape[0] == 1 else b[i].reshape(-1)

        a_norm = a_i / (a_i.norm() + 1e-8)
        b_norm = b_i / (b_i.norm() + 1e-8)

        dot = (a_norm * b_norm).sum().clamp(-1, 1)
        omega = torch.acos(dot)

        if omega.abs() < 1e-6:
            result_i = (1 - t) * a_i + t * b_i
        else:
            sin_omega = torch.sin(omega)
            result_i = (torch.sin((1 - t) * omega) * a_i + torch.sin(t * omega) * b_i) / sin_omega

        # Normalize result and scale to interpolated norm
        interp_norm = (1 - t) * a_i.norm() + t * b_i.norm()
        result_i = result_i / (result_i.norm() + 1e-8) * interp_norm
        result[i] = result_i.view(a[i].shape)

    return result


@dataclass
class DDIMSchedule:
    timesteps: torch.Tensor
    alpha_prod_t: torch.Tensor
    alpha_prod_t_prev: torch.Tensor
    sigmas: torch.Tensor


def space_timesteps(num_timesteps: int, section_counts: int) -> list[int]:
    """Match TFG's timestep spacing for DDIM (descending order)."""
    section_counts = [section_counts]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"cannot divide section of {size} steps into {section_count}")
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return sorted(list(set(all_steps)), reverse=True)


def build_ddim_schedule(
    train_steps: int,
    inference_steps: int,
    beta_start: float,
    beta_end: float,
    device: torch.device | str,
) -> DDIMSchedule:
    """Create a DDIM schedule using the OpenAI DDPM beta schedule."""
    betas = np.linspace(beta_start, beta_end, train_steps, dtype=np.float64)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)

    timesteps = np.array(space_timesteps(train_steps, inference_steps), dtype=np.int64)
    alpha_prod = torch.from_numpy(alpha_bar[timesteps]).to(device=device, dtype=torch.float32)
    alpha_prod_prev = torch.from_numpy(
        np.append(alpha_bar[timesteps[1:]], 1.0)
    ).to(device=device, dtype=torch.float32)

    sigmas = torch.sqrt((1.0 - alpha_prod) / (alpha_prod + 1e-12))
    return DDIMSchedule(
        timesteps=torch.from_numpy(timesteps).to(device=device),
        alpha_prod_t=alpha_prod,
        alpha_prod_t_prev=alpha_prod_prev,
        sigmas=sigmas,
    )


def build_ddim_schedule_from_scheduler(
    scheduler,
    inference_steps: int,
    device: torch.device | str,
) -> DDIMSchedule:
    """Create a DDIM schedule using a diffusers scheduler."""
    scheduler.set_timesteps(inference_steps)
    timesteps = scheduler.timesteps.to(device)
    # Index with CPU tensor since alphas_cumprod is on CPU, then move to device
    alpha_prod = scheduler.alphas_cumprod[timesteps.cpu()].to(device=device, dtype=torch.float32)
    alpha_prod_prev = torch.cat(
        [alpha_prod[1:], torch.tensor([scheduler.final_alpha_cumprod], device=device)]
    ).to(dtype=torch.float32)
    sigmas = torch.sqrt((1.0 - alpha_prod) / (alpha_prod + 1e-12))
    return DDIMSchedule(
        timesteps=timesteps,
        alpha_prod_t=alpha_prod,
        alpha_prod_t_prev=alpha_prod_prev,
        sigmas=sigmas,
    )


def _extract_eps(model_output: torch.Tensor, channels: int) -> torch.Tensor:
    """Handle learn-sigma outputs by taking the epsilon channels."""
    if model_output.shape[1] == 2 * channels:
        return model_output[:, :channels]
    return model_output


def predict_eps(
    model,
    x: torch.Tensor,
    t_batch: torch.Tensor,
) -> torch.Tensor:
    """Run model and return epsilon prediction tensor."""
    out = model(x, t_batch)
    if hasattr(out, "sample"):
        out = out.sample
    return _extract_eps(out, x.shape[1])


def ddim_step(
    x: torch.Tensor,
    eps: torch.Tensor,
    alpha_prod_t: torch.Tensor,
    alpha_prod_t_prev: torch.Tensor,
    eta: float,
    clip_x0: bool,
    generator: Optional[torch.Generator] = None,
    x0_guidance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Single DDIM update step.

    Args:
        x0_guidance: Optional guidance to add to x0 estimate (like EDM approach)
    """
    dtype = x.dtype
    alpha_prod_t = alpha_prod_t.to(dtype)
    alpha_prod_t_prev = alpha_prod_t_prev.to(dtype)

    x0 = (x - torch.sqrt(1.0 - alpha_prod_t) * eps) / torch.sqrt(alpha_prod_t)

    # Add x0 guidance (EDM-style: directly to denoised estimate)
    if x0_guidance is not None:
        x0 = x0 + x0_guidance

    if clip_x0:
        x0 = x0.clamp(-1.0, 1.0)

    # Recompute eps from (possibly clipped) x0 to match TFG behavior
    eps_from_x0 = (x - torch.sqrt(alpha_prod_t) * x0) / torch.sqrt(1.0 - alpha_prod_t)

    sigma = eta * torch.sqrt(
        (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t)
        * (1.0 - alpha_prod_t / alpha_prod_t_prev)
    )
    pred_dir = torch.sqrt(torch.clamp(1.0 - alpha_prod_t_prev - sigma**2, min=0.0)) * eps_from_x0
    x_prev = torch.sqrt(alpha_prod_t_prev) * x0 + pred_dir

    if eta > 0:
        if generator is not None:
            noise = torch.randn(x_prev.shape, device=x_prev.device, dtype=x_prev.dtype, generator=generator)
        else:
            noise = torch.randn_like(x_prev)
        x_prev = x_prev + sigma * noise

    return x_prev


@torch.no_grad()
def sample_ddim(
    model,
    latents: torch.Tensor,
    schedule: DDIMSchedule,
    eta: float = 0.0,
    cfg_scale: float = 1.0,
    hooks: Optional[Sequence] = None,
    gaussian_x0_guidance: Optional[object] = None,
    rfm_cfg_wrapper: Optional[object] = None,
    clip_x0: bool = True,
    generator: Optional[torch.Generator] = None,
    noise_edit_direction: Optional[torch.Tensor] = None,
    noise_edit_strength: float = 0.0,
) -> torch.Tensor:
    """DDIM sampling with optional activation hooks and X0-level Gaussian guidance.

    Args:
        gaussian_x0_guidance: X0-space Gaussian guidance (EDM-style, preferred)
        rfm_cfg_wrapper: RFM CFG wrapper for x0-level CFG-style guidance
        noise_edit_direction: Optional noise direction to SLERP toward
        noise_edit_strength: Strength of noise editing (0.0 = no edit, 0.002-0.005 typical)
    """
    x = latents

    # Apply noise editing if direction is provided
    if noise_edit_direction is not None and noise_edit_strength > 0:
        noise_dir = noise_edit_direction.to(x.device, dtype=x.dtype)
        x = slerp(x, noise_dir.expand_as(x), noise_edit_strength)

    hooks = list(hooks or [])

    for step_idx in range(len(schedule.timesteps)):
        t = schedule.timesteps[step_idx]
        alpha_prod_t = schedule.alpha_prod_t[step_idx]
        alpha_prod_t_prev = schedule.alpha_prod_t_prev[step_idx]
        sigma = schedule.sigmas[step_idx].item()

        t_batch = torch.full(
            (x.shape[0],), int(t), device=x.device, dtype=torch.long
        )

        for hook in hooks:
            if hasattr(hook, "set_step"):
                hook.set_step(step_idx, sigma)

        # Check if we should use RFM CFG wrapper for x0-level CFG
        use_rfm_cfg = (
            rfm_cfg_wrapper is not None
            and rfm_cfg_wrapper.should_apply(step_idx, sigma)
        )

        if use_rfm_cfg:
            # RFM CFG wrapper computes x0 directly with CFG at x0 level
            x0_pred = rfm_cfg_wrapper.compute_x0_cfg(model, x, t_batch, alpha_prod_t)

            # X0-space Gaussian guidance
            x0_guidance = None
            if gaussian_x0_guidance is not None:
                x0_guidance = gaussian_x0_guidance.compute_guidance(x, alpha_prod_t, step_idx, sigma)
                if x0_guidance is not None:
                    x0_pred = x0_pred + x0_guidance

            if clip_x0:
                x0_pred = x0_pred.clamp(-1.0, 1.0)

            # Recompute eps from x0 for DDIM step
            eps = (x - torch.sqrt(alpha_prod_t) * x0_pred) / torch.sqrt(1.0 - alpha_prod_t)

            # DDIM step (without x0_guidance since we already applied it)
            x = ddim_step(
                x, eps, alpha_prod_t, alpha_prod_t_prev, eta, clip_x0=False,
                generator=generator, x0_guidance=None
            )
        else:
            # Standard CFG at eps level (or no CFG)
            if cfg_scale > 1.0 and hooks:
                for hook in hooks:
                    hook.enabled = True
                eps_guided = predict_eps(model, x, t_batch)

                for hook in hooks:
                    hook.enabled = False
                eps_unguided = predict_eps(model, x, t_batch)

                for hook in hooks:
                    hook.enabled = True
                eps = eps_unguided + cfg_scale * (eps_guided - eps_unguided)
            else:
                eps = predict_eps(model, x, t_batch)

            # X0-space guidance (EDM-style approach)
            x0_guidance = None
            if gaussian_x0_guidance is not None:
                x0_guidance = gaussian_x0_guidance.compute_guidance(x, alpha_prod_t, step_idx, sigma)

            x = ddim_step(
                x, eps, alpha_prod_t, alpha_prod_t_prev, eta, clip_x0,
                generator=generator, x0_guidance=x0_guidance
            )

    return x
