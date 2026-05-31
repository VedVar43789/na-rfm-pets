#!/usr/bin/env python3
"""Validate that a PCA filter can be loaded and denoise a synthetic noisy mean."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pca.gaussian_denoiser import GaussianDenoiser, load_gaussian_filter


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Gaussian/Wiener PCA filter")
    parser.add_argument("filter", type=Path)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--components", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    filt = load_gaussian_filter(str(args.filter), device=args.device)
    denoiser = GaussianDenoiser(filt, num_components=args.components).to(args.device)

    mean = denoiser.mean.reshape(
        1, denoiser.img_channels, denoiser.img_resolution, denoiser.img_resolution
    )
    noisy = mean + torch.randn_like(mean) * args.sigma
    denoised = denoiser.denoise(noisy, args.sigma)

    report = {
        "filter": str(args.filter),
        "resolution": denoiser.img_resolution,
        "channels": denoiser.img_channels,
        "components": denoiser.num_components,
        "input_shape": list(noisy.shape),
        "output_shape": list(denoised.shape),
        "finite": bool(torch.isfinite(denoised).all().item()),
        "denoised_mean": float(denoised.mean().item()),
        "denoised_std": float(denoised.std().item()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
