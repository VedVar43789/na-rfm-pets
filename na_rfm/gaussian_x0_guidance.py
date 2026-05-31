"""X0-level Gaussian guidance using Wiener denoiser difference.

This module provides Gaussian guidance that operates at the x0 (denoised estimate)
level, matching the proven approach from test_tfg_combined_guidance.py which
achieves 37.5% accuracy with eta=1.

The key insight is that we apply guidance directly to x0, not to epsilon:
    x0_guided = x0_pred + strength * (denoiser_class(x) - denoiser_full(x))

This requires proper space conversion between DDPM and EDM parameterizations:
    - DDPM: x_t = sqrt(alpha_bar) * x0 + sqrt(1-alpha_bar) * eps
    - EDM: x = x0 + sigma * eps, where sigma = sqrt((1-alpha_bar)/alpha_bar)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

# Handle both relative and absolute imports
try:
    from .hooks import GuidanceWindow
except ImportError:
    from hooks import GuidanceWindow

# Add path for GaussianDenoiser
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pca.gaussian_denoiser import GaussianDenoiser, load_gaussian_filter


def load_pca_filter(pca_paths: dict, device: str = 'cuda') -> dict:
    """Load PCA filter from numpy files and convert to CHW format.

    The stored PCA data is typically in HWC format (e.g., 256x256x3 flattened).
    We need to convert to CHW format for PyTorch.

    Args:
        pca_paths: dict with keys 'eigenvectors', 'eigenvalues', 'mean'
        device: target device

    Returns:
        Filter dictionary for GaussianDenoiser
    """
    eigenvectors = np.load(pca_paths['eigenvectors'])
    eigenvalues = np.load(pca_paths['eigenvalues'])
    mean = np.load(pca_paths['mean'])

    # Infer image dimensions
    D = mean.shape[0]
    # Try common image sizes: 256x256x3, 32x32x3
    for res in [256, 32, 64, 128]:
        if D == res * res * 3:
            img_resolution = res
            break
    else:
        raise ValueError(f"Cannot infer image resolution from D={D}")

    # Convert mean from HWC to CHW
    mean_chw = mean.reshape(img_resolution, img_resolution, 3).transpose(2, 0, 1).flatten()

    # Convert eigenvectors from HWC to CHW
    eigenvectors_chw = np.zeros_like(eigenvectors)
    for i in range(eigenvectors.shape[1]):
        eigenvectors_chw[:, i] = eigenvectors[:, i].reshape(
            img_resolution, img_resolution, 3
        ).transpose(2, 0, 1).flatten()

    return {
        'eigenvectors': torch.from_numpy(eigenvectors_chw).float().to(device),
        'eigenvalues': torch.from_numpy(eigenvalues).float().to(device),
        'mean_patch': torch.from_numpy(mean_chw).float().to(device),
        'img_resolution': img_resolution,
        'img_channels': 3,
    }


class GaussianX0Guidance:
    """X0-level Gaussian guidance using Wiener denoiser difference.

    This applies guidance at the x0 (denoised) level following the EDM formulation.
    The guidance is computed as:
        guidance = strength * (denoiser_class(x_edm, sigma) - denoiser_full(x_edm, sigma))

    where x_edm is the EDM-space representation of the DDPM x_t.

    Args:
        class_filter_path: Path to class-specific PCA filter (.pt or .npy directory)
        full_filter_path: Path to full-dataset PCA filter
        strength: Guidance strength (typically 3.0-7.0)
        window: GuidanceWindow for step/sigma range control
        num_components: Number of PCA components to use (None = all)
        device: Target device
    """

    def __init__(
        self,
        class_filter_path: str,
        full_filter_path: str,
        strength: float = 5.0,
        window: Optional[GuidanceWindow] = None,
        num_components: Optional[int] = None,
        device: str = "cuda",
    ):
        self.strength = strength
        self.window = window or GuidanceWindow()
        self.device = device

        # Load filters - handle both .pt files and .npy dictionaries
        class_path = Path(class_filter_path)
        full_path = Path(full_filter_path)

        if class_path.suffix == '.pt':
            class_filter = load_gaussian_filter(str(class_path), device=device)
        else:
            # Assume it's a dict of paths to .npy files
            class_filter = load_pca_filter(
                {'eigenvectors': class_filter_path,
                 'eigenvalues': str(class_path).replace('eigenvectors', 'eigenvalues'),
                 'mean': str(class_path).replace('eigenvectors', 'mean')},
                device=device
            )

        if full_path.suffix == '.pt':
            full_filter = load_gaussian_filter(str(full_path), device=device)
        else:
            full_filter = load_pca_filter(
                {'eigenvectors': full_filter_path,
                 'eigenvalues': str(full_path).replace('eigenvectors', 'eigenvalues'),
                 'mean': str(full_path).replace('eigenvectors', 'mean')},
                device=device
            )

        # Create denoisers with Wiener mode (optimal for diffusion)
        self.class_denoiser = GaussianDenoiser(
            class_filter,
            num_components=num_components,
            denoising_mode='wiener'
        ).to(device)

        self.full_denoiser = GaussianDenoiser(
            full_filter,
            num_components=num_components,
            denoising_mode='wiener'
        ).to(device)

    @classmethod
    def from_pca_dir(
        cls,
        pca_dir: str,
        target_class: int,
        strength: float = 5.0,
        window: Optional[GuidanceWindow] = None,
        num_components: Optional[int] = None,
        device: str = "cuda",
    ) -> "GaussianX0Guidance":
        """Create GaussianX0Guidance from a directory of PCA files.

        Expects files like:
            - eigenvectors_class_{label}.npy
            - eigenvalues_class_{label}.npy
            - mean_class_{label}.npy
            - eigenvectors_imagenet_full.npy (or similar full-dataset files)

        Args:
            pca_dir: Directory containing PCA files
            target_class: Target class label
            strength: Guidance strength
            window: Step/sigma window
            num_components: Number of PCA components
            device: Target device
        """
        pca_dir = Path(pca_dir)

        class_paths = {
            'eigenvectors': str(pca_dir / f'eigenvectors_class_{target_class}.npy'),
            'eigenvalues': str(pca_dir / f'eigenvalues_class_{target_class}.npy'),
            'mean': str(pca_dir / f'mean_class_{target_class}.npy'),
        }

        # Look for full dataset files
        full_candidates = [
            'eigenvectors_imagenet_full.npy',
            'eigenvectors_full.npy',
            'eigenvectors_full_dataset.npy',
        ]
        full_eigenvectors = None
        for candidate in full_candidates:
            if (pca_dir / candidate).exists():
                full_eigenvectors = str(pca_dir / candidate)
                break

        if full_eigenvectors is None:
            raise FileNotFoundError(f"No full-dataset eigenvectors found in {pca_dir}")

        full_paths = {
            'eigenvectors': full_eigenvectors,
            'eigenvalues': full_eigenvectors.replace('eigenvectors', 'eigenvalues'),
            'mean': full_eigenvectors.replace('eigenvectors', 'mean'),
        }

        class_filter = load_pca_filter(class_paths, device=device)
        full_filter = load_pca_filter(full_paths, device=device)

        instance = cls.__new__(cls)
        instance.strength = strength
        instance.window = window or GuidanceWindow()
        instance.device = device

        instance.class_denoiser = GaussianDenoiser(
            class_filter,
            num_components=num_components,
            denoising_mode='wiener'
        ).to(device)

        instance.full_denoiser = GaussianDenoiser(
            full_filter,
            num_components=num_components,
            denoising_mode='wiener'
        ).to(device)

        return instance

    def compute_guidance(
        self,
        x: torch.Tensor,
        alpha_prod_t: Union[torch.Tensor, float],
        step_idx: int,
        sigma: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        """Compute x0-level guidance.

        Args:
            x: Current DDPM-space sample x_t [B, C, H, W]
            alpha_prod_t: Alpha bar (cumulative product of alphas) at current timestep
            step_idx: Current DDIM step index
            sigma: EDM-style sigma (computed if not provided)

        Returns:
            Guidance tensor to add to x0 estimate, or None if outside window
        """
        # Convert alpha_prod_t to scalar
        if torch.is_tensor(alpha_prod_t):
            ab_t = float(alpha_prod_t.item())
        else:
            ab_t = float(alpha_prod_t)

        # Compute sigma if not provided
        if sigma is None:
            sigma = float(np.sqrt((1 - ab_t) / ab_t))

        # Check window
        if not self.window.contains(step_idx, sigma):
            return None

        # Convert DDPM space to EDM space
        # DDPM: x_t = sqrt(ab) * x0 + sqrt(1-ab) * eps
        # EDM: x = x0 + sigma * eps
        # Therefore: x_edm = x_t / sqrt(ab)
        x_edm = x / np.sqrt(ab_t)

        # Apply Wiener denoisers in EDM space
        with torch.no_grad():
            d_class = self.class_denoiser.denoise(x_edm.float(), sigma)
            d_full = self.full_denoiser.denoise(x_edm.float(), sigma)

        # Compute guidance (difference of denoised estimates)
        guidance = self.strength * (d_class - d_full)

        return guidance.to(x.dtype)


class GaussianX0GuidanceSimple:
    """Simplified x0-level Gaussian guidance with pre-loaded denoisers.

    Use this when you already have GaussianDenoiser instances loaded.
    """

    def __init__(
        self,
        class_denoiser: GaussianDenoiser,
        full_denoiser: GaussianDenoiser,
        strength: float = 5.0,
        window: Optional[GuidanceWindow] = None,
    ):
        self.class_denoiser = class_denoiser
        self.full_denoiser = full_denoiser
        self.strength = strength
        self.window = window or GuidanceWindow()

    def compute_guidance(
        self,
        x: torch.Tensor,
        alpha_prod_t: Union[torch.Tensor, float],
        step_idx: int,
        sigma: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        """Compute x0-level guidance."""
        if torch.is_tensor(alpha_prod_t):
            ab_t = float(alpha_prod_t.item())
        else:
            ab_t = float(alpha_prod_t)

        if sigma is None:
            sigma = float(np.sqrt((1 - ab_t) / ab_t))

        if not self.window.contains(step_idx, sigma):
            return None

        x_edm = x / np.sqrt(ab_t)

        with torch.no_grad():
            d_class = self.class_denoiser.denoise(x_edm.float(), sigma)
            d_full = self.full_denoiser.denoise(x_edm.float(), sigma)

        return (self.strength * (d_class - d_full)).to(x.dtype)
