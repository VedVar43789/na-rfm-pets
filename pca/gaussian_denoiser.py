"""
Gaussian Denoiser - Full-image PCA/Wiener denoising.

This module provides full-image denoising using PCA decomposition with
Wiener filtering. It's essentially a PatchLinearDenoiser where patch_size
equals the image size, treating the entire image as a single "patch".

This is the linear (Gaussian) approximation to the optimal denoiser,
assuming the data distribution is Gaussian with the learned covariance.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Union


class GaussianDenoiser(nn.Module):
    """
    Full-image Gaussian Denoiser using PCA/Wiener filtering.

    This denoiser applies linear filtering in PCA space to the entire image,
    assuming the underlying data distribution is Gaussian. The Wiener filter
    provides the optimal linear estimator for Gaussian data with known covariance.

    For an image with added Gaussian noise N(0, σ²I), the Wiener filter estimate is:
        x̂ = mean + Σ(Σ + σ²I)⁻¹(y - mean)

    In PCA space with eigenvalues λ_i:
        coefficient_i_denoised = λ_i / (λ_i + σ²) * coefficient_i_noisy

    Args:
        filter_dict: Dictionary containing PCA filter components:
            - 'eigenvectors': PCA eigenvectors [D, K]
            - 'eigenvalues': PCA eigenvalues [K]
            - 'mean_patch': Mean image [D] (flattened)
            - 'img_resolution': Image resolution (H = W)
            - 'img_channels': Number of channels
        num_components: Number of PCA components to use (None = all)
        denoising_mode: 'wiener' (σ-dependent) or 'one_step' (fixed scaling)
        epsilon: Numerical stability constant

    Denoising Modes:
        - 'wiener': λ/(λ+σ²) - optimal for iterative diffusion sampling
        - 'one_step': √(λ/(λ+σ²)) - for single-step denoising from Gaussian

    Example:
        >>> # Load full-image filter (patch_size = image_size)
        >>> filter_dict = torch.load('cifar10_full_image_filter.pt')
        >>> denoiser = GaussianDenoiser(filter_dict)
        >>> denoised = denoiser(noisy_images, sigma=1.0)
    """

    def __init__(
        self,
        filter_dict: Dict[str, torch.Tensor],
        num_components: Optional[int] = None,
        denoising_mode: str = 'wiener',
        epsilon: float = 1e-8
    ):
        super().__init__()
        self.epsilon = epsilon
        self.denoising_mode = denoising_mode.lower()

        if self.denoising_mode not in ['wiener', 'one_step']:
            raise ValueError(f"denoising_mode must be 'wiener' or 'one_step', got {denoising_mode}")

        # Extract filter components
        self.register_buffer('mean', filter_dict['mean_patch'])
        self.register_buffer('eigenvectors', filter_dict['eigenvectors'])
        self.register_buffer('eigenvalues', filter_dict['eigenvalues'])

        # Image properties
        if 'img_resolution' in filter_dict:
            self.img_resolution = int(filter_dict['img_resolution'])
        elif 'patch_size' in filter_dict:
            self.img_resolution = int(filter_dict['patch_size'])
        else:
            # Infer from eigenvector dimension
            D = self.eigenvectors.shape[0]
            # Assume square image, try common channel counts
            for C in [3, 1, 4]:
                res = int((D / C) ** 0.5)
                if res * res * C == D:
                    self.img_resolution = res
                    break
            else:
                raise ValueError("Cannot infer img_resolution from filter_dict")

        self.img_channels = filter_dict.get('img_channels', 3)

        # Select components
        total_components = self.eigenvectors.shape[1]
        if num_components is None:
            self.num_components = total_components
        else:
            self.num_components = min(num_components, total_components)

        # Create views for selected components
        self.register_buffer('top_eigenvectors', self.eigenvectors[:, :self.num_components])
        self.register_buffer('top_eigenvalues', self.eigenvalues[:self.num_components])

        # EDM sampler compatibility
        self.sigma_min = 0
        self.sigma_max = float('inf')
        self.label_dim = 0

    def round_sigma(self, sigma):
        """Round sigma values (identity)."""
        return torch.as_tensor(sigma)

    def denoise(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Apply full-image Gaussian denoising.

        Args:
            x: Noisy images [B, C, H, W]
            sigma: Noise level

        Returns:
            Denoised images [B, C, H, W]
        """
        B, C, H, W = x.shape
        original_shape = x.shape

        # Flatten to [B, D]
        x_flat = x.reshape(B, -1).to(self.eigenvectors.dtype)

        # Center
        centered = x_flat - self.mean.reshape(1, -1)

        # Project to PCA space
        coeffs = torch.matmul(centered, self.top_eigenvectors)  # [B, K]

        # Apply filtering
        if self.denoising_mode == 'wiener':
            wiener = self.top_eigenvalues / (self.top_eigenvalues + sigma**2 + self.epsilon)
            filtered_coeffs = coeffs * wiener.reshape(1, -1)
        else:  # one_step
            scaling = torch.sqrt(self.top_eigenvalues / (self.top_eigenvalues + sigma**2 + self.epsilon))
            scaling = torch.nan_to_num(scaling, nan=0.0)
            filtered_coeffs = coeffs * scaling.reshape(1, -1)

        # Project back
        denoised_centered = torch.matmul(filtered_coeffs, self.top_eigenvectors.T)

        # Add mean
        denoised_flat = denoised_centered + self.mean.reshape(1, -1)

        return denoised_flat.reshape(original_shape).to(x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        sigma: Union[float, torch.Tensor],
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass compatible with EDM sampler interface."""
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.flatten()[0].item()
        return self.denoise(x, sigma)

    def project_to_eigenspace(self, x: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
        """
        Project input to top eigenspace components and back.

        Args:
            x: Input images [B, C, H, W]
            dim: Number of components (default: all)

        Returns:
            Projected images [B, C, H, W]
        """
        if dim is None:
            dim = self.num_components
        dim = min(dim, self.eigenvectors.shape[1])

        B = x.shape[0]
        original_shape = x.shape
        x_flat = x.reshape(B, -1).to(self.eigenvectors.dtype)

        centered = x_flat - self.mean.reshape(1, -1)
        proj_vecs = self.eigenvectors[:, :dim]
        coeffs = torch.matmul(centered, proj_vecs)
        projected = torch.matmul(coeffs, proj_vecs.T) + self.mean.reshape(1, -1)

        return projected.reshape(original_shape).to(x.dtype)

    def get_coefficients(self, x: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
        """
        Extract PCA coefficients for input images.

        Args:
            x: Input images [B, C, H, W]
            dim: Number of components (default: all)

        Returns:
            Coefficients [B, dim]
        """
        if dim is None:
            dim = self.num_components
        dim = min(dim, self.eigenvectors.shape[1])

        B = x.shape[0]
        x_flat = x.reshape(B, -1).to(self.eigenvectors.dtype)
        centered = x_flat - self.mean.reshape(1, -1)

        return torch.matmul(centered, self.eigenvectors[:, :dim])

    def sample_from_prior(self, num_samples: int, device: str = 'cuda') -> torch.Tensor:
        """
        Sample images from the learned Gaussian prior.

        Args:
            num_samples: Number of samples to generate
            device: Device for output

        Returns:
            Sampled images [num_samples, C, H, W]
        """
        # Sample coefficients
        coeffs = torch.randn(num_samples, self.num_components, device=device)
        coeffs = coeffs * torch.sqrt(self.top_eigenvalues).reshape(1, -1).to(device)

        # Generate images
        images_flat = torch.matmul(coeffs, self.top_eigenvectors.T.to(device))
        images_flat = images_flat + self.mean.reshape(1, -1).to(device)

        return images_flat.reshape(num_samples, self.img_channels, self.img_resolution, self.img_resolution)


def create_gaussian_filter(
    images: torch.Tensor,
    max_components: Optional[int] = None,
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """
    Create a full-image Gaussian filter from training images.

    Args:
        images: Training images [N, C, H, W] in range [0, 255] or [-1, 1]
        max_components: Maximum number of PCA components (default: all)
        device: Device for computation

    Returns:
        Filter dictionary suitable for GaussianDenoiser
    """
    images = images.to(device).float()

    # Normalize to [-1, 1] if needed
    if images.max() > 1.0:
        images = images / 127.5 - 1.0

    N, C, H, W = images.shape
    D = C * H * W

    # Flatten
    X = images.reshape(N, D)

    # Compute mean
    mean = X.mean(dim=0)
    X_centered = X - mean.reshape(1, -1)

    # Compute covariance (use SVD for numerical stability)
    # For large D, we compute X.T @ X instead of X @ X.T
    if N < D:
        # Compute N x N gram matrix, then get eigenvectors
        gram = torch.matmul(X_centered, X_centered.T) / (N - 1)
        eigenvalues, V = torch.linalg.eigh(gram)

        # Sort descending
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        V = V[:, idx]

        # Convert to eigenvectors of covariance
        # U = X.T @ V @ diag(1/sqrt(eigenvalues))
        eigenvectors = torch.matmul(X_centered.T, V)
        eigenvectors = eigenvectors / torch.sqrt(eigenvalues.reshape(1, -1) * (N - 1) + 1e-10)
    else:
        # Direct covariance
        cov = torch.matmul(X_centered.T, X_centered) / (N - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # Sort descending
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

    # Keep only positive eigenvalues
    valid = eigenvalues > 1e-10
    eigenvalues = eigenvalues[valid]
    eigenvectors = eigenvectors[:, valid]

    # Limit components
    if max_components is not None:
        eigenvalues = eigenvalues[:max_components]
        eigenvectors = eigenvectors[:, :max_components]

    return {
        'mean_patch': mean,
        'eigenvectors': eigenvectors,
        'eigenvalues': eigenvalues,
        'img_resolution': H,
        'img_channels': C,
        'patch_size': H,  # For compatibility with patch-based loaders
        'num_images': N
    }


def save_gaussian_filter(filter_dict: Dict, path: str):
    """Save Gaussian filter to file."""
    torch.save(filter_dict, path)


def load_gaussian_filter(path: str, device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """Load Gaussian filter from file."""
    return torch.load(path, map_location=device, weights_only=False)


# Functional interface
def gaussian_denoise(
    noisy_images: torch.Tensor,
    filter_dict: Dict[str, torch.Tensor],
    sigma: float,
    num_components: Optional[int] = None,
    denoising_mode: str = 'wiener'
) -> torch.Tensor:
    """
    Functional interface for Gaussian denoising.

    Args:
        noisy_images: Noisy input [B, C, H, W]
        filter_dict: Gaussian filter dictionary
        sigma: Noise level
        num_components: Number of PCA components
        denoising_mode: 'wiener' or 'one_step'

    Returns:
        Denoised images [B, C, H, W]
    """
    denoiser = GaussianDenoiser(
        filter_dict=filter_dict,
        num_components=num_components,
        denoising_mode=denoising_mode
    ).to(device=noisy_images.device)

    return denoiser.denoise(noisy_images, sigma)
