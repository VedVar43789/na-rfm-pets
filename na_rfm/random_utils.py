"""Random number generation utilities for diffusion sampling.

Provides reproducible per-seed random number generation for batch sampling.
"""

import torch
from typing import List, Union


class StackedRandomGenerator:
    """Generator that produces reproducible per-seed random numbers.
    
    This is essential for reproducible evaluation where each sample in a batch
    uses a different seed, but the same seed always produces the same sample.
    
    Example:
        >>> gen = StackedRandomGenerator('cuda', seeds=[0, 1, 2, 3])
        >>> noise = gen.randn((4, 3, 32, 32))  # 4 samples, each with different seed
        
        # Same seeds = same noise
        >>> gen2 = StackedRandomGenerator('cuda', seeds=[0, 1, 2, 3])
        >>> noise2 = gen2.randn((4, 3, 32, 32))
        >>> assert torch.allclose(noise, noise2)
    """
    
    def __init__(self, device: Union[str, torch.device], seeds: List[int]):
        """Initialize with a list of seeds.
        
        Args:
            device: Device for generated tensors ('cuda' or 'cpu')
            seeds: List of integer seeds, one per sample in batch
        """
        self.device = device
        self.seeds = seeds
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) 
            for seed in seeds
        ]

    def randn(self, size: tuple, **kwargs) -> torch.Tensor:
        """Generate random normal tensor with shape [batch, ...].
        
        Args:
            size: Tensor shape, first dim must match number of seeds
            **kwargs: Additional arguments (dtype, etc.)
            
        Returns:
            Tensor of shape `size` with each batch element from its own generator
        """
        assert size[0] == len(self.generators), (
            f"Batch size {size[0]} != number of generators {len(self.generators)}"
        )
        return torch.stack([
            torch.randn(size[1:], generator=gen, device=self.device, **kwargs) 
            for gen in self.generators
        ])

    def randn_like(self, input: torch.Tensor) -> torch.Tensor:
        """Generate random normal tensor with same shape/dtype as input."""
        return self.randn(input.shape, dtype=input.dtype)

    def randint(self, low: int, high: int, size: tuple, **kwargs) -> torch.Tensor:
        """Generate random integers with shape [batch, ...].
        
        Args:
            low: Lower bound (inclusive)
            high: Upper bound (exclusive)
            size: Tensor shape, first dim must match number of seeds
            **kwargs: Additional arguments
            
        Returns:
            Tensor of random integers
        """
        assert size[0] == len(self.generators), (
            f"Batch size {size[0]} != number of generators {len(self.generators)}"
        )
        return torch.stack([
            torch.randint(low, high, size=size[1:], generator=gen, device=self.device, **kwargs) 
            for gen in self.generators
        ])
    
    def __len__(self) -> int:
        """Return number of seeds/generators."""
        return len(self.generators)
    
    def reset(self) -> None:
        """Reset all generators to their initial seeds."""
        self.generators = [
            torch.Generator(self.device).manual_seed(int(seed) % (1 << 32)) 
            for seed in self.seeds
        ]


def create_seed_generator(
    device: Union[str, torch.device],
    num_samples: int,
    base_seed: int = 0,
) -> StackedRandomGenerator:
    """Create a StackedRandomGenerator with consecutive seeds.
    
    Args:
        device: Device for generated tensors
        num_samples: Number of samples (and seeds)
        base_seed: Starting seed value
        
    Returns:
        StackedRandomGenerator with seeds [base_seed, base_seed+1, ...]
    """
    seeds = list(range(base_seed, base_seed + num_samples))
    return StackedRandomGenerator(device, seeds)
