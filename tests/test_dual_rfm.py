import numpy as np
import pytest
import torch

from na_rfm import DualRFM, compute_guidance_direction


@pytest.mark.parametrize("helper", [False, True])
def test_dual_rfm_cpu_returns_sign_corrected_direction(helper):
    pytest.importorskip("scipy")

    generator = torch.Generator().manual_seed(1)
    target = torch.randn(4, 12, generator=generator)
    other = torch.randn(4, 12, generator=generator)
    target[:, 0] += 2.0
    other[:, 0] -= 2.0

    if helper:
        direction, mean, std, auc, sign = compute_guidance_direction(
            target,
            other,
            bandwidth=5.0,
            reg=1e-2,
            n_iters=1,
            verbose=False,
            device="cpu",
        )
        assert direction.shape == (12,)
        assert mean.shape == (12,)
        assert std.shape == (12,)
        assert sign in (-1, 1)
        assert np.isfinite(direction).all()
        assert auc == -1
    else:
        features = torch.cat([target, other], dim=0)
        labels = torch.cat([torch.ones(len(target)), torch.zeros(len(other))])
        rfm = DualRFM(bandwidth=5.0, reg=1e-2, verbose=False, device="cpu")
        result = rfm.fit(features, labels, n_iters=1, top_k=2)
        assert result.eigenvectors.shape == (12, 2)
        assert result.eigenvalues.shape == (2,)
        assert result.sign in (-1, 1)
        assert torch.isfinite(result.eigenvectors).all()
        assert torch.isfinite(result.eigenvalues).all()
