import torch

from pca.gaussian_denoiser import (
    GaussianDenoiser,
    create_gaussian_filter,
    load_gaussian_filter,
    save_gaussian_filter,
)


def test_gaussian_filter_roundtrip_and_denoise(tmp_path):
    generator = torch.Generator().manual_seed(0)
    images = torch.randn(6, 3, 4, 4, generator=generator)

    filt = create_gaussian_filter(images, max_components=3, device="cpu")
    path = tmp_path / "filter.pt"
    save_gaussian_filter(filt, str(path))

    loaded = load_gaussian_filter(str(path), device="cpu")
    denoiser = GaussianDenoiser(loaded, denoising_mode="wiener")
    noisy = images[:2] + 0.1 * torch.randn(2, 3, 4, 4, generator=generator)
    denoised = denoiser.denoise(noisy, sigma=0.5)

    assert denoised.shape == noisy.shape
    assert torch.isfinite(denoised).all()
    assert denoiser.img_resolution == 4
    assert denoiser.img_channels == 3
    assert denoiser.num_components == 3
