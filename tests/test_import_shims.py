import importlib


def test_dual_rfm_minimal_public_exports():
    module = importlib.import_module("na_rfm")

    assert hasattr(module, "DualRFM")
    assert hasattr(module, "DualRFMResult")
    assert hasattr(module, "compute_guidance_direction")


def test_pca_minimal_public_exports():
    module = importlib.import_module("pca")

    assert hasattr(module, "GaussianDenoiser")
    assert hasattr(module, "load_gaussian_filter")
    assert hasattr(module, "save_gaussian_filter")
