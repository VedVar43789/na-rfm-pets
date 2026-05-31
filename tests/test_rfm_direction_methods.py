from types import SimpleNamespace

import torch

from na_rfm import rfm_training


def _build_result_and_concept():
    generator = torch.Generator().manual_seed(0)
    eigenvectors = torch.randn(6, 4, generator=generator)
    eigenvalues = torch.tensor([4.0, 3.0, 2.0, 1.0])
    sign = -1

    result = SimpleNamespace(
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        sign=sign,
    )

    concept = {
        "eigenvectors": eigenvectors.clone(),
        "eigenvalues": eigenvalues.clone(),
        "sign": torch.tensor(sign),
    }

    return result, concept


def test_rfm_direction_matches_cifar_concept_direction():
    result, concept = _build_result_and_concept()
    topk = 3

    expected = rfm_training.get_direction(result, topk=topk)
    eigenvalues = concept["eigenvalues"][:topk]
    weights = eigenvalues / (eigenvalues.sum() + 1e-12)
    actual = (concept["eigenvectors"][:, :topk] * weights.unsqueeze(0)).sum(dim=1)
    actual = actual * int(concept["sign"].item())

    assert torch.allclose(actual, expected)
