"""Evaluation metrics wrapper matching TFG methodology.

Provides FID, Inception Score, KID computation using TFG's evaluation
infrastructure. Also includes TFG-compatible attribute definitions for
age, gender, and hair color.

Attribute Definitions (TFG-compatible):
    Age:
        - Guide network: nateraw/vit-age-classifier
        - Eval network: ibombonato/swin-age-classifier
        - Classes: 0=young (ages 0-30), 1=old (ages 50+)
        - Logic: young_prob = mean(classes 0-2), old_prob = mean(classes 5+)

    Gender:
        - Guide network: rizvandwiki/gender-classification-2
        - Eval network: rizvandwiki/gender-classification
        - Classes: 0=female, 1=male

    Hair Color:
        - Guide network: enzostvs/hair-color
        - Eval network: londe33/hair_v02
        - Classes: 0=brown, 1=red, 2=black, 3=blond
        - Note: TFG reorders outputs [2, 3, 0, 1] to standardize
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image

# TFG attribute mappings (guide network -> eval network)
TFG_GUIDE_TO_EVAL_NETWORK = {
    # CIFAR-10
    "resnet_cifar10.pt": "ahsanjavid/convnext-tiny-finetuned-cifar10",
    "timeclassifier_cifar10.pt": "ahsanjavid/convnext-tiny-finetuned-cifar10",
    # ImageNet
    "google/vit-base-patch16-224": "facebook/deit-small-patch16-224",
    "facebook/deit-small-patch16-224": "google/vit-base-patch16-224",
    "timeclassifier_imagenet.pt": "google/vit-base-patch16-224",
    # CelebA attributes
    "nateraw/vit-age-classifier": "ibombonato/swin-age-classifier",
    "rizvandwiki/gender-classification-2": "rizvandwiki/gender-classification",
    "enzostvs/hair-color": "londe33/hair_v02",
    # Bird species
    "ozzyonfire/bird-species-classifier": "chriamue/bird-species-classifier",
    "chriamue/bird-species-classifier": "dennisjooo/Birds-Classifier-EfficientNetB2",
    "dennisjooo/Birds-Classifier-EfficientNetB2": "chriamue/bird-species-classifier",
}

# Age class definitions for TFG evaluation
AGE_YOUNG_CLASSES = [0, 1, 2]  # Indices for ages 0-30 (approx)
AGE_OLD_CLASSES = [5, 6, 7, 8, 9]  # Indices for ages 50+ (approx)

# Hair color reordering for TFG evaluation
HAIR_COLOR_REORDER = [2, 3, 0, 1]  # TFG reorders: [brown, red, black, blond]


def _ensure_tfg_on_path() -> None:
    """Add TFG baseline to Python path if needed."""
    tfg_path = Path(__file__).resolve().parents[1] / "baselines" / "tfg"
    if str(tfg_path) not in sys.path:
        sys.path.insert(0, str(tfg_path))


def tensor_to_pil(samples: torch.Tensor) -> List[Image.Image]:
    """Convert [-1, 1] tensor batch to list of PIL images."""
    samples = samples.clamp(-1, 1)
    samples = (samples + 1.0) / 2.0
    samples = (samples * 255).round().byte()
    pil_images = []
    for img in samples:
        pil = Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
        pil_images.append(pil)
    return pil_images


def compute_image_quality_metrics(
    samples: torch.Tensor | List[Image.Image],
    dataset: str = "cifar10",
    target: Optional[int | List[int]] = None,
    reference_stats_path: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 64,
) -> Dict[str, float]:
    """Compute FID and Inception Score using TFG's evaluation code.

    Args:
        samples: Generated images as tensor [-1, 1] or list of PIL images.
        dataset: Dataset name ("cifar10", "imagenet", "celebahq").
        target: Target class index(es) for FID reference images.
        reference_stats_path: Path to cached FID statistics.
        device: Computation device.
        batch_size: Batch size for evaluation.

    Returns:
        Dict with "fid" and "inception_score" keys.
    """
    _ensure_tfg_on_path()
    from evaluations.utils.fid import calculate_fid
    from evaluations.utils.inception_score import inception_score
    from tasks.utils import load_image_dataset

    if torch.is_tensor(samples):
        images = tensor_to_pil(samples)
    else:
        images = samples

    metrics: Dict[str, float] = {}

    # Inception Score (always computable)
    is_score = inception_score(images, device, batch_size=batch_size)
    metrics["inception_score"] = float(is_score)

    # FID (needs reference images or cached stats)
    if dataset in ["cifar10", "imagenet", "cat"]:
        try:
            if reference_stats_path and Path(reference_stats_path).exists():
                # Use cached statistics
                ref_images = None
                fid = calculate_fid(
                    ref_images,
                    images,
                    batch_size,
                    device,
                    cache_path=reference_stats_path,
                )
            else:
                # Compute from reference images
                ref_images = load_image_dataset(
                    dataset, num_samples=-1, target=target, return_tensor=False
                )
                fid = calculate_fid(ref_images, images, batch_size, device)
            metrics["fid"] = float(fid)
        except Exception as e:
            print(f"FID computation failed: {e}")

    return metrics


def compute_kid(
    samples: torch.Tensor | List[Image.Image],
    dataset: str = "celebahq",
    device: str = "cuda",
    batch_size: int = 64,
) -> Dict[str, float]:
    """Compute KID for CelebA-HQ (TFG-style).

    Args:
        samples: Generated images.
        dataset: Dataset name.
        device: Computation device.
        batch_size: Batch size for evaluation.

    Returns:
        Dict with "kid" and "log_kid" keys.
    """
    _ensure_tfg_on_path()
    import numpy as np
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchvision import transforms

    from tasks.utils import load_image_dataset

    if torch.is_tensor(samples):
        images = tensor_to_pil(samples)
    else:
        images = samples

    kid = KernelInceptionDistance(
        subset_size=min(len(images), 50), normalize=True
    ).to(device)

    tf = transforms.Compose([transforms.Resize(299), transforms.ToTensor()])

    def image_proc(imgs):
        return torch.stack([tf(img) for img in imgs], dim=0).to(device)

    # Load reference images
    ref_images = load_image_dataset(
        dataset, num_samples=1000, target=-1, return_tensor=False
    )

    # Update with reference images
    for i in range(0, len(ref_images), batch_size):
        ref_batch = ref_images[i : i + batch_size]
        ref_batch = image_proc(ref_batch)
        kid.update(ref_batch, real=True)

    # Update with generated images
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        batch = image_proc(batch)
        kid.update(batch, real=False)

    score = kid.compute()
    kid_val = float(score[0].item())

    return {"kid": kid_val, "log_kid": float(np.log(kid_val))}


def compute_validity(
    samples: torch.Tensor | List[Image.Image],
    targets: int | List[int],
    guide_networks: str | List[str],
    device: str = "cuda",
    batch_size: int = 64,
    return_predictions: bool = True,
) -> Dict[str, Any]:
    """Compute classifier accuracy using TFG's evaluation methodology.

    Uses different classifiers for evaluation than guidance (TFG convention).

    Args:
        samples: Generated images.
        targets: Target class index(es).
        guide_networks: Network(s) used for guidance (will be mapped to eval networks).
        device: Computation device.
        batch_size: Batch size for evaluation.
        return_predictions: Whether to return predicted labels and confidences.

    Returns:
        Dict with "validity" (overall), per-network validity scores,
        and optionally "predictions" (list of predicted labels) and
        "confidences" (list of confidence scores).
    """
    _ensure_tfg_on_path()
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    if torch.is_tensor(samples):
        images = tensor_to_pil(samples)
    else:
        images = samples

    if not isinstance(guide_networks, list):
        guide_networks = [guide_networks]
    if not isinstance(targets, list):
        targets = [targets]

    correct = torch.zeros(len(images))
    validities = []
    validity_by_network = {}
    all_predictions = []
    all_confidences = []
    label_names = None

    for label, guide_network in zip(targets, guide_networks):
        # Map guide network to eval network (TFG convention)
        eval_network = TFG_GUIDE_TO_EVAL_NETWORK.get(guide_network, guide_network)

        # Load eval classifier
        processor = AutoImageProcessor.from_pretrained(eval_network)
        model = AutoModelForImageClassification.from_pretrained(eval_network)
        model.eval()
        model.to(device)

        # Get label names from model config
        if hasattr(model.config, "id2label"):
            label_names = model.config.id2label

        # Get predictions
        all_probs = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = processor(batch, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)

            # Apply TFG-specific transformations for age/hair
            if "age" in guide_network:
                # Young = mean(classes 0-2), Old = mean(classes 5+)
                probs = torch.cat(
                    [
                        probs[:, :3].mean(dim=1, keepdim=True),
                        probs[:, 5:].mean(dim=1, keepdim=True),
                    ],
                    dim=1,
                )
            elif "hair" in guide_network:
                # Reorder: [brown(2), red(3), black(0), blond(1)]
                probs = torch.cat(
                    [probs[:, 2:3], probs[:, 3:4], probs[:, 0:1], probs[:, 1:2]],
                    dim=1,
                )

            all_probs.append(probs)

        all_probs = torch.cat(all_probs, dim=0)
        preds = all_probs.argmax(dim=-1).cpu()
        confs = all_probs.max(dim=-1).values.cpu()

        correct_partial = (preds == int(label)).float()
        correct += correct_partial
        validity_partial = correct_partial.mean().item()
        validities.append(validity_partial)
        validity_by_network[f"validity_{guide_network}"] = validity_partial

        if return_predictions:
            all_predictions.extend(preds.tolist())
            all_confidences.extend(confs.tolist())

    overall_validity = correct.mean().item() / len(guide_networks)

    result = {
        "validity": overall_validity,
        **validity_by_network,
    }

    if return_predictions:
        result["predictions"] = all_predictions
        result["confidences"] = all_confidences
        # Map predictions to label names if available
        if label_names:
            result["prediction_names"] = [
                label_names.get(p, str(p)) for p in all_predictions
            ]
        result["eval_network"] = eval_network

    return result


def evaluate_all(
    samples: torch.Tensor | List[Image.Image],
    dataset: str,
    targets: int | List[int],
    guide_networks: str | List[str],
    device: str = "cuda",
    batch_size: int = 64,
    reference_stats_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run full TFG-compatible evaluation suite.

    Args:
        samples: Generated images.
        dataset: Dataset name ("cifar10", "imagenet", "celebahq").
        targets: Target class index(es).
        guide_networks: Network(s) used for guidance.
        device: Computation device.
        batch_size: Batch size.
        reference_stats_path: Path to cached FID statistics.

    Returns:
        Dict with all metrics: validity, fid, inception_score, kid/log_kid.
    """
    metrics: Dict[str, Any] = {}

    # Image quality metrics
    quality = compute_image_quality_metrics(
        samples,
        dataset=dataset,
        target=targets if isinstance(targets, list) else [targets],
        reference_stats_path=reference_stats_path,
        device=device,
        batch_size=batch_size,
    )
    metrics.update(quality)

    # KID for CelebA
    if dataset == "celebahq":
        kid_metrics = compute_kid(samples, dataset, device, batch_size)
        metrics.update(kid_metrics)

    # Classifier validity
    validity = compute_validity(
        samples, targets, guide_networks, device, batch_size
    )
    metrics.update(validity)

    return metrics
