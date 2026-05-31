#!/usr/bin/env python3
"""CIFAR-10 NA-RFM sampling using the shared x0/RFM wrapper interfaces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TFG_ROOT = PROJECT_ROOT / "baselines" / "tfg"
if TFG_ROOT.exists() and str(TFG_ROOT) not in sys.path:
    sys.path.insert(0, str(TFG_ROOT))

from repro.run_cifar10_na_only import (  # noqa: E402
    CIFAR10_CLASSES,
    classify_samples,
    load_classifier,
)
from na_rfm.ddim_sampling import build_ddim_schedule, sample_ddim  # noqa: E402
from na_rfm.gaussian_x0_guidance import GaussianX0Guidance  # noqa: E402
from na_rfm.hooks import ActivationGuidanceHook, GuidanceWindow, RFMCFGWrapper  # noqa: E402
from na_rfm.model_loaders import load_openai_unet  # noqa: E402
from na_rfm.random_utils import StackedRandomGenerator  # noqa: E402
from na_rfm.eval_metrics import compute_validity  # noqa: E402


CONCEPT_KEYS = ("direction", "concept_vector", "eigenvectors")
CIFAR_HF_EVAL_MODEL = "ahsanjavid/convnext-tiny-finetuned-cifar10"


def parse_classes(value: str) -> list[int]:
    if value == "all":
        return list(range(len(CIFAR10_CLASSES)))
    classes: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            classes.append(int(token))
        else:
            classes.append(CIFAR10_CLASSES.index(token))
    return classes


def save_grid(samples: torch.Tensor, path: Path, nrow: int = 8) -> None:
    from PIL import Image

    samples = samples.clamp(-1, 1)
    samples = ((samples + 1) / 2 * 255).byte()
    ncol = (samples.shape[0] + nrow - 1) // nrow
    grid = Image.new("RGB", (nrow * 32, ncol * 32))
    for idx, img in enumerate(samples):
        pil = Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
        grid.paste(pil, ((idx % nrow) * 32, (idx // nrow) * 32))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def load_concept(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not any(key in data for key in CONCEPT_KEYS):
        raise ValueError(f"Concept missing direction/eigenvectors: {path}")
    return data


def find_concept(concepts_root: Path, class_idx: int) -> Path:
    class_name = CIFAR10_CLASSES[class_idx]
    candidates = [
        concepts_root / f"{class_name}.pt",
        concepts_root / f"class_{class_idx}.pt",
        concepts_root / f"binary_rfm_{class_name}_t93_input_blocks_9.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    globbed = sorted(concepts_root.glob(f"**/*{class_name}*.pt"))
    if globbed:
        return globbed[0]
    raise FileNotFoundError(f"No concept found for {class_name} in {concepts_root}")


def direction_from_concept(concept: dict, topk: int) -> torch.Tensor:
    if "direction" in concept:
        direction = concept["direction"]
        return direction.float() if torch.is_tensor(direction) else torch.tensor(direction, dtype=torch.float32)
    if "concept_vector" in concept:
        direction = concept["concept_vector"]
        return direction.float() if torch.is_tensor(direction) else torch.tensor(direction, dtype=torch.float32)

    eigenvectors = concept["eigenvectors"]
    eigenvalues = concept.get("eigenvalues")
    if not torch.is_tensor(eigenvectors):
        eigenvectors = torch.tensor(eigenvectors)
    if eigenvalues is None:
        eigenvalues = torch.ones(eigenvectors.shape[0])
    if not torch.is_tensor(eigenvalues):
        eigenvalues = torch.tensor(eigenvalues)

    eigenvectors = eigenvectors.float()
    eigenvalues = eigenvalues.float()
    if eigenvectors.dim() == 2 and eigenvectors.shape[0] < eigenvectors.shape[1]:
        eigenvectors = eigenvectors.T

    k = min(topk, eigenvalues.numel(), eigenvectors.shape[1])
    weights = eigenvalues[:k] / (eigenvalues[:k].sum() + 1e-12)
    direction = (eigenvectors[:, :k] * weights.unsqueeze(0)).sum(dim=1)
    sign = concept.get("sign", 1)
    if torch.is_tensor(sign):
        sign = sign.item()
    return direction * float(sign)


def build_gaussian_guidance(args: argparse.Namespace, class_idx: int) -> GaussianX0Guidance | None:
    if args.gaussian_strength == 0.0:
        return None
    filters_dir = Path(args.filters_dir)
    class_filter = filters_dir / f"class_{class_idx}.pt"
    full_filter = filters_dir / "full_dataset.pt"
    if not class_filter.exists():
        raise FileNotFoundError(f"Missing Gaussian filter: {class_filter}")
    if not full_filter.exists():
        raise FileNotFoundError(f"Missing full Gaussian filter: {full_filter}")

    return GaussianX0Guidance(
        class_filter_path=str(class_filter),
        full_filter_path=str(full_filter),
        strength=args.gaussian_strength,
        window=GuidanceWindow(
            start_sigma=args.gaussian_sigma_end,
            end_sigma=args.gaussian_sigma_start,
        ),
        num_components=args.gaussian_components,
        device=args.device,
    )


def evaluate_local_resnet(samples: torch.Tensor, class_idx: int, classifier, args: argparse.Namespace) -> dict:
    preds = []
    for start in range(0, samples.shape[0], args.eval_batch_size):
        batch = samples[start : start + args.eval_batch_size].to(args.device)
        preds.append(classify_samples(batch, classifier, args.device).cpu())
    predictions = torch.cat(preds, dim=0)
    correct = (predictions == class_idx)
    return {
        "eval_mode": "local_resnet",
        "correct": int(correct.sum().item()),
        "total": int(samples.shape[0]),
        "accuracy": float(correct.float().mean().item() * 100.0),
        "predictions": predictions.tolist(),
    }


def tensor_to_pil(samples: torch.Tensor):
    from PIL import Image

    samples = ((samples.clamp(-1, 1) + 1) / 2 * 255).byte()
    return [Image.fromarray(img.permute(1, 2, 0).cpu().numpy()) for img in samples]


def _normal_label(label: str) -> str:
    return label.lower().replace("-", "_").replace(" ", "_")


def cifar_label_maps(model) -> tuple[dict[int, int], dict[int, int]]:
    id2label = getattr(model.config, "id2label", {}) or {}
    cifar_to_logit: dict[int, int] = {}
    logit_to_cifar: dict[int, int] = {}
    class_lookup = {_normal_label(name): idx for idx, name in enumerate(CIFAR10_CLASSES)}
    for raw_idx, label in id2label.items():
        logit_idx = int(raw_idx)
        normalized = _normal_label(str(label))
        if normalized in class_lookup:
            cifar_idx = class_lookup[normalized]
            cifar_to_logit[cifar_idx] = logit_idx
            logit_to_cifar[logit_idx] = cifar_idx
        elif normalized.startswith("label_"):
            suffix = normalized.removeprefix("label_")
            if suffix.isdigit():
                cifar_idx = int(suffix)
                cifar_to_logit.setdefault(cifar_idx, logit_idx)
                logit_to_cifar.setdefault(logit_idx, cifar_idx)

    for idx in range(len(CIFAR10_CLASSES)):
        cifar_to_logit.setdefault(idx, idx)
        logit_to_cifar.setdefault(idx, idx)
    return cifar_to_logit, logit_to_cifar


def evaluate_hf_convnext(samples: torch.Tensor, class_idx: int, args: argparse.Namespace) -> dict:
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(args.eval_model)
    model = AutoModelForImageClassification.from_pretrained(args.eval_model).to(args.device).eval()
    cifar_to_logit, logit_to_cifar = cifar_label_maps(model)
    target_logit = cifar_to_logit[class_idx]
    images = tensor_to_pil(samples)

    predictions: list[int] = []
    confidences: list[float] = []
    for start in range(0, len(images), args.eval_batch_size):
        inputs = processor(images[start : start + args.eval_batch_size], return_tensors="pt")
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        predictions.extend(logit_to_cifar[int(idx)] for idx in logits.argmax(dim=-1).cpu().tolist())
        confidences.extend(probs[:, target_logit].cpu().tolist())

    correct = sum(1 for pred in predictions if pred == class_idx)
    total = len(predictions)
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {
        "eval_mode": "hf_convnext",
        "eval_model": args.eval_model,
        "correct": correct,
        "total": total,
        "accuracy": correct / total * 100.0,
        "target_confidence": sum(confidences) / total,
        "predictions": predictions,
        "target_confidences": confidences,
    }


def evaluate_tfg(samples: torch.Tensor, class_idx: int, args: argparse.Namespace) -> dict:
    metrics = compute_validity(
        samples,
        targets=class_idx,
        guide_networks="resnet_cifar10.pt",
        device=args.device,
        batch_size=args.eval_batch_size,
    )
    return {
        "eval_mode": "tfg",
        "correct": int(round(float(metrics["validity"]) * samples.shape[0])),
        "total": int(samples.shape[0]),
        "accuracy": float(metrics["validity"] * 100.0),
        **metrics,
    }


def sample_class(
    *,
    unet,
    schedule,
    class_idx: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Path]:
    class_name = CIFAR10_CLASSES[class_idx]
    concept_path = find_concept(Path(args.concepts_root), class_idx)
    concept = load_concept(concept_path)
    direction = direction_from_concept(concept, args.rfm_topk).to(args.device, dtype=torch.float32)

    hook = ActivationGuidanceHook(
        direction=direction,
        strength=args.rfm_strength,
        window=GuidanceWindow(start_sigma=args.rfm_sigma_lo, end_sigma=args.rfm_sigma_hi),
        guidance_type="additive",
        adaptive_strength=True,
    )
    hook.register(unet, args.block_name)
    rfm_cfg = RFMCFGWrapper(
        hook=hook,
        cfg_scale=args.cfg_scale,
        window=GuidanceWindow(start_sigma=args.rfm_sigma_lo, end_sigma=args.rfm_sigma_hi),
        clip_before_combine=False,
    )
    gaussian = build_gaussian_guidance(args, class_idx)

    samples: list[torch.Tensor] = []
    try:
        for batch_start in range(0, args.num_samples, args.batch_size):
            batch_count = min(args.batch_size, args.num_samples - batch_start)
            seeds = list(range(args.seed_start + batch_start, args.seed_start + batch_start + batch_count))
            generator = StackedRandomGenerator(args.device, seeds)
            latents = generator.randn((batch_count, 3, 32, 32))
            batch = sample_ddim(
                unet,
                latents,
                schedule=schedule,
                eta=args.eta,
                cfg_scale=1.0,
                hooks=[hook],
                gaussian_x0_guidance=gaussian,
                rfm_cfg_wrapper=rfm_cfg,
                clip_x0=True,
                generator=None,
            )
            samples.append(batch.cpu())
    finally:
        hook.remove()

    return torch.cat(samples, dim=0), concept_path


def write_summary(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    avg = sum(float(row["accuracy"]) for row in rows) / len(rows) if rows else 0.0
    payload = {"average_accuracy": avg, "classes": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 NA-RFM reproduction runner")
    parser.add_argument("--classes", default="all")
    parser.add_argument("--model-path", default="data/checkpoints/openai_cifar10.pt")
    parser.add_argument("--filters-dir", default="repro/assets/cifar10_filters")
    parser.add_argument("--concepts-root", default="repro/assets/cifar10_concepts")
    parser.add_argument("--output-dir", default="results/camera_ready_repro/cifar10/na_rfm")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--inference-steps", type=int, default=100)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block-name", default="input_blocks_9")
    parser.add_argument("--rfm-topk", type=int, default=3)
    parser.add_argument("--rfm-strength", type=float, default=1.0)
    parser.add_argument("--rfm-sigma-lo", type=float, default=0.01)
    parser.add_argument("--rfm-sigma-hi", type=float, default=11.69)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--gaussian-strength", type=float, default=3.0)
    parser.add_argument("--gaussian-sigma-end", type=float, default=3.33)
    parser.add_argument("--gaussian-sigma-start", type=float, default=1e9)
    parser.add_argument("--gaussian-components", type=int, default=1000)
    parser.add_argument("--eval-mode", choices=["hf_convnext", "local_resnet", "tfg"], default="hf_convnext")
    parser.add_argument("--eval-model", default=CIFAR_HF_EVAL_MODEL)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = parse_classes(args.classes)
    print(f"Loading CIFAR-10 diffusion model: {args.model_path}")
    unet = load_openai_unet(args.model_path, image_size=32, device=args.device)
    schedule = build_ddim_schedule(1000, args.inference_steps, 0.0001, 0.02, args.device)

    classifier = None
    if args.eval_mode == "local_resnet":
        classifier = load_classifier(args.device)
        if classifier is None:
            raise RuntimeError("Local CIFAR-10 classifier could not be loaded")

    rows = []
    for class_idx in classes:
        class_name = CIFAR10_CLASSES[class_idx]
        print(f"\nSampling {class_idx} ({class_name})")
        samples, concept_path = sample_class(unet=unet, schedule=schedule, class_idx=class_idx, args=args)

        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        torch.save(samples, class_dir / "samples.pt")
        save_grid(samples, class_dir / "grid.png")

        if args.eval_mode == "hf_convnext":
            metrics = evaluate_hf_convnext(samples, class_idx, args)
        elif args.eval_mode == "local_resnet":
            metrics = evaluate_local_resnet(samples, class_idx, classifier, args)
        else:
            metrics = evaluate_tfg(samples, class_idx, args)

        row = {
            "class_idx": class_idx,
            "class_name": class_name,
            "concept_path": str(concept_path),
            "num_samples": args.num_samples,
            "seed_start": args.seed_start,
            "batch_size": args.batch_size,
            "lambda": args.gaussian_strength,
            "w_rfm": args.rfm_strength,
            "s": args.cfg_scale,
            **metrics,
        }
        rows.append(row)
        with (class_dir / "metrics.json").open("w") as f:
            json.dump(row, f, indent=2)
        print(f"{class_name}: {row['correct']}/{row['total']} = {row['accuracy']:.1f}%")

    write_summary(output_dir / "summary.json", rows)
    avg = sum(float(row["accuracy"]) for row in rows) / len(rows)
    print(f"\nAverage accuracy: {avg:.1f}%")
    print(f"Results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
