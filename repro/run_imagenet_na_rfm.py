#!/usr/bin/env python3
"""ImageNet ADM NA-RFM sampling using additive repro-only code."""

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

from na_rfm.ddim_sampling import build_ddim_schedule, sample_ddim  # noqa: E402
from na_rfm.gaussian_x0_guidance import GaussianX0Guidance  # noqa: E402
from na_rfm.hooks import ActivationGuidanceHook, GuidanceWindow, RFMCFGWrapper  # noqa: E402
from na_rfm.model_loaders import load_openai_unet  # noqa: E402
from na_rfm.random_utils import StackedRandomGenerator  # noqa: E402


TARGET_NAMES = {
    111: "nematode",
    222: "kuvasz",
    333: "hamster",
    444: "tandem_bicycle",
}
CONCEPT_KEYS = ("direction", "concept_vector", "eigenvectors")


def parse_targets(value: str) -> list[int]:
    if value == "all":
        return sorted(TARGET_NAMES)
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def save_grid(samples: torch.Tensor, path: Path, nrow: int = 4) -> None:
    from PIL import Image

    samples = samples.clamp(-1, 1)
    samples = ((samples + 1) / 2 * 255).byte()
    ncol = (samples.shape[0] + nrow - 1) // nrow
    grid = Image.new("RGB", (nrow * 256, ncol * 256))
    for idx, img in enumerate(samples):
        pil = Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
        grid.paste(pil, ((idx % nrow) * 256, (idx // nrow) * 256))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def load_concept(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not any(key in data for key in CONCEPT_KEYS):
        raise ValueError(f"Concept missing direction/eigenvectors: {path}")
    return data


def find_concept(concepts_root: Path, target: int) -> Path:
    candidates = [
        concepts_root / f"direction_label{target}.pt",
        concepts_root / f"rfm_class_{target}.pt",
        concepts_root / f"concept_{target}.pt",
        concepts_root / f"class_{target}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    globbed = sorted(concepts_root.glob(f"**/*{target}*.pt"))
    if globbed:
        return globbed[0]
    raise FileNotFoundError(f"No RFM concept found for ImageNet class {target} in {concepts_root}")


def direction_from_concept(concept: dict, topk: int) -> torch.Tensor:
    if "direction" in concept:
        direction = concept["direction"]
        return direction.float() if torch.is_tensor(direction) else torch.tensor(direction, dtype=torch.float32)
    if "concept_vector" in concept:
        direction = concept["concept_vector"]
        return direction.float() if torch.is_tensor(direction) else torch.tensor(direction, dtype=torch.float32)

    eigenvectors = concept["eigenvectors"]
    if not torch.is_tensor(eigenvectors):
        eigenvectors = torch.tensor(eigenvectors)

    eigenvectors = eigenvectors.float()
    if eigenvectors.dim() == 2 and eigenvectors.shape[0] < eigenvectors.shape[1]:
        eigenvectors = eigenvectors.T

    k = min(topk, eigenvectors.shape[1])
    eigenvalues = concept.get("eigenvalues")
    if eigenvalues is None:
        eigenvalues = torch.ones(k, dtype=eigenvectors.dtype)
    elif not torch.is_tensor(eigenvalues):
        eigenvalues = torch.tensor(eigenvalues)
    weights = eigenvalues[:k].float()
    weights = weights / (weights.sum() + 1e-12)
    direction = (eigenvectors[:, :k] * weights.unsqueeze(0)).sum(dim=1)
    sign = concept.get("sign", 1)
    if torch.is_tensor(sign):
        sign = sign.item()
    return direction * float(sign)


def build_gaussian_guidance(args: argparse.Namespace, target: int) -> GaussianX0Guidance | None:
    if args.gaussian_strength == 0.0:
        return None
    filters_dir = Path(args.filters_dir)
    class_filter = filters_dir / f"class_{target}.pt"
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


def sample_target(
    *,
    unet,
    schedule,
    target: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Path]:
    concept_path = find_concept(Path(args.concepts_root), target)
    concept = load_concept(concept_path)
    direction = direction_from_concept(concept, args.rfm_topk).to(args.device, dtype=torch.float32)

    window = GuidanceWindow(start_sigma=args.rfm_sigma_lo, end_sigma=args.rfm_sigma_hi)
    hook = ActivationGuidanceHook(
        direction=direction,
        strength=args.rfm_strength,
        window=window,
        guidance_type="additive",
        adaptive_strength=True,
    )
    hook.register(unet, args.block_name)
    rfm_cfg = RFMCFGWrapper(
        hook=hook,
        cfg_scale=args.cfg_scale,
        window=window,
        clip_before_combine=args.clip_before_combine,
    )
    gaussian = build_gaussian_guidance(args, target)

    samples: list[torch.Tensor] = []
    try:
        for batch_start in range(0, args.num_samples, args.batch_size):
            batch_count = min(args.batch_size, args.num_samples - batch_start)
            seeds = list(range(args.seed_start + batch_start, args.seed_start + batch_start + batch_count))
            generator = StackedRandomGenerator(args.device, seeds)
            latents = generator.randn((batch_count, 3, 256, 256))
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

    del gaussian
    torch.cuda.empty_cache()
    return torch.cat(samples, dim=0), concept_path


def tensor_to_pil(samples: torch.Tensor):
    from PIL import Image

    samples = ((samples.clamp(-1, 1) + 1) / 2 * 255).byte()
    return [Image.fromarray(img.permute(1, 2, 0).cpu().numpy()) for img in samples]


def evaluate_imagenet(samples: torch.Tensor, target: int, args: argparse.Namespace) -> dict:
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(args.eval_model)
    model = AutoModelForImageClassification.from_pretrained(args.eval_model).to(args.device).eval()
    images = tensor_to_pil(samples)

    predictions: list[int] = []
    confidences: list[float] = []
    top5_hits = 0
    for start in range(0, len(images), args.eval_batch_size):
        inputs = processor(images[start : start + args.eval_batch_size], return_tensors="pt")
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        confidences.extend(probs[:, target].cpu().tolist())
        top5 = logits.topk(k=5, dim=-1).indices.cpu().tolist()
        top5_hits += sum(1 for row in top5 if target in row)

    correct = sum(1 for pred in predictions if int(pred) == target)
    total = len(predictions)
    del model
    torch.cuda.empty_cache()
    return {
        "eval_model": args.eval_model,
        "correct_top1": correct,
        "total": total,
        "top1_accuracy": correct / total * 100.0,
        "top5_correct": top5_hits,
        "top5_accuracy": top5_hits / total * 100.0,
        "target_confidence": sum(confidences) / total,
        "predictions": predictions,
        "target_confidences": confidences,
    }


def write_summary(path: Path, rows: Iterable[dict]) -> None:
    rows_by_class: dict[int, dict] = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text())
            for row in previous.get("classes", []):
                rows_by_class[int(row["class_idx"])] = row
        except Exception:
            rows_by_class = {}

    for row in rows:
        rows_by_class[int(row["class_idx"])] = row
    rows = [rows_by_class[key] for key in sorted(rows_by_class)]
    payload = {
        "average_top1_accuracy": sum(row["top1_accuracy"] for row in rows) / len(rows) if rows else 0.0,
        "average_top5_accuracy": sum(row["top5_accuracy"] for row in rows) / len(rows) if rows else 0.0,
        "classes": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ImageNet ADM NA-RFM reproduction runner")
    parser.add_argument("--targets", default="all")
    parser.add_argument("--model-path", default="checkpoints/openai_imagenet.pt")
    parser.add_argument("--filters-dir", default="repro/assets/imagenet_filters")
    parser.add_argument("--concepts-root", default="repro/assets/imagenet_concepts")
    parser.add_argument("--output-dir", default="results/camera_ready_repro/imagenet_adm/na_rfm")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--inference-steps", type=int, default=100)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block-name", default="input_blocks_15")
    parser.add_argument("--rfm-topk", type=int, default=5)
    parser.add_argument("--rfm-strength", type=float, required=True)
    parser.add_argument("--rfm-sigma-lo", type=float, required=True)
    parser.add_argument("--rfm-sigma-hi", type=float, required=True)
    parser.add_argument("--cfg-scale", type=float, required=True)
    parser.add_argument("--gaussian-strength", type=float, required=True)
    parser.add_argument("--gaussian-sigma-end", type=float, default=0.0)
    parser.add_argument("--gaussian-sigma-start", type=float, default=1e9)
    parser.add_argument("--gaussian-components", type=int, default=1000)
    parser.add_argument("--eval-model", default="facebook/deit-small-patch16-224")
    parser.add_argument(
        "--clip-before-combine",
        action="store_true",
        help="Use the older x0-CFG order that clips conditional/unconditional x0 before CFG combination.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = parse_targets(args.targets)
    print(f"Loading ImageNet ADM diffusion model: {args.model_path}", flush=True)
    unet = load_openai_unet(args.model_path, image_size=256, device=args.device)
    schedule = build_ddim_schedule(1000, args.inference_steps, 0.0001, 0.02, args.device)

    rows = []
    for target in targets:
        target_name = TARGET_NAMES.get(target, f"class_{target}")
        print(f"\nSampling {target} ({target_name})", flush=True)
        samples, concept_path = sample_target(unet=unet, schedule=schedule, target=target, args=args)

        class_dir = output_dir / f"class_{target}"
        class_dir.mkdir(parents=True, exist_ok=True)
        torch.save(samples, class_dir / "samples.pt")
        save_grid(samples, class_dir / "grid.png")

        metrics = evaluate_imagenet(samples, target, args)
        row = {
            "class_idx": target,
            "class_name": target_name,
            "concept_path": str(concept_path),
            "num_samples": args.num_samples,
            "seed_start": args.seed_start,
            "batch_size": args.batch_size,
            "lambda": args.gaussian_strength,
            "w_rfm": args.rfm_strength,
            "s": args.cfg_scale,
            "sigma_end": args.gaussian_sigma_end,
            "rfm_sigma_window": [args.rfm_sigma_lo, args.rfm_sigma_hi],
            "clip_before_combine": args.clip_before_combine,
            **metrics,
        }
        rows.append(row)
        with (class_dir / "metrics.json").open("w") as f:
            json.dump(row, f, indent=2)
        print(
            f"{target_name}: top1 {row['correct_top1']}/{row['total']} = {row['top1_accuracy']:.1f}%, "
            f"top5 {row['top5_correct']}/{row['total']} = {row['top5_accuracy']:.1f}%",
            flush=True,
        )
        del samples
        torch.cuda.empty_cache()

    write_summary(output_dir / "summary.json", rows)
    avg_top1 = sum(row["top1_accuracy"] for row in rows) / len(rows)
    avg_top5 = sum(row["top5_accuracy"] for row in rows) / len(rows)
    print(f"\nAverage top1 accuracy: {avg_top1:.1f}%", flush=True)
    print(f"Average top5 accuracy: {avg_top5:.1f}%", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
