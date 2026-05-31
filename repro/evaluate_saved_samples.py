#!/usr/bin/env python3
"""Evaluate saved reproduction samples for accuracy and quality metrics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
TFG_ROOT = REPO_ROOT / "baselines" / "tfg"
if str(TFG_ROOT) not in sys.path:
    sys.path.insert(0, str(TFG_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

IMAGENET_TARGETS = {
    111: "nematode",
    222: "kuvasz",
    333: "hamster",
    444: "tandem_bicycle",
}

BIRD_TARGETS = {
    111: "brown_headed_cowbird",
    222: "fairy_tern",
    333: "lucifer_hummingbird",
    444: "scarlet_macaw",
}


def parse_targets(value: str, available: Iterable[int]) -> list[int]:
    keys = list(available)
    if value == "all":
        return keys
    targets = [int(part.strip()) for part in value.split(",") if part.strip()]
    missing = [target for target in targets if target not in keys]
    if missing:
        raise SystemExit(f"Unknown target(s): {missing}")
    return targets


def tensor_to_pil(samples: torch.Tensor) -> list[Image.Image]:
    samples = samples.clamp(-1, 1)
    samples = ((samples + 1.0) * 127.5).round().to(torch.uint8)
    return [Image.fromarray(img.permute(1, 2, 0).cpu().numpy()) for img in samples]


def load_samples(path: Path) -> list[Image.Image]:
    samples = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(samples, dict):
        for key in ("samples", "images"):
            if key in samples:
                samples = samples[key]
                break
    if not torch.is_tensor(samples):
        raise TypeError(f"Unsupported sample payload in {path}: {type(samples)!r}")
    return tensor_to_pil(samples)


def maybe_limit(images: list[Image.Image], limit: int | None) -> list[Image.Image]:
    if limit is None or limit < 0:
        return images
    return images[:limit]


def load_cifar_refs(target: int, limit: int | None) -> list[Image.Image]:
    from datasets import concatenate_datasets, load_dataset

    dataset = load_dataset("cifar10")
    dataset = dataset.filter(lambda x: x == target, input_columns="label")
    dataset = concatenate_datasets([dataset["train"], dataset["test"]])
    if limit is not None and limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [img.resize((32, 32)) for img in dataset["img"]]


def load_imagenet_refs(target: int, limit: int | None) -> list[Image.Image]:
    from datasets import load_dataset

    dataset = load_dataset("imagenet-1k", split="train", streaming=True)
    refs: list[Image.Image] = []
    for item in dataset:
        if int(item["label"]) != target:
            continue
        refs.append(item["image"].convert("RGB").resize((256, 256)))
        if limit is not None and limit > 0 and len(refs) >= limit:
            break
    if not refs:
        raise RuntimeError(f"No ImageNet reference images found for target {target}")
    return refs


def load_imagenet_refs_many(targets: list[int], limit: int | None) -> dict[int, list[Image.Image]]:
    from datasets import load_dataset

    wanted = set(targets)
    refs = {target: [] for target in targets}
    dataset = load_dataset("imagenet-1k", split="train", streaming=True)
    for item in dataset:
        label = int(item["label"])
        if label not in wanted:
            continue
        if limit is None or limit < 0 or len(refs[label]) < limit:
            refs[label].append(item["image"].convert("RGB").resize((256, 256)))
        if limit is not None and limit > 0 and all(len(refs[target]) >= limit for target in targets):
            break
    missing = [target for target, images in refs.items() if not images]
    if missing:
        raise RuntimeError(f"No ImageNet reference images found for target(s): {missing}")
    return refs


def load_bird_refs(target: int, limit: int | None) -> list[Image.Image]:
    from datasets import concatenate_datasets, load_dataset

    splits = [
        load_dataset("chriamue/bird-species-dataset", split=split, trust_remote_code=True)
        for split in ("train", "validation", "test")
    ]
    dataset = concatenate_datasets(splits)
    dataset = dataset.filter(lambda x: x == target, input_columns="label")
    if limit is not None and limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [img.convert("RGB").resize((256, 256)) for img in dataset["image"]]


def load_celeba_refs(root: Path, limit: int | None) -> list[Image.Image]:
    files = sorted(root.glob("*.png")) + sorted(root.glob("*.jpg")) + sorted(root.glob("*.jpeg"))
    if limit is not None and limit > 0:
        files = files[:limit]
    if not files:
        raise RuntimeError(f"No CelebA reference images found under {root}")
    return [Image.open(path).convert("RGB") for path in files]


def compute_fid_is(
    refs: list[Image.Image],
    samples: list[Image.Image],
    *,
    device: str,
    batch_size: int,
) -> dict[str, float]:
    from evaluations.utils.fid import calculate_fid
    from evaluations.utils.inception_score import inception_score

    fid = calculate_fid(refs, samples, batch_size, device)
    inc = inception_score(samples, device, batch_size=batch_size)
    return {"fid": float(fid), "inception_score": float(inc)}


def compute_kid(
    refs: list[Image.Image],
    samples: list[Image.Image],
    *,
    device: str,
    batch_size: int,
) -> dict[str, float]:
    from torchmetrics.image.kid import KernelInceptionDistance

    kid = KernelInceptionDistance(subset_size=min(len(samples), 50), normalize=True).to(device)
    tf = transforms.Compose([transforms.Resize(299), transforms.ToTensor()])

    def batches(images: list[Image.Image]):
        for start in range(0, len(images), batch_size):
            yield torch.stack([tf(img) for img in images[start : start + batch_size]], dim=0).to(device)

    for batch in batches(refs):
        kid.update(batch, real=True)
    for batch in batches(samples):
        kid.update(batch, real=False)

    value = float(kid.compute()[0].item())
    return {"kid": value, "log_kid": float(math.log(value))}


def read_accuracy(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    keys = [
        "accuracy",
        "top1_accuracy",
        "top5_accuracy",
        "correct",
        "correct_top1",
        "top5_correct",
        "total",
    ]
    return {key: data[key] for key in keys if key in data}


def evaluate_class_dirs(args: argparse.Namespace) -> dict:
    root = Path(args.samples_root)
    if args.dataset == "cifar10":
        targets = parse_targets(args.targets, range(10))
        name_for = lambda target: CIFAR10_CLASSES[target]
        ref_loader = load_cifar_refs
        refs_by_target = None
    elif args.dataset == "imagenet":
        targets = parse_targets(args.targets, IMAGENET_TARGETS)
        name_for = lambda target: f"class_{target}"
        ref_loader = load_imagenet_refs
        refs_by_target = load_imagenet_refs_many(targets, args.num_ref)
    elif args.dataset == "bird-species":
        targets = parse_targets(args.targets, BIRD_TARGETS)
        name_for = lambda target: f"label_{target}"
        ref_loader = load_bird_refs
        refs_by_target = None
    else:
        raise SystemExit(f"Unsupported class-dir dataset: {args.dataset}")

    rows = []
    for target in targets:
        class_dir = root / name_for(target)
        sample_path = class_dir / "samples.pt"
        samples = maybe_limit(load_samples(sample_path), args.max_samples)
        refs = refs_by_target[target] if refs_by_target is not None else ref_loader(target, args.num_ref)
        metrics = compute_fid_is(refs, samples, device=args.device, batch_size=args.batch_size)
        metrics.update(read_accuracy(class_dir / "metrics.json"))
        metrics.update({
            "target": target,
            "name": name_for(target),
            "num_samples": len(samples),
            "num_ref": len(refs),
            "samples": str(sample_path),
        })
        rows.append(metrics)
        print(f"{name_for(target)}: FID={metrics['fid']:.3f}, IS={metrics['inception_score']:.3f}", flush=True)

    summary = {
        "dataset": args.dataset,
        "average_fid": sum(row["fid"] for row in rows) / len(rows),
        "average_inception_score": sum(row["inception_score"] for row in rows) / len(rows),
        "classes": rows,
    }
    if all("accuracy" in row for row in rows):
        summary["average_accuracy"] = sum(row["accuracy"] for row in rows) / len(rows)
    if all("top1_accuracy" in row for row in rows):
        summary["average_top1_accuracy"] = sum(row["top1_accuracy"] for row in rows) / len(rows)
    return summary


def evaluate_celeba(args: argparse.Namespace) -> dict:
    root = Path(args.samples_root)
    refs = load_celeba_refs(Path(args.celeba_ref_root), args.num_ref)
    rows = []
    for sample_path in sorted(root.glob("samples_*.pt")):
        samples = maybe_limit(load_samples(sample_path), args.max_samples)
        metrics = compute_kid(refs, samples, device=args.device, batch_size=args.batch_size)
        metrics.update({
            "name": sample_path.stem.removeprefix("samples_"),
            "num_samples": len(samples),
            "num_ref": len(refs),
            "samples": str(sample_path),
        })
        rows.append(metrics)
        print(f"{sample_path.name}: log-KID={metrics['log_kid']:.4f}", flush=True)

    if not rows:
        raise RuntimeError(f"No samples_*.pt files found under {root}")
    return {
        "dataset": "celebahq",
        "average_log_kid": sum(row["log_kid"] for row in rows) / len(rows),
        "average_kid": sum(row["kid"] for row in rows) / len(rows),
        "items": rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved reproduction samples")
    parser.add_argument("--dataset", choices=["cifar10", "imagenet", "bird-species", "celebahq"], required=True)
    parser.add_argument("--samples-root", required=True)
    parser.add_argument("--targets", default="all")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-ref", type=int, default=-1)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--celeba-ref-root", default="data/celeba_hq_256_30k")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dataset == "celebahq":
        summary = evaluate_celeba(args)
    else:
        summary = evaluate_class_dirs(args)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved quality summary to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
