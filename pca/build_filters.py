#!/usr/bin/env python3
"""Build full-image PCA/Wiener filters for NA-RFM noise alignment."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from pca.gaussian_denoiser import save_gaussian_filter


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
LARGE_DATA_THRESHOLD = 100_000


def choose_backend(requested: str, num_images: int, device: str) -> str:
    """Select the PCA preprocessing backend."""
    if requested != "auto":
        return requested
    if num_images > LARGE_DATA_THRESHOLD and torch.cuda.is_available() and device.startswith("cuda"):
        return "gpu-preprocess"
    return "cpu"


def image_paths(root: Path, limit: int) -> list[Path]:
    paths = sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if limit > 0:
        paths = paths[:limit]
    if len(paths) < 2:
        raise ValueError(f"Need at least two images in {root}, found {len(paths)}")
    return paths


def all_image_paths(root: Path, limit: int) -> list[Path]:
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if limit > 0:
        paths = paths[:limit]
    if len(paths) < 2:
        raise ValueError(f"Need at least two images under {root}, found {len(paths)}")
    return paths


def pil_to_chw_array(image: Image.Image, image_size: int) -> np.ndarray:
    image = image.convert("RGB").resize((image_size, image_size), Image.LANCZOS)
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
    return array / 127.5 - 1.0


def preprocess_cpu(images: Iterable[Image.Image | Path], image_size: int) -> np.ndarray:
    arrays = []
    for item in images:
        image = Image.open(item) if isinstance(item, Path) else item
        arrays.append(pil_to_chw_array(image, image_size))
    batch = np.stack(arrays, axis=0)
    return batch.reshape(batch.shape[0], -1)


@torch.no_grad()
def preprocess_gpu(images: Iterable[Image.Image | Path], image_size: int, device: str) -> np.ndarray:
    tensors = []
    for item in images:
        image = Image.open(item) if isinstance(item, Path) else item
        image = image.convert("RGB").resize((image_size, image_size), Image.LANCZOS)
        tensor = torch.from_numpy(np.asarray(image)).to(device=device, dtype=torch.float32)
        tensor = tensor.permute(2, 0, 1) / 127.5 - 1.0
        tensors.append(tensor)
    batch = torch.stack(tensors, dim=0)
    return batch.reshape(batch.shape[0], -1).cpu().numpy()


def batches(items: list[Any], batch_size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_dataset_items(args: argparse.Namespace, label: int | None) -> Iterator[Image.Image]:
    if args.dataset_dir:
        from datasets import load_from_disk

        dataset = load_from_disk(str(args.dataset_dir))
        if hasattr(dataset, "keys") and "train" in dataset.keys():
            dataset = dataset["train"]
    elif args.dataset and args.dataset.startswith("hf:"):
        from datasets import load_dataset

        dataset = load_dataset(args.dataset.removeprefix("hf:"), split=args.split, streaming=True)
    else:
        raise ValueError("Dataset mode requires --dataset hf:<name> or --dataset-dir")

    count = 0
    for item in dataset:
        if label is not None and int(item["label"]) != label:
            continue
        image = item.get("image") or item.get("img")
        if image is None:
            raise ValueError("Dataset rows must contain an image or img field")
        yield image.convert("RGB")
        count += 1
        if args.num_images > 0 and count >= args.num_images:
            break


def fit_incremental_pca(
    batches_iter: Iterable[np.ndarray],
    *,
    components: int,
    batch_size: int,
    desc: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    from sklearn.decomposition import IncrementalPCA

    ipca = IncrementalPCA(n_components=components, batch_size=batch_size)
    mean_accumulator: np.ndarray | None = None
    n_samples = 0

    for batch in tqdm(batches_iter, desc=desc):
        if batch.shape[0] == 0:
            continue
        if batch.shape[0] < components and n_samples == 0:
            raise ValueError(
                f"First PCA batch has {batch.shape[0]} rows but {components} components were requested. "
                "Increase --batch-size or reduce --components."
            )
        ipca.partial_fit(batch)
        batch_sum = batch.sum(axis=0, dtype=np.float64)
        mean_accumulator = batch_sum if mean_accumulator is None else mean_accumulator + batch_sum
        n_samples += batch.shape[0]

    if n_samples < 2:
        raise ValueError("Need at least two images to fit a PCA filter")
    mean = (mean_accumulator / n_samples).astype(np.float32)
    eigenvectors = ipca.components_.T.astype(np.float32)
    eigenvalues = ipca.explained_variance_.astype(np.float32)
    return mean, eigenvalues, eigenvectors, n_samples


def filter_payload(
    *,
    mean: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    image_size: int,
    num_images: int,
) -> dict[str, torch.Tensor | int]:
    return {
        "mean_patch": torch.from_numpy(mean).float(),
        "eigenvectors": torch.from_numpy(eigenvectors).float(),
        "eigenvalues": torch.from_numpy(eigenvalues).float(),
        "img_resolution": image_size,
        "img_channels": 3,
        "num_images": int(num_images),
    }


def save_filter(
    output_dir: Path,
    name: str,
    *,
    mean: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    image_size: int,
    num_images: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"mean_{name}.npy", mean)
    np.save(output_dir / f"eigenvalues_{name}.npy", eigenvalues)
    np.save(output_dir / f"eigenvectors_{name}.npy", eigenvectors)
    save_gaussian_filter(
        filter_payload(
            mean=mean,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            image_size=image_size,
            num_images=num_images,
        ),
        str(output_dir / f"{name}.pt"),
    )


def fit_from_image_dir(args: argparse.Namespace, class_id: str, backend: str):
    class_dir = args.image_root / args.class_dir_template.format(class_id=class_id)
    paths = image_paths(class_dir, args.num_images)
    components = min(args.components, len(paths) - 1, 3 * args.image_size * args.image_size)
    batch_size = max(args.batch_size, components)

    def iter_batches():
        for batch_paths in batches(paths, batch_size):
            if backend == "gpu-preprocess":
                yield preprocess_gpu(batch_paths, args.image_size, args.device)
            else:
                yield preprocess_cpu(batch_paths, args.image_size)

    return fit_incremental_pca(
        iter_batches(),
        components=components,
        batch_size=batch_size,
        desc=f"Fitting PCA for {class_id}",
    )


def fit_full_from_image_root(args: argparse.Namespace, backend: str):
    paths = all_image_paths(args.image_root, args.num_images)
    components = min(args.components, len(paths) - 1, 3 * args.image_size * args.image_size)
    batch_size = max(args.batch_size, components)

    def iter_batches():
        for batch_paths in batches(paths, batch_size):
            if backend == "gpu-preprocess":
                yield preprocess_gpu(batch_paths, args.image_size, args.device)
            else:
                yield preprocess_cpu(batch_paths, args.image_size)

    return fit_incremental_pca(
        iter_batches(),
        components=components,
        batch_size=batch_size,
        desc="Fitting full-dataset PCA",
    )


def fit_from_dataset(args: argparse.Namespace, label: int | None, backend: str):
    requested = args.num_images if args.num_images > 0 else LARGE_DATA_THRESHOLD + 1
    components = min(args.components, requested - 1, 3 * args.image_size * args.image_size)
    batch_size = max(args.batch_size, components)

    def iter_batches():
        pending: list[Image.Image] = []
        for image in load_dataset_items(args, label):
            pending.append(image)
            if len(pending) >= batch_size:
                if backend == "gpu-preprocess":
                    yield preprocess_gpu(pending, args.image_size, args.device)
                else:
                    yield preprocess_cpu(pending, args.image_size)
                pending.clear()
        if pending:
            if backend == "gpu-preprocess":
                yield preprocess_gpu(pending, args.image_size, args.device)
            else:
                yield preprocess_cpu(pending, args.image_size)

    return fit_incremental_pca(
        iter_batches(),
        components=components,
        batch_size=batch_size,
        desc="Fitting PCA",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PCA/Wiener filters")
    parser.add_argument("--image-root", type=Path, help="Root containing class image directories")
    parser.add_argument("--dataset", help="Dataset spec, e.g. hf:imagenet-1k")
    parser.add_argument("--dataset-dir", type=Path, help="Hugging Face dataset saved with load_from_disk")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--classes", nargs="*", default=[], help="Class names or numeric labels")
    parser.add_argument("--full-dataset", action="store_true", help="Build a filter over all selected dataset images")
    parser.add_argument("--class-dir-template", default="label_{class_id}")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-images", type=int, default=1300)
    parser.add_argument("--components", type=int, default=1300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--backend", choices=["auto", "cpu", "gpu-preprocess"], default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    backend = choose_backend(args.backend, args.num_images, args.device)
    print(f"Using PCA backend: {backend}")

    if args.full_dataset:
        if args.image_root:
            mean, eigenvalues, eigenvectors, n_samples = fit_full_from_image_root(args, backend)
        else:
            mean, eigenvalues, eigenvectors, n_samples = fit_from_dataset(args, None, backend)
        save_filter(
            args.output_dir,
            "full_dataset",
            mean=mean,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            image_size=args.image_size,
            num_images=n_samples,
        )

    if not args.classes and not args.full_dataset:
        raise SystemExit("Specify --classes and/or --full-dataset")

    for class_id in args.classes:
        if args.image_root:
            mean, eigenvalues, eigenvectors, n_samples = fit_from_image_dir(args, class_id, backend)
        else:
            mean, eigenvalues, eigenvectors, n_samples = fit_from_dataset(args, int(class_id), backend)
        save_filter(
            args.output_dir,
            f"class_{class_id}",
            mean=mean,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            image_size=args.image_size,
            num_images=n_samples,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
