# NA-RFM Diffusion Guidance

This repository contains the camera-ready reproduction code for noise alignment with RFM activation guidance. It provides:

- `na_rfm/`: RFM direction learning, activation collection, DDIM sampling helpers, guidance hooks, and evaluation utilities.
- `pca/`: PCA/Wiener filter loading, denoising, validation, and filter-building utilities.
- `repro/`: dataset-specific Python runners and metric evaluation helpers.
- `cifar10.sh`, `imagenet.sh`, `celeba.sh`, `birds.sh`: paper-facing experiment launchers.

The diffusion checkpoints and baseline Training-Free Guidance setup are provided by the TFG project:

```text
https://github.com/YWolfeee/Training-Free-Guidance
```

Place the required checkpoints, PCA/Wiener filters, and RFM concept files in the paths checked by the launchers under `data/`, `checkpoints/`, and `repro/assets/`.

## Quick Start

Install this repository together with the TFG dependencies and make the TFG source importable, for example:

```bash
export PYTHONPATH=".:third_party/edm:baselines/tfg:${PYTHONPATH:-}"
```

Run smoke checks:

```bash
MODE=smoke TARGETS=5 bash cifar10.sh
MODE=smoke TARGETS=444 bash imagenet.sh
MODE=smoke COMBOS=male_blond bash celeba.sh
MODE=smoke TARGETS=444 bash birds.sh
```

Run the full camera-ready protocols:

```bash
MODE=full bash cifar10.sh
MODE=full bash imagenet.sh
MODE=full bash celeba.sh
MODE=full bash birds.sh
```

Full mode uses 2048 samples per CIFAR-10 class and 256 samples per ImageNet, CelebA-HQ, and Birds-525 target. Outputs are written to `results/camera_ready_repro` by default.

## Assets

The launchers expect assets in this layout:

```text
data/checkpoints/openai_cifar10.pt
checkpoints/openai_imagenet.pt
repro/assets/cifar10_filters/
repro/assets/cifar10_concepts/
repro/assets/imagenet_filters/
repro/assets/imagenet_concepts/
repro/assets/celeba_filters/
repro/assets/celeba_concepts/
repro/assets/birds_filters/
repro/assets/birds_concepts/
```

These generated assets are intentionally not committed.

## Preparing PCA and RFM Assets

After placing data on disk, build the PCA/Wiener filters used by noise alignment. For local image folders, use a layout such as:

```text
/path/to/cifar10_train_by_class/
  label_0/
  label_1/
  ...
```

Then run:

```bash
python pca/build_filters.py \
  --image-root /path/to/cifar10_train_by_class \
  --output-dir repro/assets/cifar10_filters \
  --full-dataset \
  --classes 0 1 2 3 4 5 6 7 8 9 \
  --image-size 32 \
  --num-images 5000 \
  --components 1000
```

This writes `full_dataset.pt` and `class_<id>.pt` filters under the output directory. If your folders are named by class id instead of `label_<id>`, add `--class-dir-template "{class_id}"`.

The same builder can read a Hugging Face dataset or a dataset saved with `load_from_disk`:

```bash
python pca/build_filters.py \
  --dataset hf:imagenet-1k \
  --split train \
  --output-dir repro/assets/imagenet_filters \
  --full-dataset \
  --classes 111 222 333 444 \
  --image-size 256 \
  --num-images 1300 \
  --components 1000
```

For large datasets, `pca/build_filters.py --backend auto` uses CUDA preprocessing when more than 100,000 images are requested and CUDA is available. Incremental PCA itself remains CPU-backed through scikit-learn, while image decode, resize, normalization, and flattening can be GPU-assisted.

RFM concept files are trained from noised model activations. The training data loader should yield `(images, labels)` with images normalized to `[-1, 1]` and integer labels. A minimal CIFAR-10 example is:

```python
from pathlib import Path
import torch

from na_rfm.activation_collection import collect_activations
from na_rfm.model_loaders import load_openai_unet
from na_rfm.rfm_training import RFMTrainConfig, get_direction, train_binary_rfm

device = "cuda"
target_class = 5
sigma = 0.21
alpha_prod_t = 1.0 / (1.0 + sigma**2)

# dataloader must yield images in [-1, 1] and integer labels.
unet = load_openai_unet("data/checkpoints/openai_cifar10.pt", image_size=32, device=device)
activations, labels = collect_activations(
    unet,
    dataloader,
    block_name="input_blocks_9",
    alpha_prod_t=alpha_prod_t,
    timestep=60,
    device=device,
)

config = RFMTrainConfig(
    bandwidth=100.0,
    reg=1e-3,
    top_k=3,
    device=device,
)
result = train_binary_rfm(activations, labels, target_value=target_class, config=config)

concept = {
    "direction": get_direction(result, topk=3).cpu(),
    "eigenvectors": result.eigenvectors.cpu(),
    "eigenvalues": result.eigenvalues.cpu(),
    "sign": int(result.sign),
    "auc": float(result.best_auc),
    "block_name": "input_blocks_9",
    "timestep": 60,
    "sigma": sigma,
}
Path("repro/assets/cifar10_concepts").mkdir(parents=True, exist_ok=True)
torch.save(concept, f"repro/assets/cifar10_concepts/class_{target_class}.pt")
```

The runners accept concept files containing either a ready-to-use `direction` or `eigenvectors`, `eigenvalues`, and `sign`. Preferred filenames are `class_<id>.pt` for CIFAR-10, `direction_label<label>.pt` for ImageNet, and `label_<id>.pt` for Birds-525. For ImageNet and Birds, use the ADM ImageNet checkpoint, `image_size=256`, `block_name="input_blocks_15"`, and place concept files under `repro/assets/imagenet_concepts/` or `repro/assets/birds_concepts/`. For CelebA-HQ, use the DDPM CelebA-HQ model, `mid_block` concepts, and place files under `repro/assets/celeba_concepts/`.

## Runner

Render commands without running:

```bash
python repro/run_camera_ready.py --task cifar10 --method na_rfm --targets 5 --mode dry-run
```

## License

This code is released under the MIT License. See `LICENSE`.
