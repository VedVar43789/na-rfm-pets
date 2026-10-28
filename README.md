# NA-RFM on Oxford-IIIT Pet

We reproduce noise-aligned RFM steering on Oxford-IIIT Pet, guiding a pretrained ImageNet diffusion model toward cat and dog breeds without gradient steps at sampling time.

UCSD DSC 180A, Fall 2026, Halicioglu Data Science Institute (HDSI). Mentor: Prof. Yusu Wang.

## Group

Vedant Vardhaan, Shiv Mehta, and Vishudh Shah.

## What the method does

Classifier guidance can point an unconditional diffusion model at a target class, but every denoising step has to backpropagate through the model and through a classifier. NA-RFM (Noise-Aligned RFM Steering) removes that cost. The method is from "General and Efficient Steering of Unconditional Diffusion" by Qingsong Wang, Mikhail Belkin, and Yusu Wang (arXiv:2602.11395, February 2026).

Two observations make the cheaper procedure possible. Class signal shows up in the U-Net activations very early, while the sample is still mostly noise. A steering direction fit at one timestep stays useful across many other timesteps, so the same vector can be reused at other noise levels.

Denoising is split into two phases. Both phases use statistics computed offline, before any sample is drawn.

In the early phase the noise level is high. Steering is noise alignment. PCA statistics of the target class define a class-conditional Gaussian denoiser, and the update is the difference between that denoiser and an unconditional Gaussian denoiser.

In the later phase the noise level is low. Steering is a single concept direction in the activation space of one U-Net encoder block. Recursive Feature Machines (RFM) learn that direction once. Sampling reuses it across timesteps and applies a classifier-free guidance style scale.

At generation time the sampler only runs forward passes and adds those vector corrections. Accuracies reported for the method are 96.6 percent on CIFAR-10 (77.1 percent for TFG-4), 75.8 percent on ImageNet, 96.0 percent average on CelebA multi-attribute targets, and 14.1 percent on fine-grained bird species (2.2 percent for TFG-4). Inference is roughly 10 times faster than TFG-4.

## What we change

The method stays intact: noise alignment in the high-noise phase, and one reused RFM direction in the low-noise phase. The diffusion model stays the public ImageNet ADM checkpoint. Evaluation stays the protocol from the paper, namely target-class accuracy, FID, and inference cost.

The target data is Oxford-IIIT Pet (Parkhi et al., 2012). The dataset has 37 breeds, 12 cat breeds and 25 dog breeds, with about 200 images each. The paper reports CIFAR-10, ImageNet, CelebA, and Birds-525. We recompute the PCA filters and the RFM directions on Pet images, and we run the same ADM denoiser at 256 pixels.

We add a granularity axis. The coarse task is binary, cat versus dog. The fine task is one breed at a time. The same denoiser has to carry both a species direction and a much narrower breed direction.

Two measurements sit on that split. We fit an RFM direction at one timestep and apply that fixed vector at other timesteps, to see how far cross-timestep transfer extends on Pet. We also treat distribution shift as part of the result. The denoiser was trained on ImageNet, while the target statistics come from Pet. ImageNet contains many dog breeds and fewer cat classes, but the label set, the framing, and the breed coverage differ from Oxford-IIIT Pet.

Those measurements are the Fall 2026 scope. They also set up a later project on the geometry and transport of semantic directions.

## Planned experiments

1. **ImageNet sanity check.** Rerun the published ImageNet ADM setting already wired in this repo (classes 111, 222, 333, and 444) before reporting any Pet number. Pet uses that checkpoint and the same 256-pixel resolution, so this run checks the stack we will reuse. Smoke mode on class 444 is the first check. Full mode is the comparison with the paper.

2. **Binary species steering.** Build PCA filters and an RFM direction for cat versus dog on Oxford-IIIT Pet, then sample with the ImageNet ADM model. Record target accuracy, FID, and inference cost.

3. **Breed steering.** Repeat the procedure for individual breeds. This is the fine-grained setting, in the same spirit as Birds-525 on the ImageNet ADM model.

4. **Cross-timestep transfer.** Fit the RFM direction at one collection timestep and apply that same vector at other timesteps. Record how target accuracy changes as the gap between the fit timestep and the apply timestep grows.

5. **Steering strength sweep.** Vary the noise-alignment weight and the RFM scale (`lambda`, `w_rfm`, and `s` in `configs/camera_ready_repro.json`). Record accuracy and FID at each scale.

## Repository layout

This fork started from the camera-ready NA-RFM code at https://github.com/isotrivial/na-rfm.

Upstream directories and launchers:

- `na_rfm/` holds RFM direction learning, activation collection, DDIM sampling helpers, guidance hooks, and evaluation utilities.
- `pca/` holds PCA and Wiener filter building, denoising, and validation.
- `repro/` holds the dataset runners for CIFAR-10, ImageNet, CelebA-HQ, and Birds-525, plus metric helpers. See `repro/README.md`.
- `configs/camera_ready_repro.json` holds the default reproduction settings.
- `tests/` holds unit tests.
- `cifar10.sh`, `imagenet.sh`, `celeba.sh`, and `birds.sh` are the paper-facing launchers.

The capstone file in the tree today is this `README.md`. Pet experiments will follow the ImageNet pattern:

- a runner under `repro/`
- a top-level launcher in the style of `imagenet.sh`
- config entries beside `imagenet_adm` in `configs/camera_ready_repro.json`
- generated filters and concepts under `repro/assets/pets_filters/`, `repro/assets/pets_binary_filters/`, `repro/assets/pets_concepts/`, and `repro/assets/pets_binary_concepts/`

Checkpoints, filters, and concept files stay on disk locally. The upstream `.gitignore` already excludes them.

## Setup

Clone the group fork, then clone Training-Free Guidance into `baselines/tfg`:

```bash
git clone https://github.com/VedVar43789/na-rfm-pets.git
cd na-rfm-pets
git clone https://github.com/YWolfeee/Training-Free-Guidance.git baselines/tfg
```

The diffusion checkpoints and baseline Training-Free Guidance setup are provided by the TFG project:

```text
https://github.com/YWolfeee/Training-Free-Guidance
```

Place the required checkpoints, PCA/Wiener filters, and RFM concept files in the paths checked by the launchers under `data/`, `checkpoints/`, and `repro/assets/`.

Pet and the ImageNet sanity check both read `checkpoints/openai_imagenet.pt`. CIFAR-10 runs read `data/checkpoints/openai_cifar10.pt`. Download those weights through the TFG setup and copy them onto the paths in the asset layout below.

RFM training uses the `xrfm` package when it is installed (`pip install xrfm`). Otherwise `na_rfm/dual_rfm.py` uses the dual solver shipped in this repo.

Unit tests run from the local tree:

```bash
python -m pytest tests/
```

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

The ImageNet smoke command is also the checkpoint check for Pet, because both read `checkpoints/openai_imagenet.pt`. A `pets.sh` launcher is future work. CelebA expects EDM sources on `third_party/edm` in addition to TFG. ImageNet, Birds, and the planned Pet runs load the OpenAI ADM U-Net through `baselines/tfg`.

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

### Oxford-IIIT Pet filters and concepts

Pet uses the same ADM checkpoint, resolution, and encoder block as ImageNet and Birds: `checkpoints/openai_imagenet.pt`, `image_size` 256, and `block_name="input_blocks_15"`.

Put images in class folders. The default folder name is `label_<id>`. For breeds, use ids 0 through 36. For cat versus dog, use a separate root with `label_0` for cat and `label_1` for dog. When the folders are named by breed, pass `--class-dir-template "{class_id}"` and pass those names to `--classes`.

```bash
python pca/build_filters.py \
  --image-root /path/to/oxford_pets_by_breed \
  --output-dir repro/assets/pets_filters \
  --full-dataset \
  --classes $(seq 0 36) \
  --image-size 256 \
  --num-images 0 \
  --components 1000
```

```bash
python pca/build_filters.py \
  --image-root /path/to/oxford_pets_cat_dog \
  --output-dir repro/assets/pets_binary_filters \
  --full-dataset \
  --classes 0 1 \
  --image-size 256 \
  --num-images 0 \
  --components 1000
```

`--num-images 0` keeps every file. A positive value caps the list. Under `--full-dataset` that cap counts files across the whole tree. On a class directory the same flag caps that directory only.

The saved component count is the minimum of `--components`, one fewer than the number of images, and the flattened image dimension. Each breed folder has about 200 images, so a breed filter lands below 200 components even when `--components` is 1000. The pool of all Pet images is large enough for 1000 components, and so is each side of the cat versus dog split. The Birds-525 config uses 180 components, which is a practical scale for a fine-grained class of this size.

Train Pet concept files with the CIFAR-10 snippet above, using the ImageNet loader settings:

- checkpoint `checkpoints/openai_imagenet.pt`
- `image_size=256`
- `block_name="input_blocks_15"`
- images in `[-1, 1]` with integer Pet labels
- a first collection noise near the Birds setting, `sigma` 0.60

The cross-timestep experiment refits that direction at other timesteps on purpose. Save breed concepts to `repro/assets/pets_concepts/class_<id>.pt` and species concepts to `repro/assets/pets_binary_concepts/class_<id>.pt`. A file may store a ready `direction`, or `eigenvectors`, `eigenvalues`, and `sign`.

## Runner

Render commands without running:

```bash
python repro/run_camera_ready.py --task cifar10 --method na_rfm --targets 5 --mode dry-run
```

## Status

The course checkpoint is due November 6, 2026. The final submission is due December 4, 2026.

The repository is the upstream fork plus this README. Next engineering work is the ImageNet sanity check and the Pet filter build. Runnable launchers today are `cifar10.sh`, `imagenet.sh`, `celeba.sh`, and `birds.sh`.

## Credits

- Upstream code: https://github.com/isotrivial/na-rfm (MIT).
- Paper: Qingsong Wang, Mikhail Belkin, and Yusu Wang, "General and Efficient Steering of Unconditional Diffusion", arXiv:2602.11395, February 2026.
- Diffusion code and checkpoints: Training-Free Guidance, https://github.com/YWolfeee/Training-Free-Guidance.
- Course: UCSD DSC 180A, Fall 2026, HDSI. Mentor: Prof. Yusu Wang.

## License

This code is released under the MIT License inherited from the upstream repository. See `LICENSE`. Copyright in that file is held by the NA-RFM authors. Capstone additions stay under the same license.

## Reproduction scope

`na_rfm/`, `pca/`, `repro/`, `configs/`, `tests/`, `cifar10.sh`, `imagenet.sh`, `celeba.sh`, `birds.sh`, and `LICENSE` are upstream files from https://github.com/isotrivial/na-rfm and are unchanged in this fork. The capstone-specific file is `README.md`. Oxford-IIIT Pet runners, config entries, and the `pets_` asset directories listed above will be our additions when we add them. The official release covers CIFAR-10, ImageNet, CelebA, and Birds-525 only. A grader can separate the two by that split: every path other than this README is upstream at the time of writing, and every Pet experiment file added later is ours.
