# Reproduction Runners

The top-level launchers are the public entry points:

```bash
bash cifar10.sh
bash imagenet.sh
bash celeba.sh
bash birds.sh
```

This directory contains the Python runners used by those shell scripts plus metric evaluation helpers. The manifest `configs/camera_ready_repro.json` records the default reproduction settings used by the wrapper.

Full-mode sample counts follow the paper protocol: CIFAR-10 uses 2048 samples per class, while ImageNet, CelebA-HQ, and Birds-525 use 256 samples per target or combo.

## Wrapper

Render commands without running:

```bash
python repro/run_camera_ready.py --task cifar10 --method na_rfm --targets 5 --mode dry-run
```

Run a small CIFAR-10 smoke test:

```bash
python repro/run_camera_ready.py --task cifar10 --method na_rfm --targets 5 --mode smoke
```
