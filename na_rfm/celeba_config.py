"""
Configuration for CelebA binary RFM training.

This module trains 4 separate binary RFM classifiers:
1. male_vs_female - Gender classification
2. young_vs_other - Young (age < 20) vs other
3. old_vs_other - Old (age >= 50) vs other
4. blond_vs_other - Blond hair vs other
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class RFMTrainConfig:
    """Configuration for RFM training."""
    bandwidth: float = 100.0
    reg: float = 1e-3
    center_grads: bool = False
    center_data: bool = True
    normalize: bool = True
    n_iters: int = 5
    top_k: Optional[int] = 5
    early_stop: bool = True
    early_stop_threshold: float = 1.1
    device: str = "auto"
    verbose: bool = False


# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DIFF_GUIDANCE_DATA", REPO_ROOT / "data"))
DATA_DIR = Path(os.environ.get("CELEBA_HQ_DIR", DATA_ROOT / "celeba_hq_256_30k"))
LABELS_PATH = Path(os.environ.get(
    "CELEBA_LABELS_PATH",
    REPO_ROOT / "repro" / "assets" / "celeba_filters" / "labels_with_hair.pt",
))
ACTIVATIONS_DIR = Path(os.environ.get(
    "CELEBA_RFM_ACTIVATIONS_DIR",
    REPO_ROOT / "repro" / "assets" / "celeba_activations",
))
CONCEPTS_DIR = Path(os.environ.get(
    "CELEBA_RFM_CONCEPTS_DIR",
    REPO_ROOT / "repro" / "assets" / "celeba_concepts",
))

# Model
MODEL_ID = "google/ddpm-ema-celebahq-256"

# Activation collection
BLOCK_NAME = "mid_block"
TIMESTEP = 250
BATCH_SIZE = 16

# RFM training config
RFM_CONFIG = RFMTrainConfig(
    bandwidth=100.0,
    reg=1e-3,
    center_grads=False,  # NA-RFM default
    center_data=True,
    normalize=True,
    n_iters=5,
    top_k=5,
)

# Binary tasks to train
BINARY_TASKS = {
    "male_vs_female": {
        "label_field": "gender",
        "target_value": 1,
        "description": "Male (1) vs Female (0)",
    },
    "young_vs_other": {
        "label_field": "young_vs_other",
        "target_value": 1,
        "description": "Young (1) vs Other (0)",
    },
    "old_vs_other": {
        "label_field": "old_vs_other",
        "target_value": 1,
        "description": "Old (1) vs Other (0)",
    },
    "blond_vs_other": {
        "label_field": "is_blond_hair",
        "target_value": 1,
        "description": "Blond (1) vs Other (0)",
    },
}
