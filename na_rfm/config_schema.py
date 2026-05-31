"""Structured configuration schema for diffusion steering experiments.

Provides dataclasses for each config section with defaults derived from
existing scripts. Supports nested CLI args like --sampling.seed and
JSON serialization.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ConceptConfig:
    """Activation collection settings."""

    collect_ddpm_t: int = 930
    num_per_class: int = 1000
    collect_batch_size: int = 128
    block_name: str = "input_blocks_9"
    download: bool = False


@dataclass
class SamplingConfig:
    """DDIM sampling settings."""

    num_samples: int = 128
    inference_steps: int = 50
    eta: float = 0.0
    clip_x0: bool = True
    cfg_scale: float = 3.0
    seed: int = 42
    # Noise editing (SLERP initial noise toward class mean)
    noise_edit_strength: float = 0.0  # 0.0 = disabled, typical: 0.002-0.005
    noise_edit_n_images: int = 200  # Number of target images for noise direction


@dataclass
class RFMGuidanceConfig:
    """RFM-based guidance settings."""

    guidance_strength: float = 0.235
    guide_start: int = 0
    guide_end: int = 999
    guide_sigma_start: Optional[float] = None
    guide_sigma_end: Optional[float] = None
    bandwidth: float = 100.0
    reg: float = 1e-3
    center_grads: bool = False
    topk: int = 3


@dataclass
class GaussianGuidanceConfig:
    """Gaussian epsilon guidance settings."""

    filters_dir: Optional[str] = None
    mode: str = "full_diff"  # full_diff | class_only
    strength: float = 0.0
    start: int = 0
    end: int = 999
    score_scaling: bool = False
    components: Optional[int] = None


@dataclass
class EvalConfig:
    """Evaluation settings."""

    batch_size: int = 64
    device: str = "cuda"
    # TFG-compatible evaluation networks
    guide_network: Optional[str] = None  # For guidance (differs from eval)
    eval_network: Optional[str] = None  # For evaluation (differs from guidance)


@dataclass
class PathsConfig:
    """File and directory paths."""

    model_path: str = "openai_cifar10.pt"
    data_root: str = "data/cifar10"
    output_dir: str = "results/cifar10"
    runs_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    concepts_dir: Optional[str] = None
    concepts_root: str = "repro/assets/cifar10_concepts"


@dataclass
class PerformanceMetrics:
    """Runtime metrics (populated during run)."""

    collection_time_sec: Optional[float] = None
    training_time_sec: Optional[float] = None
    sampling_time_sec: Optional[float] = None
    eval_time_sec: Optional[float] = None
    total_time_sec: Optional[float] = None
    samples_per_second: Optional[float] = None


@dataclass
class EvalMetrics:
    """Evaluation metrics (populated after eval stage)."""

    # Classifier accuracy (TFG calls this "validity")
    classifier_accuracy: Optional[float] = None
    classifier_name: Optional[str] = None

    # Image quality - CIFAR-10/ImageNet
    fid: Optional[float] = None
    inception_score: Optional[float] = None

    # Image quality - CelebA-HQ
    kid: Optional[float] = None
    log_kid: Optional[float] = None

    # Per-class breakdown
    per_class_accuracy: Optional[Dict[str, float]] = None
    per_class_fid: Optional[Dict[str, float]] = None


@dataclass
class ExperimentConfig:
    """Top-level config combining all sections."""

    concept: ConceptConfig = field(default_factory=ConceptConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    rfm_guidance: RFMGuidanceConfig = field(default_factory=RFMGuidanceConfig)
    gaussian_guidance: GaussianGuidanceConfig = field(default_factory=GaussianGuidanceConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    metrics: EvalMetrics = field(default_factory=EvalMetrics)

    # Run settings (top-level)
    stage: str = "sample"
    classes: str = "all"
    target_class: Optional[int] = None
    run_name: Optional[str] = None
    reuse_concepts: bool = False
    log_summary: bool = False


def _get_field_type(field_type):
    """Extract base type from Optional[T] or other type annotations."""
    # Handle string annotations (from __future__ annotations)
    if isinstance(field_type, str):
        if field_type == "int" or field_type.endswith("[int]"):
            return int
        if field_type == "float" or field_type.endswith("[float]"):
            return float
        if field_type == "bool" or field_type.endswith("[bool]"):
            return bool
        if field_type == "str" or field_type.endswith("[str]"):
            return str
        if "Optional[int]" in field_type:
            return int
        if "Optional[float]" in field_type:
            return float
        if "Optional[str]" in field_type:
            return str
        return str

    origin = getattr(field_type, "__origin__", None)
    if origin is type(None):
        return str
    if origin is not None:
        args = getattr(field_type, "__args__", ())
        for arg in args:
            if arg is not type(None):
                return arg
    return field_type


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add nested CLI args to parser (e.g., --sampling.seed, --rfm_guidance.strength)."""
    config = ExperimentConfig()
    section_classes = {
        "concept": ConceptConfig,
        "sampling": SamplingConfig,
        "rfm_guidance": RFMGuidanceConfig,
        "gaussian_guidance": GaussianGuidanceConfig,
        "eval": EvalConfig,
        "paths": PathsConfig,
    }

    for section_name, section_cls in section_classes.items():
        section_obj = getattr(config, section_name)
        for f in fields(section_cls):
            arg_name = f"--{section_name}.{f.name}"
            default_val = getattr(section_obj, f.name)
            field_type = _get_field_type(f.type)

            if field_type is bool:
                parser.add_argument(arg_name, type=_str_to_bool, default=default_val)
            elif field_type in (int, float, str):
                parser.add_argument(arg_name, type=field_type, default=default_val)
            else:
                parser.add_argument(arg_name, default=default_val)

    # Top-level args
    parser.add_argument("--stage", default=config.stage)
    parser.add_argument("--classes", default=config.classes)
    parser.add_argument("--target_class", type=int, default=config.target_class)
    parser.add_argument("--run_name", default=config.run_name)
    parser.add_argument("--reuse_concepts", type=_str_to_bool, default=config.reuse_concepts)
    parser.add_argument("--log_summary", type=_str_to_bool, default=config.log_summary)


def _str_to_bool(v: str) -> bool:
    """Parse boolean from CLI string."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}")


def _convert_value(val: Any, field_type) -> Any:
    """Convert a value to the appropriate type based on field annotation."""
    if val is None:
        return None

    target_type = _get_field_type(field_type)

    # Already correct type
    if isinstance(val, target_type) and not isinstance(val, bool):
        return val

    # Convert strings to appropriate types
    if isinstance(val, str):
        if target_type is bool:
            return _str_to_bool(val)
        if target_type is int:
            return int(val)
        if target_type is float:
            return float(val)
    return val


def apply_cli_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    """Apply CLI overrides to config object."""
    section_names = ["concept", "sampling", "rfm_guidance", "gaussian_guidance", "eval", "paths"]

    for section_name in section_names:
        section_obj = getattr(config, section_name)
        for f in fields(type(section_obj)):
            cli_key = f"{section_name}.{f.name}"
            if hasattr(args, cli_key):
                val = getattr(args, cli_key)
                if val is not None:
                    converted = _convert_value(val, f.type)
                    setattr(section_obj, f.name, converted)

    # Top-level overrides
    top_level = ["stage", "classes", "target_class", "run_name", "reuse_concepts", "log_summary"]
    for key in top_level:
        if hasattr(args, key):
            val = getattr(args, key)
            if val is not None:
                setattr(config, key, val)


def config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
    """Convert config to nested dict for JSON serialization."""

    def _convert(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _convert(v) for k, v in asdict(obj).items()}
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    return _convert(config)


def save_config(path: Path, config: ExperimentConfig) -> None:
    """Save config to JSON file with organized sections."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(config_to_dict(config), f, indent=2, sort_keys=False)


def load_config(path: Path) -> ExperimentConfig:
    """Load config from JSON file."""
    path = Path(path)
    with path.open("r") as f:
        data = json.load(f)

    config = ExperimentConfig()
    section_classes = {
        "concept": ConceptConfig,
        "sampling": SamplingConfig,
        "rfm_guidance": RFMGuidanceConfig,
        "gaussian_guidance": GaussianGuidanceConfig,
        "eval": EvalConfig,
        "paths": PathsConfig,
        "performance": PerformanceMetrics,
        "metrics": EvalMetrics,
    }

    for section_name, section_cls in section_classes.items():
        if section_name in data:
            section_data = data[section_name]
            section_obj = getattr(config, section_name)
            for key, val in section_data.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, val)

    # Top-level fields
    top_level = ["stage", "classes", "target_class", "run_name", "reuse_concepts", "log_summary"]
    for key in top_level:
        if key in data:
            setattr(config, key, data[key])

    return config


def config_to_flat_dict(config: ExperimentConfig) -> Dict[str, Any]:
    """Flatten config to single-level dict for backward compatibility."""
    flat = {}
    nested = config_to_dict(config)

    def _flatten(d: dict, prefix: str = ""):
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(v, f"{key}_")
            else:
                flat[key] = v

    _flatten(nested)
    return flat
