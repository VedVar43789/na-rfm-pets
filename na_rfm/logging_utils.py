"""Structured logging utilities for diffusion steering experiments.

Provides console + file + JSON logging with automatic metric tracking.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentLogger:
    """Structured logger with console, file, and JSON outputs."""

    def __init__(self, output_dir: Path, name: str = "experiment"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.start_time = datetime.now()

        # Setup Python logger
        self.logger = logging.getLogger(f"{name}_{id(self)}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        self.logger.propagate = False

        # Console handler: INFO level, concise format
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        self.logger.addHandler(console)

        # File handler: DEBUG level, detailed with timestamps
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"{name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        self.logger.addHandler(file_handler)

        self._log_file = log_file

    @property
    def log_file(self) -> Path:
        return self._log_file

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def log_stage(self, stage: str) -> None:
        """Log stage start with visual separator."""
        sep = "=" * 50
        self.logger.info(f"\n{sep}")
        self.logger.info(f"STAGE: {stage.upper()}")
        self.logger.info(sep)

    def log_metric(
        self, key: str, value: Any, category: str = "general"
    ) -> None:
        """Log and store a metric value."""
        if category not in self.metrics:
            self.metrics[category] = {}
        self.metrics[category][key] = _to_jsonable(value)
        self.logger.info(f"[METRIC] {category}.{key} = {value}")

    def log_config(self, config: Any, name: str = "config") -> None:
        """Log config summary."""
        self.debug(f"Configuration ({name}):")
        if hasattr(config, "__dataclass_fields__"):
            from dataclasses import asdict

            for k, v in asdict(config).items():
                self.debug(f"  {k}: {v}")
        elif isinstance(config, dict):
            for k, v in config.items():
                self.debug(f"  {k}: {v}")

    def save_metrics(self, filename: str = "metrics.json") -> Path:
        """Save all accumulated metrics to JSON file."""
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2, default=str)
        self.debug(f"Saved metrics: {path}")
        return path

    def finalize(self) -> Dict[str, Any]:
        """Save final metrics and return elapsed time."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.log_metric("total_time_sec", elapsed, "performance")
        self.save_metrics()
        self.info(f"Experiment completed in {elapsed:.1f}s")
        return {"total_time_sec": elapsed, "metrics": self.metrics}

    def close(self) -> None:
        """Close all handlers."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


def _to_jsonable(obj: Any) -> Any:
    """Convert object to JSON-serializable form."""
    if obj is None:
        return None
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    # Handle torch tensors and numpy arrays
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


class StageTimer:
    """Context manager for timing stages."""

    def __init__(self, logger: ExperimentLogger, stage_name: str, category: str = "performance"):
        self.logger = logger
        self.stage_name = stage_name
        self.category = category
        self.start: Optional[datetime] = None

    def __enter__(self) -> "StageTimer":
        self.logger.log_stage(self.stage_name)
        self.start = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start is not None:
            elapsed = (datetime.now() - self.start).total_seconds()
            metric_key = f"{self.stage_name.lower().replace(' ', '_')}_time_sec"
            self.logger.log_metric(metric_key, elapsed, self.category)
