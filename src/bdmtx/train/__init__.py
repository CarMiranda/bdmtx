"""Training entrypoints for segmentation and enhancement models."""

from __future__ import annotations

from pathlib import Path

from bdmtx.enhancement.train_enhancer import train_enhancer
from bdmtx.segmentation.train_yoloseg import (
    make_ultralytics_dataset_yaml,
    train_ultralytics,
)


def train_segmentation_entry(dataset_root: str, cfg_out: str | None = None) -> None:
    """CLI wrapper to prepare dataset and invoke segmentation training helper."""
    ds = Path(dataset_root)
    cfg = Path(cfg_out) if cfg_out else ds / "dataset.yaml"
    make_ultralytics_dataset_yaml(ds, cfg)
    train_ultralytics(cfg)


def train_enhancement_entry(dataset_root: str) -> None:
    """CLI wrapper to prepare dataset and invoke enhancement training helper."""
    train_enhancer(Path(dataset_root))
