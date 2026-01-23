"""Training entrypoints for segmentation and enhancement models."""

from __future__ import annotations

from pathlib import Path


def train_segmentation_entry(dataset_root: str, cfg_out: str | None = None) -> None:
    """CLI wrapper to prepare dataset and invoke segmentation training helper."""
    from bdmtx.segmentation.train_yoloseg import make_ultralytics_dataset_yaml, train_ultralytics

    ds = Path(dataset_root)
    cfg = Path(cfg_out) if cfg_out else ds / "dataset.yaml"
    make_ultralytics_dataset_yaml(ds, cfg)
    train_ultralytics(cfg)


def train_enhancement_entry(dataset_root: str) -> None:
    from bdmtx.enhancement.train_enhancer import train_enhancer

    train_enhancer(Path(dataset_root))
