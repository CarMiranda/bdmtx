"""Dataset utilities for synthetic DPM dataset and simple augmentations.

Provides a lightweight, framework-agnostic dataset reader that loads
paired clean/degraded images and per-image COCO-style JSON annotations
(created by the synthetic generator). Also provides simple augmentations
and validation helpers.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np


def build_mask_from_segmentation(
    segmentation: list[float] | list[list[float]], width: int, height: int
) -> np.ndarray:
    """Build a binary mask (uint8 0/255) from a COCO-style segmentation.

    segmentation may be either a single flat list [x1,y1,x2,y2,...] or a
    list of lists for multiple polygons.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if not segmentation:
        return mask

    # Normalize to list of polygons
    polys: list[list[float]]
    if isinstance(segmentation[0], (int, float)):
        polys = [segmentation]  # type: ignore[assignment]
    else:
        polys = segmentation  # type: ignore[assignment]

    for poly in polys:
        coords = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        pts = np.round(coords).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


class SyntheticDataset:
    """Lightweight dataset for the synthetic generator output.

    The dataset expects files written directly under `root` with names like
    `clean_000001.png`, `degraded_000001.png`, and `clean_000001.json` /
    `degraded_000001.json` (one JSON per image). The JSON should follow the
    minimal COCO-like structure produced by the generator.
    """

    def __init__(
        self, root: Path, split: str = "degraded", transforms: Callable | None = None
    ):
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.samples: list[dict] = []

        pattern = "degraded_*.png" if split == "degraded" else "clean_*.png"
        for p in sorted(self.root.glob(pattern)):
            idx = p.stem.split("_")[-1]
            sample = {
                "clean": (self.root / f"clean_{idx}.png")
                if (self.root / f"clean_{idx}.png").exists()
                else None,
                "degraded": (self.root / f"degraded_{idx}.png")
                if (self.root / f"degraded_{idx}.png").exists()
                else None,
                "image": p,
                "annotation": (self.root / f"{p.stem}.json")
                if (self.root / f"{p.stem}.json").exists()
                else None,
            }
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        img_path = s["image"]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        h, w = img.shape[:2]

        annotation: dict = {}
        mask = np.zeros((h, w), dtype=np.uint8)

        if s["annotation"] and s["annotation"].exists():
            with open(s["annotation"], encoding="utf-8") as fh:
                annotation = json.load(fh)
            anns = annotation.get("annotations", [])
            if anns:
                seg = anns[0].get("segmentation", [])
                if seg:
                    # seg may be either [[x1,y1,...]] or [x1,y1,...]
                    poly = seg[0] if isinstance(seg[0], list) else seg
                    mask = build_mask_from_segmentation(poly, w, h)

        sample = {"image": img, "mask": mask, "annotation": annotation}
        if s["clean"] and s["clean"].exists():
            sample["clean"] = cv2.imread(str(s["clean"]), cv2.IMREAD_COLOR)
        if s["degraded"] and s["degraded"].exists():
            sample["degraded"] = cv2.imread(str(s["degraded"]), cv2.IMREAD_COLOR)

        if self.transforms:
            sample = self.transforms(sample)
        return sample


def simple_augmentation(sample: dict) -> dict:
    """Apply simple, fast augmentations (in-place) to a sample.

    - Random horizontal flip
    - Random small rotation (-10..10 degrees)
    - Random brightness/contrast on the input image
    """
    if np.random.rand() < 0.5:
        for k in ("image", "clean", "degraded"):
            if k in sample and sample[k] is not None:
                sample[k] = np.ascontiguousarray(np.fliplr(sample[k]))
        sample["mask"] = np.ascontiguousarray(np.fliplr(sample["mask"]))

    if np.random.rand() < 0.5:
        angle = float(np.random.uniform(-10, 10))
        for k in ("image", "clean", "degraded", "mask"):
            if k in sample and sample[k] is not None:
                img = sample[k]
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
                border = cv2.BORDER_REFLECT
                interp = cv2.INTER_LINEAR if img.ndim == 3 else cv2.INTER_NEAREST
                sample[k] = cv2.warpAffine(
                    img, M, (w, h), flags=interp, borderMode=border
                )

    if "image" in sample and sample["image"] is not None:
        alpha = float(np.random.uniform(0.9, 1.1))
        beta = float(np.random.uniform(-10, 10))
        sample["image"] = np.clip(
            alpha * sample["image"].astype(np.float32) + beta, 0, 255
        ).astype(np.uint8)

    return sample


def validate_dataset(root: Path) -> list[str]:
    """Validate dataset files and annotations; return list of human-readable errors.

    Checks performed:
    - presence of at least one clean/degraded image
    - presence of corresponding per-image JSON annotation
    - segmentation exists and contains >= 3 points
    - segmentation coordinates lie inside image bounds
    - segmentation area > 0
    """
    root = Path(root)
    errors: list[str] = []

    imgs = sorted(root.glob("clean_*.png")) + sorted(root.glob("degraded_*.png"))
    if not imgs:
        errors.append("No images found (clean_*.png or degraded_*.png)")
        return errors

    seen = set()
    for p in imgs:
        idx = p.stem.split("_")[-1]
        if idx in seen:
            continue
        seen.add(idx)

        clean_p = root / f"clean_{idx}.png"
        degraded_p = root / f"degraded_{idx}.png"
        ann_p_degraded = root / f"degraded_{idx}.json"
        ann_p_clean = root / f"clean_{idx}.json"
        ann_p = (
            ann_p_degraded
            if ann_p_degraded.exists()
            else (ann_p_clean if ann_p_clean.exists() else None)
        )

        if not (clean_p.exists() or degraded_p.exists()):
            errors.append(f"Missing both clean and degraded images for idx {idx}")
            continue
        if ann_p is None:
            errors.append(f"Missing JSON annotation for idx {idx}")
            continue

        try:
            with open(ann_p, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:  # pragma: no cover - robust reading
            errors.append(f"Failed to read JSON for idx {idx}: {exc}")
            continue

        anns = data.get("annotations", [])
        if not anns:
            errors.append(f"No annotations array for idx {idx}")
            continue
        ann = anns[0]
        seg = ann.get("segmentation", [])
        if not seg:
            errors.append(f"Empty segmentation for idx {idx}")
            continue
        poly = seg[0] if isinstance(seg[0], list) else seg
        if len(poly) < 6:
            errors.append(f"Segmentation has fewer than 3 points for idx {idx}")
            continue

        # load an image to get bounds
        img_path = clean_p if clean_p.exists() else degraded_p
        img = cv2.imread(str(img_path))
        if img is None:
            errors.append(f"Failed to read image for idx {idx}")
            continue
        h, w = img.shape[:2]

        coords = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if (
            np.any(coords[:, 0] < 0)
            or np.any(coords[:, 1] < 0)
            or np.any(coords[:, 0] > w)
            or np.any(coords[:, 1] > h)
        ):
            errors.append(
                f"Segmentation coordinates outside image bounds for idx {idx}"
            )
            continue

        # polygon area (shoelace)
        area = abs(
            np.sum(
                coords[:, 0] * np.roll(coords[:, 1], -1)
                - coords[:, 1] * np.roll(coords[:, 0], -1)
            )
            / 2.0
        )
        if area <= 0:
            errors.append(f"Segmentation area is zero for idx {idx}")

    return errors
