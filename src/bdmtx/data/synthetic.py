"""Synthetic Data Generator for DPM images (basic implementation).

This module creates paired samples: a clean DataMatrix rendered on a simple background
and a degraded version with blur, embossing-like effects, noise and lighting gradients.

This is a conservative starter implementation to produce training samples for Phase 1.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np


def render_datamatrix(size: int = 64) -> np.ndarray:
    """Render a simple binary DataMatrix-like square (placeholder).

    Currently generates a checkerboard with a finder L-pattern to approximate a
    DataMatrix symbol for synthetic experiments. Replace with actual encoder later.
    """
    symbol = np.zeros((size, size), dtype=np.uint8)
    # Finder L: left column and bottom row
    symbol[:, 0:2] = 255
    symbol[-2:, :] = 255
    # Fill with pseudo-random modules
    rng = np.random.default_rng()
    modules = rng.integers(0, 2, size=(size // 4, size // 4))
    modules = cv2.resize(modules.astype(np.uint8) * 255, (size - 6, size - 6), interpolation=cv2.INTER_NEAREST)
    symbol[3:-3, 3:-3] = modules
    return symbol


def overlay_on_texture(symbol: np.ndarray, texture: np.ndarray, x: int | None = None, y: int | None = None) -> np.ndarray:
    """Overlay the symbol onto a background texture with slight blending.
    If x,y provided, place symbol at those coordinates; otherwise choose random placement.
    """
    h, w = texture.shape[:2]
    sh, sw = symbol.shape
    if y is None:
        y = (h - sh) // 2 + random.randint(-10, 10)
    if x is None:
        x = (w - sw) // 2 + random.randint(-10, 10)
    out = texture.copy().astype(np.float32) / 255.0
    sym = np.stack([symbol / 255.0] * 3, axis=-1)
    # Emulate engraving by darkening the symbol area and adding specular
    mask = sym[..., 0] > 0.5
    out[y : y + sh, x : x + sw][mask] *= 0.3 + 0.2 * random.random()
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


def random_texture(size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Generate a random grayish metal/wax-like texture using noise and blur."""
    h, w = size
    base = np.random.normal(loc=128, scale=20, size=(h, w)).astype(np.uint8)
    for _ in range(3):
        k = random.choice([3, 5, 7])
        base = cv2.GaussianBlur(base, (k, k), 0)
    # subtle directional gradient
    yy = np.linspace(0, 1, h)[:, None]
    grad = (yy * 30).astype(np.uint8)
    tex = np.clip(base + grad, 0, 255).astype(np.uint8)
    return cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)


def degrade_image(image: np.ndarray) -> np.ndarray:
    """Apply a sequence of degradations to emulate DPM defects."""
    out = image.copy()
    # convert to gray then back to 3-channel for simplicity
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    # motion blur
    if random.random() < 0.6:
        k = random.choice([3, 5, 7, 9])
        kernel = np.zeros((k, k))
        kernel[k // 2, :] = np.ones(k)
        kernel = kernel / kernel.sum()
        gray = cv2.filter2D(gray, -1, kernel)
    # gaussian blur
    if random.random() < 0.8:
        sigma = random.uniform(0.5, 2.0)
        gray = cv2.GaussianBlur(gray, (0, 0), sigma)
    # add speckle noise
    if random.random() < 0.7:
        noise = np.random.normal(0, random.uniform(5, 25), gray.shape).astype(np.int16)
        gray = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # contrast/brightness
    alpha = random.uniform(0.6, 1.2)
    beta = random.uniform(-30, 30)
    gray = np.clip(alpha * gray + beta, 0, 255).astype(np.uint8)
    # emboss-like effect
    if random.random() < 0.5:
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
        gray = cv2.filter2D(gray, -1, kernel) + 128
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    # return 3-channel
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def generate_pair(out_dir: Path, index: int, size: tuple[int, int] = (256, 256)) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    texture = random_texture(size)
    symbol = render_datamatrix(size=64)
    sh, sw = symbol.shape

    h, w = size
    # place symbol roughly centered (deterministic offset used for annotation)
    y = (h - sh) // 2 + random.randint(-10, 10)
    x = (w - sw) // 2 + random.randint(-10, 10)

    clean = overlay_on_texture(symbol, texture, x=x, y=y)
    degraded = degrade_image(clean)

    clean_path = out_dir / f"clean_{index:06d}.png"
    degraded_path = out_dir / f"degraded_{index:06d}.png"
    cv2.imwrite(str(clean_path), clean)
    cv2.imwrite(str(degraded_path), degraded)

    # COCO-style annotation (single annotation per image)
    poly = [float(x), float(y), float(x + sw), float(y), float(x + sw), float(y + sh), float(x), float(y + sh)]
    annotation = {
        "id": index,
        "image_id": index,
        "category_id": 1,
        "bbox": [int(x), int(y), int(sw), int(sh)],
        "segmentation": [poly],
        "area": int(sw * sh),
        "attributes": {"content": f"dm_{index:06d}", "type": "datamatrix"},
    }

    ann_common = {
        "categories": [{"id": 1, "name": "datamatrix"}],
        "annotations": [annotation],
    }

    import json

    ann_clean = {
        "images": [{"id": index, "width": w, "height": h, "file_name": clean_path.name}],
        **ann_common,
    }
    ann_degraded = {
        "images": [{"id": index, "width": w, "height": h, "file_name": degraded_path.name}],
        **ann_common,
    }

    ann_path_clean = out_dir / f"{clean_path.stem}.json"
    ann_path_degraded = out_dir / f"{degraded_path.stem}.json"

    with open(ann_path_clean, "w", encoding="utf-8") as fh:
        json.dump(ann_clean, fh, ensure_ascii=False)

    with open(ann_path_degraded, "w", encoding="utf-8") as fh:
        json.dump(ann_degraded, fh, ensure_ascii=False)


def create_dataset(root: Path, n: int = 1000, size: tuple[int, int] = (256, 256)) -> None:
    """Create n paired samples under root/clean and root/degraded."""
    out = root
    for i in range(n):
        generate_pair(out, i, size=size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic DPM dataset (basic)")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()
    create_dataset(args.out_dir, n=args.n)
