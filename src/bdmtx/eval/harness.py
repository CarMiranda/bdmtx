"""Evaluation harness for measuring decode-rate, latency and per-stage metrics.

Runs the pipeline on a dataset and records per-image results including
segmentation IoU (via mask overlap), ROI extraction success, enhancement status,
decoder success, and timings.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bdmtx.data.dataset import SyntheticDataset, build_mask_from_segmentation
from bdmtx.segmentation import postprocess_mask, predict_mask


@dataclass
class Result:
    """Evaluation result."""

    image: str
    seg_time_ms: float
    roi_time_ms: float
    enhance_time_ms: float
    decode_time_ms: float
    seg_iou: float | None
    roi_extracted: bool
    decode_success: bool
    decode_data: str | None
    logs: list[str]


def compute_iou(mask_gt: np.ndarray | None, mask_pred: np.ndarray | None) -> float:
    """Compute intersection-over-union.

    Args:
        mask_gt: mask ground truth, binary array
        mask_pred: mask prediction

    Returns:
        the IoU metric
    """
    if mask_gt is None or mask_pred is None:
        return 0.0
    gt = (mask_gt > 0).astype(np.uint8)
    pd = (mask_pred > 0).astype(np.uint8)
    inter = int((gt & pd).sum())
    union = int((gt | pd).sum())
    if union == 0:
        return 0.0
    return inter / union


def evaluate_dataset(
    root: Path, model_path: str | None = None, max_images: int | None = None
) -> list[dict[str, Any]]:
    """Evaluation model on a dataset.

    Args:
        root: images root directory
        model_path: path to model's weights
        max_images: limit number of evaluated images

    Returns:
        a list of metrics
    """
    ds = SyntheticDataset(root, split="degraded")
    results: list[dict[str, Any]] = []

    n = len(ds)
    for i in range(n if max_images is None else min(n, max_images)):
        sample = ds[i]
        img = sample["image"]

        # call segmentation directly

        t0 = time.time()
        seg_mask = predict_mask(img, model_path)
        seg_mask = postprocess_mask(seg_mask)
        t1 = time.time()
        seg_ms = (t1 - t0) * 1000.0

        # ROI extraction timing
        t0 = time.time()
        from bdmtx.roi import extract_and_normalize_roi

        roi, quad = extract_and_normalize_roi(img, seg_mask, out_size=256)
        t1 = time.time()
        roi_ms = (t1 - t0) * 1000.0

        # enhancement timing
        t0 = time.time()
        enhanced = None
        if roi is not None:
            enhanced = None
            try:
                from bdmtx.enhancement import enhance

                enhanced = enhance(roi)
            except Exception:
                enhanced = roi
        t1 = time.time()
        enhance_ms = (t1 - t0) * 1000.0

        # decoding timing
        t0 = time.time()
        decode_res = {"success": False, "data": None, "logs": []}
        if enhanced is not None:
            from bdmtx.decoder import decode_image

            decode_res = decode_image(enhanced)
        t1 = time.time()
        decode_ms = (t1 - t0) * 1000.0

        # compute seg iou if annotation exists
        seg_iou = None
        if (ann := sample.get("annotation")) and ann.exists():
            with open(ann, encoding="utf-8") as fh:
                import json

                ann = json.load(fh)
            anns = ann.get("annotations", [])
            if anns:
                poly = anns[0].get("segmentation", [])
                if poly:
                    poly0 = poly[0] if isinstance(poly[0], list) else poly
                    gt_mask = build_mask_from_segmentation(
                        poly0, img.shape[1], img.shape[0]
                    )
                    seg_iou = compute_iou(gt_mask, seg_mask)

        res = Result(
            image=str(sample.get("image")),
            seg_time_ms=seg_ms,
            roi_time_ms=roi_ms,
            enhance_time_ms=enhance_ms,
            decode_time_ms=decode_ms,
            seg_iou=seg_iou,
            roi_extracted=roi is not None,
            decode_success=bool(decode_res.get("success")),
            decode_data=decode_res.get("data"),
            logs=decode_res.get("logs", list()),
        )
        results.append(asdict(res))

    return results
