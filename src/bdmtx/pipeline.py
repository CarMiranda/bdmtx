"""End-to-end pipeline: segmentation -> ROI -> enhancement -> decode wrapper.

This module wires the segmentation, ROI extraction, enhancement and decoder
stubs together for end-to-end inference and debugging.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np

from .decoder import decode_image
from .enhancement import enhance
from .roi import extract_and_normalize_roi
from .segmentation import postprocess_mask, predict_mask


class PipelineResult(TypedDict):
    """A pipeline result record."""

    # Whether the pipeline ran successfully
    success: bool

    # Decoded data
    data: Any | None

    # A list of coordinates representing the detected code
    quad: Any | None

    # Reason the pipeline failed
    reason: str


def run_pipeline(image: np.ndarray, model_path: str | None = None) -> PipelineResult:
    """Run the end-to-end pipeline on `image`.

    Args:
        image: input images
        model_path: model to use for prediction

    Returns:
        result dictionary
    """
    # segmentation
    mask = predict_mask(image, model_path=model_path)
    mask = postprocess_mask(mask)

    # ROI extraction
    roi, quad = extract_and_normalize_roi(image, mask, out_size=256)
    if roi is None:
        return {
            "success": False,
            "reason": "no_roi",
            "data": None,
            "quad": None,
        }

    # enhance
    enhanced = enhance(roi)

    # decode
    result = decode_image(enhanced)
    success = bool(result.get("success"))
    return {
        "success": success,
        "data": result.get("data"),
        "quad": quad.tolist() if quad is not None else None,
        "reason": "" if success else "could_not_decode",
    }
