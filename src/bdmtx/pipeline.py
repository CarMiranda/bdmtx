"""End-to-end pipeline: segmentation -> ROI -> enhancement -> decode wrapper.

This module wires the segmentation, ROI extraction, enhancement and decoder
stubs together for end-to-end inference and debugging.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .segmentation import predict_mask, postprocess_mask
from .roi import extract_and_normalize_roi
from .enhancement import enhance
from .decoder import decode_image


def run_pipeline(image: "np.ndarray", model_path: str | None = None) -> dict:
    # segmentation
    mask = predict_mask(image, model_path=model_path)
    mask = postprocess_mask(mask)

    # ROI extraction
    roi, quad = extract_and_normalize_roi(image, mask, out_size=256)
    if roi is None:
        return {"success": False, "reason": "no_roi"}

    # enhance
    enhanced = enhance(roi)

    # decode
    result = decode_image(enhanced)
    return {
        "success": bool(result.get("success")),
        "data": result.get("data"),
        "quad": quad.tolist() if quad is not None else None,
    }
