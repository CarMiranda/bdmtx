"""Segmentation model interface and YOLO-seg integration."""

from __future__ import annotations

import os
import warnings

import numpy as np

from .yoloseg import YOLOSegModel

_model: YOLOSegModel | None = None


def load_default_model(model_path: str | None = None) -> YOLOSegModel:
    """Load default model.

    Args:
        model_path: path to the model's weight

    Returns:
        A yolo-seg model
    """
    global _model
    if _model is None:
        _model = YOLOSegModel(model_path=model_path)
        try:
            _model.load()
        except RuntimeError as exc:
            warnings.warn(str(exc))
            raise
    return _model


def predict_mask(
    image: np.ndarray, model_path: str | None = None, conf: float = 0.3
) -> np.ndarray:
    """Predict binary mask for input image using YOLO-seg.

    Args:
        image: BGR uint8 image
        model_path: optional model weight or name (e.g., 'yolov8n-seg')
        conf: confidence threshold

    Returns:
        uint8 binary mask (0/255)
    """
    model = load_default_model(
        model_path=model_path or os.environ.get("BDMTX_YOLO_MODEL")
    )
    return model.predict_mask(image, conf=conf)


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """Postprocess mask: keep largest connected component and return binary mask."""
    try:
        import cv2

        num_labels, labels = cv2.connectedComponents(mask)
        if num_labels <= 1:
            return mask
        max_area = 0
        best = 0
        for lab in range(1, num_labels):
            area = int((labels == lab).sum())
            if area > max_area:
                max_area = area
                best = lab
        return (labels == best).astype(np.uint8) * 255
    except Exception:
        return mask
