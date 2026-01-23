"""YOLO-seg (ultralytics) segmentation wrapper.

Provides a lightweight wrapper around the ultralytics YOLO model for segmentation
that returns a binary mask for the largest detected DataMatrix instance.

This module is defensive: if ultralytics is not installed it raises a helpful error
when load_model() is called.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class YOLOSegModel:
    def __init__(self, model_path: str | None = None, device: str | None = None):
        self.model_path = model_path or "yolov8n-seg"
        self.device = device
        self._model = None

    def load(self):
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("ultralytics is required for YOLOSegModel. Install with `pip install ultralytics`") from exc

        self._model = YOLO(self.model_path)
        return self

    def predict_mask(self, image: "np.ndarray", conf: float = 0.3) -> np.ndarray:
        """Run segmentation on a BGR uint8 image and return a binary mask (0/255).

        If multiple instance masks are detected they are merged and the largest
        connected component is returned.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded; call load() first")

        # ultralytics accepts BGR numpy arrays directly
        results = self._model.predict(source=image, conf=conf, verbose=False)
        if not results:
            h, w = image.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

        r = results[0]
        mask_out: Optional[np.ndarray] = None

        # try to access segmentation masks (may be present for seg models)
        masks = getattr(r, "masks", None)
        if masks is not None:
            try:
                # masks.data is expected to be a boolean array (num_masks, H, W)
                arr = masks.data.cpu().numpy() if hasattr(masks.data, "cpu") else masks.data.numpy()
            except Exception:
                arr = np.asarray(masks.data)

            if arr.ndim == 3:
                merged = np.any(arr, axis=0)
                mask_out = (merged.astype(np.uint8) * 255)
            elif arr.ndim == 2:
                mask_out = (arr.astype(np.uint8) * 255)

        # fallback: try to rasterize polygons (segmentation field)
        if mask_out is None and hasattr(r, "boxes"):
            segs = getattr(r.boxes, "segmentation", None)
            if segs is not None:
                # segs may be list of flat lists
                h, w = image.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                for s in segs:
                    if s is None:
                        continue
                    poly = np.asarray(s).reshape(-1, 2).round().astype(int)
                    try:
                        import cv2

                        cv2.fillPoly(mask, [poly], 255)
                    except Exception:
                        pass
                mask_out = mask

        if mask_out is None:
            h, w = image.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

        # simple postprocessing: keep largest connected component
        try:
            import cv2

            num_labels, labels = cv2.connectedComponents(mask_out)
            if num_labels <= 1:
                return mask_out
            # find largest non-zero component
            max_area = 0
            best = 0
            for lab in range(1, num_labels):
                area = int((labels == lab).sum())
                if area > max_area:
                    max_area = area
                    best = lab
            return (labels == best).astype(np.uint8) * 255
        except Exception:
            # if cv2 not available, return merged mask
            return mask_out
