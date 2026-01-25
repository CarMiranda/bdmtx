from __future__ import annotations

from typing import Any, TypedDict

import cv2
import numpy as np
from pylibdmtx.pylibdmtx import decode as _decode


class Result(TypedDict):
    """DataMatrix decoding result record."""

    success: bool
    data: Any | None
    confidence: float | None
    logs: list[str]


def _try_libdmtx_decode(image: np.ndarray) -> Result:
    try:
        results = _decode(image)
        if not results:
            return {
                "success": False,
                "data": None,
                "confidence": None,
                "logs": ["no result"],
            }
        r = results[0]
        data = (
            r.data.decode("utf-8")
            if isinstance(r.data, bytes | bytearray)
            else str(r.data)
        )
        # pylibdmtx does not expose confidence; estimate it from quality if present
        conf = None
        if hasattr(r, "quality"):
            try:
                conf = float(r.quality)
            except Exception:
                conf = None
        return {"success": True, "data": data, "confidence": conf, "logs": [str(r)]}
    except Exception as exc:
        return {
            "success": False,
            "data": None,
            "confidence": None,
            "logs": [f"libdmtx error: {exc}"],
        }


def decode_with_fallbacks(image: np.ndarray) -> Result:
    """Decode an image.

    Args:
        image: `numpy.ndarray`, `PIL.Image` or tuple (pixels, width, height)

    Returns:
        a datamatrix decode result record
    """
    logs = []
    # Try direct decode
    res = _try_libdmtx_decode(image)
    logs.extend(res.get("logs", []))
    if res.get("success"):
        return res

    # Prepare variants
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1) inverted
    inv = 255 - gray
    res = _try_libdmtx_decode(inv)
    logs.extend(res.get("logs", []))
    if res.get("success"):
        return res

    # 2) adaptive thresholds
    for block in (11, 15, 21):
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 2
        )
        res = _try_libdmtx_decode(th)
        logs.extend(res.get("logs", []))
        if res.get("success"):
            return res

    # 3) contrast variants
    for alpha in (0.8, 1.0, 1.2):
        for beta in (-20, 0, 20):
            adj = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
            res = _try_libdmtx_decode(adj)
            # include which transform was used in logs for debugging
            logs.extend([f"contrast alpha={alpha} beta={beta}"] + res.get("logs", []))
            if res.get("success"):
                return res

    # 4) multi-scale attempts
    for scale in (0.5, 1.0, 1.5):
        h, w = gray.shape
        resized = cv2.resize(
            gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
        )
        res = _try_libdmtx_decode(resized)
        logs.extend([f"scale={scale}"] + res.get("logs", []))
        if res.get("success"):
            return res

    return {"success": False, "data": None, "confidence": None, "logs": logs}
