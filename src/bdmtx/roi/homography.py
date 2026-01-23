"""ROI extraction, contour processing and homography normalization utilities.

Provides functions to:
- extract largest contour from a binary mask
- approximate oriented bounding box / quadrilateral
- compute homography to warp to a square ROI while preserving finder patterns
- simple heuristics to handle missing corners
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np


def largest_contour_from_mask(mask: "np.ndarray") -> Tuple[Optional[np.ndarray], float]:
    """Return largest contour (Nx2 int coords) and its area from binary mask (0/255).

    Returns (contour, area) where contour is None if no contours found.
    """
    try:
        import cv2
    except Exception:
        raise RuntimeError("opencv-python is required for contour processing")

    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    best = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(best))
    return best.reshape(-1, 2), area


def quadrilateral_from_contour(cnt: "np.ndarray") -> Optional[np.ndarray]:
    """Approximate a quadrilateral from contour using approxPolyDP or minAreaRect.

    Returns 4x2 float coordinates in consistent order (tl,tr,br,bl) or None.
    """
    try:
        import cv2
    except Exception:
        raise RuntimeError("opencv-python is required for contour processing")

    if cnt is None or len(cnt) < 4:
        return None

    peri = cv2.arcLength(cnt.astype(np.float32), True)
    approx = cv2.approxPolyDP(cnt.astype(np.float32), 0.02 * peri, True)
    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(float)
    else:
        # fallback to minAreaRect
        rect = cv2.minAreaRect(cnt.astype(np.float32))
        pts = cv2.boxPoints(rect)
    # order points tl, tr, br, bl
    pts_ordered = _order_quad_points(pts)
    return pts_ordered


def _order_quad_points(pts: "np.ndarray") -> "np.ndarray":
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.vstack([tl, tr, br, bl]).astype(float)


def warp_quad_to_square(image: "np.ndarray", quad: "np.ndarray", out_size: int = 256) -> "np.ndarray":
    """Warp quadrilateral region to a square output of size out_size x out_size.

    Preserves module geometry by using INTER_NEAREST for resizing.
    """
    try:
        import cv2
    except Exception:
        raise RuntimeError("opencv-python is required for homography")

    dst = np.array([[0.0, 0.0], [out_size - 1.0, 0.0], [out_size - 1.0, out_size - 1.0], [0.0, out_size - 1.0]], dtype=np.float32)
    H, _ = cv2.findHomography(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, H, (out_size, out_size), flags=cv2.INTER_NEAREST)
    return warped


def extract_and_normalize_roi(image: "np.ndarray", mask: "np.ndarray", out_size: int = 256) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Given image and binary mask, extract the largest DataMatrix ROI and return (warped_roi, quad_pts).

    Returns (None, None) if no valid ROI could be extracted.
    """
    cnt, area = largest_contour_from_mask(mask)
    if cnt is None or area <= 1.0:
        return None, None
    quad = quadrilateral_from_contour(cnt)
    if quad is None:
        return None, None

    warped = warp_quad_to_square(image, quad, out_size=out_size)
    return warped, quad
