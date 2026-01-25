"""ROI utilities package re-export."""

from __future__ import annotations

from .geometry import expand_quad, is_convex_quad, quad_area
from .homography import (
    extract_and_normalize_roi,
    largest_contour_from_mask,
    quadrilateral_from_contour,
    warp_quad_to_square,
)

__all__ = [
    "extract_and_normalize_roi",
    "quadrilateral_from_contour",
    "largest_contour_from_mask",
    "warp_quad_to_square",
    "quad_area",
    "is_convex_quad",
    "expand_quad",
]
