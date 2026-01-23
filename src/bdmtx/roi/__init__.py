"""ROI utilities package re-export."""

from __future__ import annotations

from .homography import (
    extract_and_normalize_roi,
    quadrilateral_from_contour,
    largest_contour_from_mask,
    warp_quad_to_square,
)
from .geometry import quad_area, is_convex_quad, expand_quad

__all__ = [
    "extract_and_normalize_roi",
    "quadrilateral_from_contour",
    "largest_contour_from_mask",
    "warp_quad_to_square",
    "quad_area",
    "is_convex_quad",
    "expand_quad",
]
