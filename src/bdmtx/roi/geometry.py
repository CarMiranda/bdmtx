"""Additional geometric helpers for quadrilateral validation and fallback heuristics."""

from __future__ import annotations

import numpy as np


def quad_area(quad: np.ndarray) -> float:
    """Compute the area of a quadrilateral.

    Args:
        quad: [N, 2] array with coordinates (x, y)

    Returns:
        the area of the quadrilateral
    """
    coords = quad.reshape(-1, 2)
    x = coords[:, 0]
    y = coords[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def is_convex_quad(quad: np.ndarray) -> bool:
    """Whether the input quadrilateral is convex.

    Args:
        quad: [N, 2] array with coordinates (x, y)

    Returns:
        whether `quad` is convex
    """
    coords = quad.reshape(-1, 2)
    # compute cross products sign
    signs = []
    for i in range(4):
        a = coords[i]
        b = coords[(i + 1) % 4]
        c = coords[(i + 2) % 4]
        ab = b - a
        bc = c - b
        cross = ab[0] * bc[1] - ab[1] * bc[0]
        signs.append(cross)
    return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)


def expand_quad(quad: np.ndarray, factor: float = 1.1) -> np.ndarray:
    """Resize quadrilateral from its center.

    Args:
        quad: [N, 2] array with coordinates (x, y)
        factor: resizing factor (multiplicative)

    Returns:
        the resized quadrilateral
    """
    center = quad.mean(axis=0)
    return center + (quad - center) * factor
