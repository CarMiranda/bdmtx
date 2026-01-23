"""Additional geometric helpers for quadrilateral validation and fallback heuristics."""

from __future__ import annotations

import numpy as np


def quad_area(quad: "np.ndarray") -> float:
    coords = quad.reshape(-1, 2)
    x = coords[:, 0]
    y = coords[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def is_convex_quad(quad: "np.ndarray") -> bool:
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


def expand_quad(quad: "np.ndarray", factor: float = 1.1) -> "np.ndarray":
    center = quad.mean(axis=0)
    return center + (quad - center) * factor
