"""Segmentation model interface and postprocessing stubs."""

from __future__ import annotations


def predict_mask(image) -> object:
    """Stub: returns placeholder mask."""
    return None


def postprocess_mask(mask) -> object:
    """Stub: largest component / contour extraction placeholder."""
    return None
