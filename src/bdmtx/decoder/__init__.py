"""Decoder wrapper module (stub).

Provides a minimal API that will be expanded when integrating libdmtx.
"""

from __future__ import annotations


def decode_image(image) -> dict:
    """Stub decoder: returns a fake failed decode result.

    Args:
        image: image ndarray or path-like (implementation TBD)

    Returns:
        dict: {"success": bool, "data": str | None}
    """
    return {"success": False, "data": None}
