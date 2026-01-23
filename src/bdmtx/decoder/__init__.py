"""Decoder wrapper module. Prefer libdmtx, provide ZXing fallback if needed."""

from __future__ import annotations

import warnings

try:
    from .libdmtx_wrapper import decode_image as decode_image_libdmtx
except Exception:  # pragma: no cover - dependency optional
    decode_image_libdmtx = None


def decode_image(image) -> dict:
    if decode_image_libdmtx is None:
        warnings.warn("libdmtx wrapper not available; decoder will always fail unless pylibdmtx is installed")
        return {"success": False, "data": None}
    return decode_image_libdmtx(image)
