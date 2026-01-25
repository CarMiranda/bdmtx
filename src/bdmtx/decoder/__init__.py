"""Decoder wrapper module. Prefer libdmtx, provide ZXing fallback if needed."""

from __future__ import annotations

from bdmtx.decoder.libdmtx_wrapper import (
    Result,
)
from bdmtx.decoder.libdmtx_wrapper import (
    decode_with_fallbacks as decode_image_libdmtx,
)


def decode_image(image) -> Result:
    """Decode an image.

    Args:
        image: np.ndarray, or PIL.image

    Returns:
        datamatrix decoding result record
    """
    return decode_image_libdmtx(image)
