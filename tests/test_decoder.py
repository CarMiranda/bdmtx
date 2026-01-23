import numpy as np
import cv2

from bdmtx.decoder import decode_image


def test_decoder_no_libdmtx():
    img = np.zeros((100,100,3), dtype=np.uint8)
    res = decode_image(img)
    assert isinstance(res, dict)
    assert not res.get('success')
