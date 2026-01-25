import cv2
import numpy as np

from bdmtx.roi import extract_and_normalize_roi


def make_test_image():
    """Make an image of a rectangle and its mask."""
    img = np.zeros((200, 300, 3), dtype=np.uint8) + 128
    # draw a white rectangle (symbol)
    cv2.rectangle(img, (80, 60), (160, 140), (255, 255, 255), -1)
    mask = np.zeros((200, 300), dtype=np.uint8)
    cv2.rectangle(mask, (80, 60), (160, 140), 255, -1)
    return img, mask


def test_extract_and_warp():
    """Tests extraction and warping."""
    img, mask = make_test_image()
    roi, quad = extract_and_normalize_roi(img, mask, out_size=64)
    assert roi is not None
    assert roi.shape == (64, 64, 3)
    # ensure ROI contains bright region
    assert roi.mean() > 100
