from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.onnx

from bdmtx.enhancement.train_enhancer import TinyEnhancer


def enhance(roi: np.ndarray) -> np.ndarray:
    """Naive enhancement: apply unsharp masking and CLAHE on grayscale ROI.

    This is a placeholder until a trained model is added. ROI expected as BGR uint8.
    """
    import cv2

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    # unsharp
    blur = cv2.GaussianBlur(cl, (3, 3), 0)
    sharp = cv2.addWeighted(cl, 1.5, blur, -0.5, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


# ONNX export helper (requires torch)
def export_to_onnx(out_path: Path, input_size: int = 256):
    """Export model to ONNX.

    Args:
        out_path: output path
        input_size: model input size
    """
    model = TinyEnhancer()
    model.eval()
    dummy = torch.randn(1, 1, input_size, input_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
    )
