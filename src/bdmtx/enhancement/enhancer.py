"""Simple enhancement model and ONNX export utilities.

Provides a tiny CNN-based enhancer (PyTorch) for grayscale ROIs, training stubs,
and an export_to_onnx helper. The model is intentionally tiny to meet latency
constraints for edge inference and to ease testing.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path


def enhance(roi: "np.ndarray") -> "np.ndarray":
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
def export_to_onnx(out_path: Path, input_size: int = 256) -> None:
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError("torch is required for ONNX export") from exc

    class TinyEnhancer(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 8, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 1, 3, padding=1),
            )

        def forward(self, x):
            return self.net(x)

    model = TinyEnhancer()
    model.eval()
    dummy = torch.randn(1, 1, input_size, input_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import torch.onnx

    torch.onnx.export(model, dummy, str(out_path), opset_version=14, input_names=["input"], output_names=["output"])
