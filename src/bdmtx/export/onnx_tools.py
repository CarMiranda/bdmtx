"""ONNX export and simple benchmark utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_onnx_model(model_path: Path, dummy_shape: tuple[int, int, int, int]):
    """Placeholder wrapper that indicates how to export models to ONNX.

    Actual export helpers are provided in respective modules
    (enhancement.export_to_onnx).
    """
    print(f"Exported ONNX to {model_path} (dummy shape {dummy_shape})")


def benchmark_onnx(model_path: Path, input_shape: tuple[int, int, int]):
    """Run a simple ONNX Runtime benchmark for a model with given input shape.

    Returns ms per inference.
    """
    try:
        import onnxruntime as ort
    except Exception:
        raise RuntimeError("onnxruntime required for benchmarking")

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    dummy = np.random.randn(1, *input_shape).astype(np.float32)
    import time

    # warmup
    for _ in range(5):
        sess.run(None, {inp_name: dummy})
    iters = 20
    t0 = time.time()
    for _ in range(iters):
        sess.run(None, {inp_name: dummy})
    t1 = time.time()
    avg_ms = (t1 - t0) / iters * 1000.0
    return avg_ms
