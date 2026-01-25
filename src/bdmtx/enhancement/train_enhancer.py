"""Training loop stub for the enhancement model (PyTorch).

This provides dataset plumbing for paired degraded->clean training samples using
our SyntheticDataset and a tiny training loop for the TinyEnhancer network.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from bdmtx.data.dataset import SyntheticDataset

from .enhancer import export_to_onnx


class EnhancementTorchDataset(Dataset):
    """Enhancement training dataset."""

    def __init__(self, root: Path, split: str = "train"):
        self.ds = SyntheticDataset(root, split="degraded")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        s = self.ds[idx]
        # use degraded as input, clean as target if available
        inp = s.get("degraded") if s.get("degraded") is not None else s.get("image")
        tgt = s.get("clean") if s.get("clean") is not None else s.get("image")
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        inp = np.expand_dims(inp, 0)
        tgt = np.expand_dims(tgt, 0)
        return torch.from_numpy(inp), torch.from_numpy(tgt)


class TinyEnhancer(nn.Module):
    """Shallow U-net model for image enhancement."""

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
        """Forward pass.

        Args:
            x: input tensor
        """
        return self.net(x)


def train_enhancer(root: Path, epochs: int = 20, batch: int = 8, lr: float = 1e-3):
    """Enhancer training entrypoint.

    Train a `TinyEnhancer` on a `EnhancementTorchDataset`, then export to ONNX.

    Args:
        root: root image directory
        epochs: number of epoch to train for
        batch: batch size
        lr: learning rate
    """
    ds = EnhancementTorchDataset(root)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2)

    model = TinyEnhancer()
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for inp, tgt in dl:
            pred = model(inp)
            loss = loss_fn(pred, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * inp.size(0)
        print(f"Epoch {epoch + 1}/{epochs}: loss={total / len(ds):.4f}")

    # export model
    export_to_onnx(Path("models/enhancer.onnx"), input_size=256)
