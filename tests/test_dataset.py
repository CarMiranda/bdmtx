from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")


def test_dataset_and_validation(tmp_path: Path):
    """Test dataset and validation."""
    from bdmtx.data import create_dataset
    from bdmtx.data.dataset import (
        SyntheticDataset,
        simple_augmentation,
        validate_dataset,
    )

    out = tmp_path
    # create a small dataset
    create_dataset(out, n=3, size=(128, 128))

    errors = validate_dataset(out)
    assert errors == [], f"Validation errors: {errors}"

    ds = SyntheticDataset(out, split="degraded", transforms=simple_augmentation)
    assert len(ds) == 3
    sample = ds[0]
    assert "image" in sample and sample["image"] is not None
    assert "mask" in sample and sample["mask"].shape == sample["image"].shape[:2]
    assert sample["mask"].dtype == sample["mask"].dtype
