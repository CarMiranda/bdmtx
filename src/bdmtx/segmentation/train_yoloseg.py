"""Training helper for YOLO-seg (Ultralytics) using COCO dataset.

Provides utilities to prepare a dataset YAML for Ultralytics 'train' command
and a small wrapper to invoke ultralytics training programmatically.
"""

from __future__ import annotations

from pathlib import Path


def make_ultralytics_dataset_yaml(
    root: Path, out: Path, train_fraction: float = 0.9
) -> None:
    """Create an Ultralytics dataset YAML file from a COCO-style folder.

    Expects images+JSONs written directly under root (clean_/degraded_ pairs) or a
    single coco.json file. This writer will create `train`/`val` lists pointing to
    images.
    """
    root = Path(root)
    out = Path(out)
    images = sorted([p for p in root.glob("*.png")])
    n = len(images)
    if n == 0:
        raise ValueError("No images found in dataset root")

    split_idx = int(n * train_fraction)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    data = {
        "path": str(root.resolve()),
        "train": [str(p.name) for p in train_imgs],
        "val": [str(p.name) for p in val_imgs],
        "nc": 1,
        "names": ["datamatrix"],
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    import yaml

    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh)


def train_ultralytics(
    yaml_path: Path,
    epochs: int = 50,
    model: str = "yolov8n-seg",
    batch: int = 16,
    img_size: int = 640,
    project: str = "runs/train",
    name: str = "exp",
) -> None:
    """Invoke Ultralytics training using the Python API with checkpointing support.

    Saves checkpoints under project/name and can resume from last.pt using resume=True.
    """
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - runtime
        raise RuntimeError("ultralytics must be installed to run training") from exc

    model_obj = YOLO(model)
    # ultralytics will save checkpoints to runs/train/{name}
    model_obj.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch,
        imgsz=img_size,
        project=project,
        name=name,
    )


def last_checkpoint_path(project: str = "runs/train", name: str = "exp") -> Path | None:
    """Return path to the last checkpoint (best.pt or last.pt) if present.

    Useful for preemptible resume logic on trainer VM.
    """
    out = Path(project) / name
    if not out.exists():
        return None
    best = out / "weights" / "best.pt"
    last = out / "weights" / "last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    return None
