"""VIA (VGG Image Annotator) -> COCO converter.

Creates a single COCO-format JSON containing all images and annotations
found in the provided VIA JSON file. For each VIA region the converter
produces a COCO annotation with segmentation (polygon), bbox, area and
attributes. The datamatrix text (if present in VIA region attributes)
is copied to annotations[].attributes["content"] and annotations[].attributes["type"] is set to "datamatrix".

Usage:
    python via_to_coco.py annotations_via.json /path/to/images out_coco.json

This is intentionally lightweight and defensive to handle common VIA variants.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image


def _load_via(via_path: Path) -> List[Dict[str, Any]]:
    with open(via_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    entries: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        if "_via_settings" in data:
            data = data["_via_img_metadata"]
            entries = list(data.values())
        else:
            # VIA export commonly uses file-ids as top-level keys; values are entries
            for _, v in data.items():
                if isinstance(v, dict):
                    entries.append(v)
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError("Unsupported VIA JSON structure")
    return entries


def _regions_list(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    regions = entry.get("regions", [])
    if isinstance(regions, dict):
        # older VIA versions use a dict mapping region-id -> region
        return list(regions.values())
    if isinstance(regions, list):
        return regions
    return []


def _shape_to_polygons(shape: Dict[str, Any]) -> List[List[float]]:
    """Convert VIA shape_attributes into a list of polygons (each polygon is flat [x0,y0,x1,y1,...])."""
    if not shape:
        return []

    name = (shape.get("name") or shape.get("shape") or shape.get("type") or "").lower()

    # Direct polygon fields (common)
    if "all_points_x" in shape and "all_points_y" in shape:
        xs = shape.get("all_points_x", [])
        ys = shape.get("all_points_y", [])
        if len(xs) != len(ys) or len(xs) < 3:
            return []
        poly: List[float] = []
        for x, y in zip(xs, ys):
            poly.append(float(x))
            poly.append(float(y))
        return [poly]

    # Polygon indicated by name
    if name in ("polygon", "poly"):
        xs = shape.get("all_points_x", [])
        ys = shape.get("all_points_y", [])
        if len(xs) != len(ys) or len(xs) < 3:
            return []
        poly = [float(v) for pair in zip(xs, ys) for v in pair]
        return [poly]

    # Rectangle -> convert to polygon
    if name in ("rect", "rectangle") or (
        "x" in shape and "y" in shape and ("width" in shape or "w" in shape)
    ):
        x = float(shape.get("x", 0))
        y = float(shape.get("y", 0))
        w = float(shape.get("width", shape.get("w", 0)))
        h = float(shape.get("height", shape.get("h", 0)))
        if w <= 0 or h <= 0:
            return []
        poly = [x, y, x + w, y, x + w, y + h, x, y + h]
        return [poly]

    # unsupported shape types (circle, ellipse, polyline) -> ignore
    return []


def _polygon_area(poly: List[float]) -> float:
    coords = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    x = coords[:, 0]
    y = coords[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _bbox_from_poly(poly: List[float]) -> List[float]:
    coords = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    x_min = float(coords[:, 0].min())
    y_min = float(coords[:, 1].min())
    x_max = float(coords[:, 0].max())
    y_max = float(coords[:, 1].max())
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def via_to_coco(
    via_json: Path | str,
    images_dir: Path | str | None,
    out_json: Path | str,
    category_name: str = "datamatrix",
) -> None:
    """Convert VIA JSON + images directory into a single COCO JSON file.

    Args:
        via_json: path to VIA JSON file
        images_dir: directory containing the image files referenced in VIA
        out_json: path to write combined COCO JSON
        category_name: category name to use for all regions (default: datamatrix)
    """
    via_json = Path(via_json)
    out_json = Path(out_json)
    images_dir = Path(images_dir or "/tmp/inexistent")

    entries = _load_via(via_json)
    breakpoint()

    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []

    ann_id = 1
    img_id = 1

    for entry in entries:
        filename = (
            entry.get("filename")
            or entry.get("file_name")
            or entry.get("file")
            or entry.get("name")
        )
        if not filename:
            # skip entries without a filename
            continue

        img_path = images_dir / filename
        if not img_path.exists():
            # fall back to treating filename as a path
            img_path = Path(filename)

        if img_path.exists():
            w, h = Image.open(img_path).size
            # img = cv2.imread(str(img_path))
            # if img is None:
            #     print(f"Warning: failed to read image {img_path}; skipping")
            #     continue
            # h, w = img.shape[:2]
        else:
            # try to read size metadata from VIA entry; if absent, skip
            w = int(entry.get("width", 0) or 0)
            h = int(entry.get("height", 0) or 0)
            if w <= 0 or h <= 0:
                print(
                    f"Warning: image {filename} not found and missing size metadata; skipping"
                )
                continue

        images.append({"id": img_id, "file_name": filename, "width": w, "height": h})

        regions = _regions_list(entry)
        for region in regions:
            shape = region.get("shape_attributes", {}) or region.get("shape", {})
            region_attrs = region.get("region_attributes", {}) or {}
            polys = _shape_to_polygons(shape)
            for poly in polys:
                if len(poly) < 6:
                    continue
                bbox = _bbox_from_poly(poly)
                area = _polygon_area(poly)

                attributes = (
                    dict(region_attrs)
                    if isinstance(region_attrs, dict)
                    else {"_raw": region_attrs}
                )
                # ensure content is present in attributes["content"] if available under other keys
                if not attributes.get("content"):
                    for k in ("content", "text", "label", "value", "chars"):
                        if k in attributes and attributes[k]:
                            attributes["content"] = attributes[k]
                            break
                if "content" not in attributes:
                    attributes["content"] = ""
                # required type field for downstream pipeline
                attributes["type"] = "datamatrix"

                ann = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": [poly],
                    "bbox": [
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ],
                    "area": float(area),
                    "iscrowd": 0,
                    "attributes": attributes,
                }
                annotations.append(ann)
                ann_id += 1

        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": category_name}],
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(coco, fh, ensure_ascii=False, indent=2)

    print(
        f"Wrote COCO JSON to {out_json} with {len(images)} images and {len(annotations)} annotations."
    )


def _main_cli() -> None:
    import argparse

    p = argparse.ArgumentParser(
        prog="via_to_coco",
        description="Convert VGG Image Annotator JSON into a single COCO JSON",
    )
    p.add_argument("via_json", help="Path to VIA JSON file")
    p.add_argument("images_dir", help="Directory containing images referenced by VIA")
    p.add_argument("out_json", help="Output COCO JSON path")
    p.add_argument(
        "--category", default="DATAMATRIX", help="Category name to use for all regions"
    )
    args = p.parse_args()
    via_to_coco(args.via_json, args.images_dir, args.out_json, args.category)


if __name__ == "__main__":
    _main_cli()
