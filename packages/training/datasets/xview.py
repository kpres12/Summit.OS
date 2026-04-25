"""
xView (1) — Overhead Object Detection Dataset
================================================
DARPA / DIUx large-scale aerial / satellite object detection dataset.
~1M annotated objects across 60 classes, 0.3 m WorldView-3 imagery.

Source:    http://xviewdataset.org
License:   xView dataset license (research, free; account required).

Why this dataset:
  - Real overhead imagery — same geometry as ISR / federal customers
  - 60 classes covering vehicles (passenger, truck, military), aircraft
    (small, helicopter, jet), maritime (small boat, container ship,
    tugboat, etc.), structures (hangar, shed, tower), construction
    equipment (bulldozer, crane, excavator) — full ATR taxonomy
  - Civilian use: traffic monitoring, port operations, infrastructure
    inspection, deforestation
  - Federal use: ATR — every federal customer wants overhead object
    detection

Expected on-disk layout after manual download:
  packages/training/data/xview/
    train_images/<image_id>.tif       (1m to 10m+ each)
    train_labels.geojson              (single GeoJSON, ~1.4M features)
    val_images/<image_id>.tif         (optional)
    xview_class_labels.txt            (id -> name mapping)

Usage:
    from packages.training.datasets.xview import (
        load_class_map, load_labels, load_as_training_samples,
    )
    samples = load_as_training_samples()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

OUT_DIR     = Path(__file__).parent.parent / "data" / "xview"
LABELS_PATH = OUT_DIR / "train_labels.geojson"
CLASSES_TXT = OUT_DIR / "xview_class_labels.txt"
IMAGES_DIR  = OUT_DIR / "train_images"


# ---------------------------------------------------------------------------
# Class map (the canonical 60-class taxonomy as published by xView)
# ---------------------------------------------------------------------------
# Maintained in code so we can run without the txt file present. Source:
# https://github.com/DIUx-xView/xview-yolt/blob/master/utils/xview_class_labels.txt

DEFAULT_CLASS_MAP: dict[int, str] = {
    11: "Fixed-wing Aircraft", 12: "Small Aircraft", 13: "Cargo Plane",
    15: "Helicopter",
    17: "Passenger Vehicle", 18: "Small Car", 19: "Bus", 20: "Pickup Truck",
    21: "Utility Truck", 23: "Truck", 24: "Cargo Truck", 25: "Truck w/Box",
    26: "Truck Tractor", 27: "Trailer", 28: "Truck w/Flatbed",
    29: "Truck w/Liquid",
    32: "Crane Truck", 33: "Railway Vehicle", 34: "Passenger Car",
    35: "Cargo Car", 36: "Flat Car", 37: "Tank car",
    38: "Locomotive",
    40: "Maritime Vessel", 41: "Motorboat", 42: "Sailboat", 44: "Tugboat",
    45: "Barge", 47: "Fishing Vessel", 49: "Ferry", 50: "Yacht",
    51: "Container Ship", 52: "Oil Tanker",
    53: "Engineering Vehicle", 54: "Tower crane", 55: "Container Crane",
    56: "Reach Stacker", 57: "Straddle Carrier", 59: "Mobile Crane",
    60: "Dump Truck", 61: "Haul Truck", 62: "Scraper/Tractor",
    63: "Front loader/Bulldozer", 64: "Excavator",
    65: "Cement Mixer", 66: "Ground Grader",
    71: "Hut/Tent", 72: "Shed", 73: "Building", 74: "Aircraft Hangar",
    76: "Damaged Building", 77: "Facility",
    79: "Construction Site", 83: "Vehicle Lot",
    84: "Helipad", 86: "Storage Tank", 89: "Shipping container lot",
    91: "Shipping Container",
    93: "Pylon", 94: "Tower",
}


# Group classes into operational supercategories useful for ranking + filtering
SUPERCATEGORY = {
    "aircraft":     {11, 12, 13, 15, 84, 74},
    "vehicle":      {17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 32, 60, 61},
    "rail":         {33, 34, 35, 36, 37, 38},
    "maritime":     {40, 41, 42, 44, 45, 47, 49, 50, 51, 52},
    "construction": {53, 54, 55, 56, 57, 59, 62, 63, 64, 65, 66, 79, 89, 91},
    "structure":    {71, 72, 73, 76, 77, 83, 86, 93, 94},
}


def class_supercategory(class_id: int) -> str:
    for sc, ids in SUPERCATEGORY.items():
        if class_id in ids:
            return sc
    return "other"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_class_map() -> dict[int, str]:
    """Load the class id -> name map. Falls back to DEFAULT_CLASS_MAP."""
    if not CLASSES_TXT.exists():
        return dict(DEFAULT_CLASS_MAP)
    out: dict[int, str] = {}
    for line in CLASSES_TXT.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: "11:Fixed-wing Aircraft"
        if ":" in line:
            cid, name = line.split(":", 1)
            try:
                out[int(cid.strip())] = name.strip()
            except ValueError:
                continue
    return out or dict(DEFAULT_CLASS_MAP)


def load_labels() -> list[dict]:
    """Stream the train_labels.geojson FeatureCollection into a list of dicts.

    Each label dict:
        {image_id, class_id, class_name, supercategory, bbox: [x_min, y_min, x_max, y_max]}
    """
    if not LABELS_PATH.exists():
        logger.warning(
            "[xview] %s not found. Download xView from http://xviewdataset.org "
            "and unpack to %s",
            LABELS_PATH, OUT_DIR,
        )
        return []

    cmap = load_class_map()
    out: list[dict] = []
    try:
        data = json.loads(LABELS_PATH.read_text())
    except Exception as e:
        logger.error("[xview] Could not parse %s: %s", LABELS_PATH, e)
        return []

    features = data.get("features") or data
    if not isinstance(features, list):
        logger.error("[xview] Unexpected GeoJSON shape (no 'features' list)")
        return []

    for feat in features:
        props = feat.get("properties") or feat
        try:
            class_id = int(props.get("type_id") or props.get("class_id") or 0)
            image_id = str(props.get("image_id") or props.get("img_id") or "")
            bbox_str = props.get("bounds_imcoords") or props.get("bbox") or ""
            if isinstance(bbox_str, str) and bbox_str:
                parts = [float(x) for x in bbox_str.split(",")]
                if len(parts) == 4:
                    bbox = parts
                else:
                    continue
            elif isinstance(bbox_str, list) and len(bbox_str) == 4:
                bbox = [float(x) for x in bbox_str]
            else:
                continue
        except (ValueError, TypeError):
            continue
        if class_id not in cmap or not image_id:
            continue
        out.append({
            "image_id":    image_id,
            "class_id":    class_id,
            "class_name":  cmap[class_id],
            "supercategory": class_supercategory(class_id),
            "bbox":        bbox,   # [x_min, y_min, x_max, y_max] in image px
        })

    logger.info("[xview] Loaded %d annotations across %d classes",
                len(out), len({d["class_id"] for d in out}))
    return out


def iter_labels_by_image() -> Iterator[tuple[str, list[dict]]]:
    """Yield (image_id, list_of_labels) for each image in the manifest."""
    by_image: dict[str, list[dict]] = {}
    for d in load_labels():
        by_image.setdefault(d["image_id"], []).append(d)
    for image_id, labels in by_image.items():
        yield image_id, labels


def load_as_training_samples(data_dir: Path = OUT_DIR) -> list[dict]:
    """Pure dict view, compatible with packages/ml/train_visual_detector.py."""
    samples: list[dict] = []
    for d in load_labels():
        samples.append({
            "image_id":      d["image_id"],
            "bbox":          d["bbox"],          # [x_min, y_min, x_max, y_max]
            "category_id":   d["class_id"],
            "category_name": d["class_name"],
            "supercategory": d["supercategory"],
            "source":        "xview1_real",
        })
    return samples
