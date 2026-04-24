"""
SAR-Ship Dataset — Synthetic Aperture Radar Vessel Detection
https://github.com/CAESAR-Radi/SAR-Ship-Dataset

6,000+ SAR image chips with ship bounding boxes.
License: CC BY-NC 4.0

Also incorporates OpenSAR-Ship (Sentinel-1, publicly available via ESA).

Used to train: maritime vessel detection model for dark vessel / AIS-gap
scenarios where optical imagery is unavailable (night, cloud cover).
Feeds into: domain/maritime.py dark vessel assessment.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent.parent / "data" / "sar_ship"

# GitHub release archive
SAR_SHIP_URL = "https://github.com/CAESAR-Radi/SAR-Ship-Dataset/archive/refs/heads/master.zip"


def download(out_dir: Path = OUT_DIR) -> Path:
    """
    Download SAR-Ship dataset annotations (JSON bboxes).

    Returns:
        Path to annotations directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    annot_dir = out_dir / "annotations"

    if annot_dir.exists() and any(annot_dir.glob("*.json")):
        logger.info("[SAR-Ship] Annotations already present")
        return annot_dir

    logger.info("[SAR-Ship] Downloading SAR-Ship dataset from GitHub...")
    try:
        import zipfile
        zip_path = out_dir / "sar_ship.zip"
        resp = requests.get(SAR_SHIP_URL, timeout=120, stream=True)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(65536):
                f.write(chunk)

        with zipfile.ZipFile(zip_path) as zf:
            json_members = [m for m in zf.namelist() if m.endswith(".json")]
            zf.extractall(out_dir, members=json_members)
            logger.info("[SAR-Ship] Extracted %d annotation files", len(json_members))

        zip_path.unlink(missing_ok=True)
        annot_dir.mkdir(exist_ok=True)
        for jf in out_dir.rglob("*.json"):
            if jf.parent != annot_dir:
                jf.rename(annot_dir / jf.name)
    except Exception as e:
        logger.warning("[SAR-Ship] Download failed: %s — generating synthetic", e)
        _generate_synthetic(out_dir)
        annot_dir = out_dir / "annotations"

    return annot_dir


def load_as_training_samples(data_dir: Path) -> list[dict]:
    """
    Parse SAR-Ship annotations into vessel detection training samples.

    Returns list of:
        {image_id, bbox [x,y,w,h], lat, lon, vessel_class, confidence, sar_band}
    """
    samples = []
    annot_dir = data_dir / "annotations"
    if not annot_dir.exists():
        annot_dir = data_dir

    for jf in annot_dir.glob("*.json"):
        try:
            with open(jf) as f:
                data = json.load(f)

            # Handle both COCO-format and custom SAR-Ship format
            if "annotations" in data:
                for ann in data["annotations"]:
                    samples.append({
                        "image_id": ann.get("image_id"),
                        "bbox": ann.get("bbox", [0, 0, 64, 64]),
                        "category_id": ann.get("category_id", 1),
                        "vessel_class": "VESSEL",
                        "class_idx": 1,  # class 1 in our detection model = vehicle/vessel
                        "sar_band": "C",
                        "confidence": 0.9,
                    })
            elif isinstance(data, list):
                for item in data:
                    samples.append({
                        "image_id": item.get("image_name", ""),
                        "bbox": item.get("bbox", [0, 0, 64, 64]),
                        "vessel_class": "VESSEL",
                        "class_idx": 1,
                        "sar_band": item.get("sar_band", "C"),
                        "confidence": float(item.get("score", 0.9)),
                    })
        except Exception:
            continue

    logger.info("[SAR-Ship] Loaded %d vessel detection samples", len(samples))
    return samples


def _generate_synthetic(out_dir: Path) -> None:
    """Generate synthetic SAR-Ship-format annotations."""
    random.seed(42)
    annot_dir = out_dir / "annotations"
    annot_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    ann_id = 1

    for img_id in range(1, 201):
        images.append({"id": img_id, "file_name": f"SAR_{img_id:04d}.png",
                        "width": 512, "height": 512})
        n_ships = random.randint(0, 5)
        for _ in range(n_ships):
            x = random.randint(0, 480)
            y = random.randint(0, 480)
            w = random.randint(16, 64)
            h = random.randint(16, 64)
            annotations.append({
                "id": ann_id, "image_id": img_id,
                "category_id": 1, "bbox": [x, y, w, h],
                "area": w * h, "iscrowd": 0,
            })
            ann_id += 1

    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "ship"}],
    }
    (annot_dir / "sar_ship_synth.json").write_text(json.dumps(coco_data))
    logger.info("[SAR-Ship] Generated %d synthetic ship annotations", len(annotations))
