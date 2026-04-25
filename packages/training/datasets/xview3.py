"""
xView3 — Maritime Object Detection in SAR Imagery
=====================================================
SpaceNet/xView3-SAR challenge dataset: ~1,000 Sentinel-1 GRD scenes
globally with vessel labels (length, position, vessel class, fishing
attribute, dark-vessel flag).

Source:    https://iuu.xview.us/dataset
License:   xView3 dataset license (research, free).
Auth:      Account required; download via the iuu.xview.us portal then
           drop the manifest + chips at packages/training/data/xview3/.

Why this dataset:
  - Real Sentinel-1 SAR (which the Sentinel Hub adapter can also stream
    on demand for inference at deployment time)
  - Real vessel labels including DARK-VESSEL flag — exactly the federal
    Maritime Domain Awareness (MDA) capability we want
  - Civilian USCG / illegal-fishing use cases AND federal MDA / counter-
    smuggling AND HADR ship-tracking — full dual-market

Expected on-disk layout after manual download:
  packages/training/data/xview3/
    train.csv               Manifest of all training labels
    validation.csv          Validation split labels
    chips/
      <scene_id>/
        VV_dB.tif           VV polarization, dB scale
        VH_dB.tif           VH polarization, dB scale
        owiMask.tif         Wind/inference quality mask
        bathymetry.tif      Sea depth raster
        ...

Manifest CSV columns (xView3 spec):
  scene_id, detect_id, detect_lat, detect_lon, detect_scene_row,
  detect_scene_column, vessel_length_m, source, confidence,
  is_vessel, is_fishing, is_dark, ...

Usage:
    from packages.training.datasets.xview3 import (
        load_manifest, iter_chips_with_labels,
    )
    samples = load_manifest()  # returns list of label dicts
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

OUT_DIR        = Path(__file__).parent.parent / "data" / "xview3"
MANIFEST_TRAIN = OUT_DIR / "train.csv"
MANIFEST_VAL   = OUT_DIR / "validation.csv"
CHIPS_DIR      = OUT_DIR / "chips"
ANNOTATIONS    = OUT_DIR / "annotations.json"


# ---------------------------------------------------------------------------
# Sample model
# ---------------------------------------------------------------------------


@dataclass
class XView3Detection:
    scene_id:        str
    detect_id:       str
    lat:             float
    lon:             float
    row:             int          # pixel coordinates within scene chip
    col:             int
    vessel_length_m: float        # 0 if non-vessel detection
    is_vessel:       bool
    is_fishing:      bool
    is_dark:         bool         # dark-vessel flag (no AIS broadcast)
    confidence:      str          # "HIGH" | "MEDIUM" | "LOW"
    source:          str          # "AIS_MATCHED" | "PUBLIC" | etc.

    @property
    def class_label(self) -> str:
        """Multi-class vessel label for training."""
        if not self.is_vessel:
            return "non_vessel"
        if self.is_dark:
            return "dark_vessel" if not self.is_fishing else "dark_fishing_vessel"
        if self.is_fishing:
            return "fishing_vessel"
        if self.vessel_length_m >= 100.0:
            return "large_vessel"
        if self.vessel_length_m >= 25.0:
            return "medium_vessel"
        return "small_vessel"


# ---------------------------------------------------------------------------
# Manifest parsing
# ---------------------------------------------------------------------------


def _truthy(s: str) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes", "y")


def _parse_row(row: dict) -> Optional[XView3Detection]:
    try:
        return XView3Detection(
            scene_id=        row.get("scene_id", "").strip(),
            detect_id=       row.get("detect_id", "").strip()
                              or f"{row.get('scene_id','')}-{row.get('detect_scene_row','0')}-{row.get('detect_scene_column','0')}",
            lat=             float(row.get("detect_lat") or 0),
            lon=             float(row.get("detect_lon") or 0),
            row=             int(float(row.get("detect_scene_row") or 0)),
            col=             int(float(row.get("detect_scene_column") or 0)),
            vessel_length_m= float(row.get("vessel_length_m") or 0),
            is_vessel=       _truthy(row.get("is_vessel", "")),
            is_fishing=      _truthy(row.get("is_fishing", "")),
            is_dark=         _truthy(row.get("is_dark", "")),
            confidence=      (row.get("confidence", "") or "MEDIUM").strip().upper(),
            source=          (row.get("source", "") or "PUBLIC").strip().upper(),
        )
    except (KeyError, ValueError, TypeError):
        return None


def load_manifest(split: str = "train") -> list[XView3Detection]:
    """Load detection labels from the train or validation manifest CSV."""
    if split == "train":
        path = MANIFEST_TRAIN
    elif split in ("val", "validation"):
        path = MANIFEST_VAL
    else:
        raise ValueError(f"split must be 'train' or 'validation', got {split!r}")

    if not path.exists():
        logger.warning(
            "[xview3] Manifest not found at %s. Download the dataset from "
            "https://iuu.xview.us/dataset and place it under %s",
            path, OUT_DIR,
        )
        return []

    out: list[XView3Detection] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            det = _parse_row(row)
            if det is not None and det.scene_id:
                out.append(det)
    logger.info("[xview3] Loaded %d detections from %s split", len(out), split)
    return out


def list_scenes(split: str = "train") -> list[str]:
    """Return unique scene_ids present in the manifest."""
    return sorted({d.scene_id for d in load_manifest(split)})


def iter_chips_with_labels(split: str = "train") -> Iterator[tuple[Path, list[XView3Detection]]]:
    """Yield (scene_chip_dir, list_of_detections) for each scene present
    on disk. Scenes referenced in the manifest but missing on-disk are
    skipped with a warning."""
    by_scene: dict[str, list[XView3Detection]] = {}
    for det in load_manifest(split):
        by_scene.setdefault(det.scene_id, []).append(det)

    for scene_id, dets in by_scene.items():
        chip_dir = CHIPS_DIR / scene_id
        if not chip_dir.exists():
            logger.warning("[xview3] Chip dir missing for %s: %s", scene_id, chip_dir)
            continue
        yield chip_dir, dets


def load_as_training_samples(data_dir: Path = OUT_DIR) -> list[dict]:
    """Pure dict view of detections — usable from train_visual_detector.py."""
    samples: list[dict] = []
    for det in load_manifest("train"):
        samples.append({
            "image_id":      det.scene_id,
            "detect_id":     det.detect_id,
            "lat":           det.lat,
            "lon":           det.lon,
            "row":           det.row,
            "col":           det.col,
            "vessel_class":  det.class_label,
            "is_vessel":     det.is_vessel,
            "is_fishing":    det.is_fishing,
            "is_dark":       det.is_dark,
            "confidence":    det.confidence,
            "vessel_length_m": det.vessel_length_m,
            "sar_band":      "C",     # Sentinel-1 C-band
            "source":        f"xview3_{det.source.lower()}",
        })
    logger.info("[xview3] Built %d training samples", len(samples))
    return samples
