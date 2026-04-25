"""
EuroSAT — Sentinel-2 Land Use / Land Cover Classification Dataset
=====================================================================
Real Sentinel-2 RGB / multispectral patches across 10 land-use classes,
27,000 labeled images. **Direct HTTPS download, no auth, no portal.**

Sources (mirror as needed):
  RGB:  https://madm.dfki.de/files/sentinel/EuroSAT.zip          (~90 MB)
  MS:   https://madm.dfki.de/files/sentinel/EuroSATallBands.zip  (~2.3 GB)

Why this matters:
  - Real Sentinel-2 imagery without needing Sentinel Hub or CDSE creds
  - Pretrain backbone for Heli.OS vision models (transfer learning)
  - Benchmark + sanity check for any new EO model we wire up
  - Land-cover context layer for civilian operations (forestry, ag)

Classes:
  AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial,
  Pasture, PermanentCrop, Residential, River, SeaLake

Usage:
    from packages.training.datasets.eurosat import (
        ensure_downloaded, load_eurosat_rgb,
    )
    ensure_downloaded()
    X, y, classes = load_eurosat_rgb()
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)

OUT_DIR     = Path(__file__).parent.parent / "data" / "eurosat"
RGB_MIRRORS = [
    "https://huggingface.co/datasets/blanchon/EuroSAT_RGB/resolve/main/EuroSAT_RGB.zip",
    "https://zenodo.org/record/7711810/files/EuroSAT_RGB.zip",
    "https://madm.dfki.de/files/sentinel/EuroSAT.zip",
]
MS_MIRRORS = [
    "https://huggingface.co/datasets/blanchon/EuroSAT_MSI/resolve/main/EuroSAT_MSI.zip",
    "https://zenodo.org/record/7711810/files/EuroSAT_MS.zip",
    "https://madm.dfki.de/files/sentinel/EuroSATallBands.zip",
]
RGB_ROOT    = OUT_DIR / "2750"   # extracted top-level dir for RGB
MS_ROOT     = OUT_DIR / "ds"     # extracted top-level dir for multispectral

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]


def ensure_downloaded(variant: str = "rgb", force: bool = False) -> Path:
    """Download + extract EuroSAT into OUT_DIR. Returns root extraction dir."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if variant == "rgb":
        urls, root = RGB_MIRRORS, RGB_ROOT
    elif variant in ("ms", "multispectral", "all_bands"):
        urls, root = MS_MIRRORS, MS_ROOT
    else:
        raise ValueError(f"variant must be 'rgb' or 'ms', got {variant!r}")

    if root.exists() and not force and (any(root.rglob("*.jpg")) or any(root.rglob("*.tif"))):
        return root

    zip_bytes: bytes = b""
    for url in urls:
        logger.info("[eurosat] trying mirror %s (variant=%s)", url, variant)
        for verify in (True, False):
            try:
                r = requests.get(url, stream=True, timeout=600, verify=verify,
                                 headers={"User-Agent": "Heli.OS/1.0"})
                r.raise_for_status()
                zip_bytes = r.content
                if not verify:
                    logger.warning("[eurosat] used verify=False fallback for %s", url)
                break
            except Exception as e:
                logger.warning("[eurosat] mirror %s (verify=%s) failed: %s", url, verify, e)
        if zip_bytes:
            break

    if not zip_bytes:
        logger.error("[eurosat] all mirrors failed")
        return root

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(OUT_DIR)
        # HF / Zenodo may unpack to a top-level dir different from "2750".
        # Find a class subdir and use its parent as root.
        for cand in OUT_DIR.rglob("AnnualCrop"):
            if cand.is_dir():
                detected_root = cand.parent
                if detected_root != root:
                    logger.info("[eurosat] detected dataset root at %s", detected_root)
                    return detected_root
                break
    except Exception as e:
        logger.error("[eurosat] extract failed: %s", e)
        return root

    logger.info("[eurosat] extracted -> %s", OUT_DIR)
    return root


def load_eurosat_rgb(root: Optional[Path] = None
                     ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load EuroSAT RGB (64×64 JPGs) into (X, y, class_names).

    X: (N, 3, 64, 64) float32 in [0, 1]
    y: (N,) int64 class indices
    """
    try:
        from PIL import Image
    except ImportError:
        logger.error("[eurosat] Pillow required — pip install Pillow")
        return np.zeros((0, 3, 64, 64), dtype=np.float32), np.zeros(0, dtype=np.int64), CLASSES

    base = root or RGB_ROOT
    if not base.exists():
        base = ensure_downloaded("rgb")
    cls_to_idx = {c: i for i, c in enumerate(CLASSES)}

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    for cls_dir in sorted(base.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_idx = cls_to_idx.get(cls_dir.name)
        if cls_idx is None:
            continue
        for img_path in cls_dir.glob("*.jpg"):
            try:
                im = Image.open(img_path).convert("RGB").resize((64, 64))
                arr = np.asarray(im, dtype=np.float32) / 255.0
                X_list.append(np.transpose(arr, (2, 0, 1)))
                y_list.append(cls_idx)
            except Exception:
                continue
    if not X_list:
        return np.zeros((0, 3, 64, 64), dtype=np.float32), np.zeros(0, dtype=np.int64), CLASSES
    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    logger.info("[eurosat] loaded %d images × %d classes",
                len(X), len(set(y.tolist())))
    return X, y, CLASSES
