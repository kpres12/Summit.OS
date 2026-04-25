"""
VisDrone2019-DET Dataset Downloader + YOLO Converter
======================================================
Downloads VisDrone2019 object detection dataset and converts annotations
from native format to YOLO format with Heli.OS class mapping.

VisDrone classes → Heli.OS classes:
  0: pedestrian  → 0: person
  1: people      → 0: person
  2: bicycle     → (skip)
  3: car         → 1: vehicle
  4: van         → 1: vehicle
  5: truck       → 1: vehicle
  6: tricycle    → (skip)
  7: awning-tri  → (skip)
  8: bus         → 1: vehicle
  9: motor       → (skip)

Source: https://github.com/VisDrone/VisDrone-Dataset
Official mirrors vary — uses ultralytics CDN (most reliable).
"""
from __future__ import annotations

import argparse
import logging
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ultralytics hosts a clean mirror — direct zip links
_SPLITS = {
    "train": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "val":   "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
    "test":  "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip",
}

# VisDrone class index → Heli.OS class index (None = skip)
_CLASS_MAP: dict[int, int | None] = {
    0: 0,    # pedestrian → person
    1: 0,    # people     → person
    2: None, # bicycle    → skip
    3: 1,    # car        → vehicle
    4: 1,    # van        → vehicle
    5: 1,    # truck      → vehicle
    6: None, # tricycle   → skip
    7: None, # awning-tri → skip
    8: 1,    # bus        → vehicle
    9: None, # motor      → skip
}


def _download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("Already downloaded: %s", dest.name)
        return dest
    logger.info("Downloading %s ...", url)
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    return dest


def _convert_annotations(src_ann_dir: Path, src_img_dir: Path,
                          out_label_dir: Path, out_img_dir: Path) -> int:
    """Convert VisDrone annotation format to YOLO format."""
    out_label_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    converted = 0

    for ann_file in src_ann_dir.glob("*.txt"):
        img_file = src_img_dir / ann_file.with_suffix(".jpg").name
        if not img_file.exists():
            continue

        lines = ann_file.read_text().strip().split("\n")
        yolo_lines = []

        # Read image dimensions
        try:
            from PIL import Image
            with Image.open(img_file) as im:
                W, H = im.size
        except Exception:
            continue

        for line in lines:
            if not line.strip():
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            score = int(parts[4])  # 0 = ignored
            cat = int(parts[5])

            if score == 0 or w <= 0 or h <= 0:
                continue
            heli_cls = _CLASS_MAP.get(cat)
            if heli_cls is None:
                continue

            # Convert to YOLO normalized xywh
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            yolo_lines.append(f"{heli_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if yolo_lines:
            (out_label_dir / ann_file.name).write_text("\n".join(yolo_lines))
            shutil.copy(img_file, out_img_dir / img_file.name)
            converted += 1

    return converted


def download(output_dir: str = "/tmp/heli-training-data/visdrone") -> Path:
    out = Path(output_dir)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    for split, url in _SPLITS.items():
        zip_path = raw / f"VisDrone2019-DET-{split}.zip"
        _download_file(url, zip_path)

        extract_dir = raw / split
        if not extract_dir.exists():
            logger.info("Extracting %s ...", zip_path.name)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(raw)

        # VisDrone zip extracts to VisDrone2019-DET-{split}/
        src_root = raw / f"VisDrone2019-DET-{split}"
        if not src_root.exists():
            # Some zips extract flat
            src_root = raw / split

        src_imgs = src_root / "images"
        src_anns = src_root / "annotations"

        if not src_imgs.exists() or not src_anns.exists():
            logger.warning("Expected images/annotations dirs not found in %s", src_root)
            continue

        n = _convert_annotations(
            src_anns, src_imgs,
            out / "labels" / split,
            out / "images" / split,
        )
        logger.info("VisDrone %s: %d images converted", split, n)

    logger.info("VisDrone dataset ready at %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/tmp/heli-training-data/visdrone")
    args = p.parse_args()
    download(args.output)
