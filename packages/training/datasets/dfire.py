"""
D-Fire Dataset Downloader
==========================
Downloads the D-Fire dataset for smoke and fire detection.

D-Fire classes → Heli.OS classes:
  0: fire   → 2: fire
  1: smoke  → 3: smoke

Source: https://github.com/gaiasd/DFireDataset
License: CC BY 4.0
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

# D-Fire releases on GitHub
_DFIRE_URL = "https://github.com/gaiasd/DFireDataset/archive/refs/heads/master.zip"

# D-Fire class → Heli.OS class offset (fire=2, smoke=3)
_CLASS_OFFSET = 2  # D-Fire 0→heli 2, D-Fire 1→heli 3


def _download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("Already downloaded: %s", dest.name)
        return dest
    logger.info("Downloading D-Fire dataset (~1.5 GB) ...")
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="D-Fire") as bar:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))
    return dest


def _remap_labels(src_label_dir: Path, out_label_dir: Path) -> int:
    """Remap D-Fire class indices to Heli.OS class indices."""
    out_label_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for lf in src_label_dir.glob("*.txt"):
        lines = lf.read_text().strip().split("\n")
        remapped = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0]) + _CLASS_OFFSET
            remapped.append(f"{cls} {' '.join(parts[1:])}")
        if remapped:
            (out_label_dir / lf.name).write_text("\n".join(remapped))
            count += 1
    return count


def download(output_dir: str = "/tmp/heli-training-data/dfire") -> Path:
    out = Path(output_dir)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    zip_path = raw / "dfire.zip"
    _download_file(_DFIRE_URL, zip_path)

    extract_dir = raw / "extracted"
    if not extract_dir.exists():
        logger.info("Extracting D-Fire ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

    # D-Fire structure: DFireDataset-master/train/, DFireDataset-master/test/
    src_root = extract_dir / "DFireDataset-master"
    if not src_root.exists():
        # Try finding it
        candidates = list(extract_dir.iterdir())
        if candidates:
            src_root = candidates[0]

    for split in ("train", "test"):
        src_split = src_root / split
        if not src_split.exists():
            logger.warning("D-Fire split not found: %s", src_split)
            continue

        # D-Fire uses images/ and labels/ subdirs
        src_imgs   = src_split / "images"
        src_labels = src_split / "labels"

        if not src_imgs.exists() or not src_labels.exists():
            logger.warning("D-Fire missing images/labels in %s", src_split)
            continue

        out_split = "val" if split == "test" else split
        out_img_dir   = out / "images" / out_split
        out_label_dir = out / "labels" / out_split

        out_img_dir.mkdir(parents=True, exist_ok=True)
        for img in src_imgs.glob("*"):
            shutil.copy(img, out_img_dir / img.name)

        n = _remap_labels(src_labels, out_label_dir)
        logger.info("D-Fire %s: %d label files remapped", split, n)

    logger.info("D-Fire dataset ready at %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/tmp/heli-training-data/dfire")
    args = p.parse_args()
    download(args.output)
