"""
HERIDAL Dataset Downloader
===========================
Human dEtection in Aerial Images with Deep Learning.
Person detection in SAR aerial imagery.

HERIDAL class → Heli.OS class:
  0: person → 0: person

Source: http://ipsar.fesb.unist.hr/HERIDAL%20database.html
License: Academic/research use
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

_HERIDAL_URL = "http://ipsar.fesb.unist.hr/HERIDAL/HERIDAL_database.zip"

# HERIDAL uses Pascal VOC XML annotations — need conversion
_VOC_CLASS = "person"


def _download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("Already downloaded: %s", dest.name)
        return dest
    logger.info("Downloading HERIDAL dataset ...")
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="HERIDAL") as bar:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))
    return dest


def _voc_xml_to_yolo(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    """Convert Pascal VOC XML to YOLO format lines."""
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return []

    lines = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "")
        if name.lower() != _VOC_CLASS:
            continue
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = float(bndbox.findtext("xmin", "0"))
        ymin = float(bndbox.findtext("ymin", "0"))
        xmax = float(bndbox.findtext("xmax", "0"))
        ymax = float(bndbox.findtext("ymax", "0"))

        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        w  = (xmax - xmin) / img_w
        h  = (ymax - ymin) / img_h
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def download(output_dir: str = "/tmp/heli-training-data/heridal") -> Path:
    out = Path(output_dir)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    zip_path = raw / "heridal.zip"
    try:
        _download_file(_HERIDAL_URL, zip_path)
    except Exception as e:
        logger.warning("HERIDAL primary download failed (%s). Trying alternate ...", e)
        # Fallback: skip HERIDAL if unavailable (VisDrone covers person detection)
        logger.warning("HERIDAL unavailable — skipping. VisDrone pedestrian data will be used.")
        return out

    extract_dir = raw / "extracted"
    if not extract_dir.exists():
        logger.info("Extracting HERIDAL ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

    # HERIDAL structure varies by release — walk to find images + annotations
    from PIL import Image

    img_dirs = list(extract_dir.rglob("*.JPG"))[:1]  # sample to find img dir
    if not img_dirs:
        img_dirs = list(extract_dir.rglob("*.jpg"))[:1]

    if not img_dirs:
        logger.warning("No images found in HERIDAL archive")
        return out

    img_root = img_dirs[0].parent

    converted = 0
    for img_path in img_root.glob("*.JPG"):
        xml_path = img_path.with_suffix(".xml")
        if not xml_path.exists():
            xml_path = img_path.parent / "Annotations" / img_path.with_suffix(".xml").name
        if not xml_path.exists():
            continue

        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception:
            continue

        yolo_lines = _voc_xml_to_yolo(xml_path, W, H)
        if not yolo_lines:
            continue

        # All HERIDAL data goes to train (small dataset, split manually)
        out_img_dir   = out / "images" / "train"
        out_label_dir = out / "labels" / "train"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(img_path, out_img_dir / img_path.name)
        (out_label_dir / img_path.with_suffix(".txt").name).write_text("\n".join(yolo_lines))
        converted += 1

    logger.info("HERIDAL: %d images converted", converted)
    logger.info("HERIDAL dataset ready at %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/tmp/heli-training-data/heridal")
    args = p.parse_args()
    download(args.output)
