"""
NASA SMAP/MSL Telemetry Anomaly Dataset Downloader
====================================================
Downloads the Soil Moisture Active Passive (SMAP) and Mars Science
Laboratory (MSL) telemetry anomaly datasets used in NASA's Telemanom paper.

These multivariate time-series datasets represent spacecraft telemetry with
labeled anomaly windows — a good proxy for drone/robot sensor anomalies.

Source: https://github.com/khundman/telemanom
Data: s3://telemanom-data (public, no auth)
License: NASA Open Data
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

_BASE_URL = "https://s3-us-west-2.amazonaws.com/telemanom-data"
_LABELS_URL = f"{_BASE_URL}/labeled_anomalies.csv"

# SMAP has 55 channels; MSL has 27. Download all.
_SMAP_CHANNELS = [
    "P-1","S-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7",
    "A-1","D-1","P-2","P-3","D-2","D-3","D-4","A-2","A-3",
    "A-4","G-1","C-1","C-2","C-3","D-5","D-6","D-7","F-1",
    "P-4","G-2","T-1","T-2","D-8","D-9","F-2","M-6","M-1",
    "M-2","S-2","P-5","P-6","P-7","R-1","A-5","A-6","A-7",
    "D-11","D-12","B-1","G-3","G-4","D-13","A-8","A-9","F-3",
    "M-3","M-4","M-5","D-14","D-15","D-16",
]
_MSL_CHANNELS = [
    "C-1","C-2","D-14","D-15","D-16","D-17","F-4","F-5","F-7","F-8",
    "M-6","M-7","P-10","P-11","P-14","P-15","S-2","T-3","T-4",
    "T-5","T-8","T-9","T-12","T-13","UCR-1","UCR-2","UCR-3",
]


def _download(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as e:
        logger.debug("Download failed %s: %s", url, e)
        return False


def download(output_dir: str = "/tmp/heli-training-data/smap_msl") -> Path:
    out = Path(output_dir)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)

    # Download anomaly labels
    labels_path = out / "labeled_anomalies.csv"
    logger.info("Downloading anomaly labels ...")
    _download(_LABELS_URL, labels_path)

    # Download train/test numpy arrays
    all_channels = _SMAP_CHANNELS + _MSL_CHANNELS
    logger.info("Downloading %d channels (SMAP + MSL) ...", len(all_channels))

    downloaded = 0
    for ch in tqdm(all_channels, desc="SMAP/MSL channels"):
        for split in ("train", "test"):
            url  = f"{_BASE_URL}/{split}/{ch}.npy"
            dest = out / split / f"{ch}.npy"
            if _download(url, dest):
                downloaded += 1

    logger.info("Downloaded %d channel splits", downloaded)
    logger.info("SMAP/MSL dataset ready at %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/tmp/heli-training-data/smap_msl")
    args = p.parse_args()
    download(args.output)
