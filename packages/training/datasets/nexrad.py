"""
NEXRAD Level-II Doppler Weather Radar Loader (AWS Open Data)
================================================================
NOAA's WSR-88D radar network — 160+ radars across CONUS, Alaska, Hawaii,
Guam, Puerto Rico — publishes Level-II base data to a public AWS S3
bucket every ~5 minutes. **Completely no auth, no rate limit.**

Bucket: s3://noaa-nexrad-level2/ (us-east-1, requester-pays = false)
Layout: <YYYY>/<MM>/<DD>/<STATION>/<STATION><YYYYMMDD>_<HHMMSS>_V06

Each scan is a binary Level-II archive (typically 1-10 MB depending on
mode). Decoded with pyart, MetPy, or pure-Python NEXRAD parsers.

Why this matters:
  - Tornado nowcasting (sequence-to-sequence on radar mosaics)
  - Severe storm precursor detection
  - Hail / mesocyclone signatures
  - Same data feed the National Weather Service uses operationally

Usage:
    from packages.training.datasets.nexrad import (
        list_available_scans, download_scan,
    )
    scans = list_available_scans("KOUN", date(2024, 5, 7))  # Oklahoma City
    local = download_scan(scans[0], dest_dir="data/nexrad/KOUN/")
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional
from urllib.parse import quote
import xml.etree.ElementTree as ET

import requests

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent.parent / "data" / "nexrad"
NEXRAD_S3_BASE = "https://noaa-nexrad-level2.s3.amazonaws.com"

# Useful station shortlist (high-tornado-risk / coverage representative)
TORNADO_ALLEY_STATIONS = [
    "KOUN", "KAMA", "KFWS", "KICT", "KTLX", "KDDC", "KINX", "KSRX",
    "KFDR", "KGLD", "KLNX", "KOAX", "KTWX", "KUEX", "KVNX",
]


def list_available_scans(station: str, on: date) -> list[str]:
    """List S3 keys for one station on one UTC date.

    Returns:
        List of S3 keys (e.g. ['2024/05/07/KOUN/KOUN20240507_001234_V06']).
    """
    prefix = f"{on.year:04d}/{on.month:02d}/{on.day:02d}/{station.upper()}/"
    url = f"{NEXRAD_S3_BASE}/?prefix={quote(prefix)}&list-type=2"
    keys: list[str] = []
    cont_token: Optional[str] = None
    while True:
        list_url = url + (f"&continuation-token={quote(cont_token)}" if cont_token else "")
        try:
            r = requests.get(list_url, timeout=30,
                             headers={"User-Agent": "Heli.OS/1.0"})
            r.raise_for_status()
        except Exception as e:
            logger.warning("[nexrad] list failed for %s %s: %s", station, on, e)
            break
        ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
        root = ET.fromstring(r.text)
        for c in root.findall(f"{ns}Contents"):
            key_el = c.find(f"{ns}Key")
            if key_el is not None and key_el.text:
                k = key_el.text
                if k.endswith("_V06") or k.endswith(".gz"):
                    keys.append(k)
        truncated = root.find(f"{ns}IsTruncated")
        if truncated is None or truncated.text != "true":
            break
        next_tok = root.find(f"{ns}NextContinuationToken")
        cont_token = next_tok.text if next_tok is not None else None
        if not cont_token:
            break
    logger.info("[nexrad] %s on %s -> %d scans", station, on, len(keys))
    return keys


def download_scan(s3_key: str, dest_dir: Path = OUT_DIR) -> Optional[Path]:
    """Download one Level-II scan from the NOAA bucket. Returns local path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / s3_key.replace("/", "_")
    if out.exists() and out.stat().st_size > 0:
        return out
    url = f"{NEXRAD_S3_BASE}/{quote(s3_key)}"
    try:
        with requests.get(url, stream=True, timeout=120,
                          headers={"User-Agent": "Heli.OS/1.0"}) as r:
            r.raise_for_status()
            with out.open("wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
    except Exception as e:
        logger.warning("[nexrad] download failed %s: %s", s3_key, e)
        return None
    logger.debug("[nexrad] %s -> %s (%.1f KB)", s3_key, out,
                 out.stat().st_size / 1024)
    return out


def parse_scan_to_grid(local_path: Path, n_gates: int = 256,
                       n_azimuths: int = 360, sweep: int = 0
                       ) -> Optional["np.ndarray"]:  # type: ignore
    """Parse a Level-II scan into a 2D reflectivity grid using pyart if available.

    Returns array shape (n_azimuths, n_gates) of reflectivity dBZ, or None
    if pyart isn't installed (the loader still works for downloading)."""
    try:
        import pyart  # type: ignore
        import numpy as np
    except ImportError:
        logger.debug("[nexrad] pyart not installed — install to parse scans")
        return None
    try:
        radar = pyart.io.read_nexrad_archive(str(local_path))
    except Exception as e:
        logger.warning("[nexrad] read failed %s: %s", local_path, e)
        return None
    try:
        ref = radar.get_field(sweep, "reflectivity")
        # Resample to fixed grid by simple slicing
        ref = ref.filled(-32.0)  # mask -> very low dBZ
        ref = ref[:n_azimuths, :n_gates]
        if ref.shape != (n_azimuths, n_gates):
            padded = np.full((n_azimuths, n_gates), -32.0, dtype="float32")
            padded[:ref.shape[0], :ref.shape[1]] = ref
            ref = padded
        return ref.astype("float32")
    except Exception as e:
        logger.warning("[nexrad] grid extract failed %s: %s", local_path, e)
        return None
