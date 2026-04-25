"""
NASA FIRMS (Fire Information for Resource Management System)
Active fire / thermal anomaly dataset for wildfire detection training.

Data: MODIS and VIIRS active fire detections — public, updated daily.
URL: https://firms.modaps.eosdis.nasa.gov/api/
License: Public domain (US government data)

Used to train: wildfire detection model and fire classification in the
aerial detection pipeline.
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

FIRMS_API_KEY = os.getenv("FIRMS_API_KEY", "")
OUT_DIR = Path(__file__).parent.parent / "data" / "firms"

# VIIRS SNPP — 375m resolution, best for active fire
FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/VIIRS_SNPP_NRT/{bbox}/1"

# Public CSV endpoints — no API key required (MODIS 7-day global, updated daily)
FIRMS_MODIS_URL = (
    "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/"
    "MODIS_C6_1_Global_7d.csv"
)
# VIIRS NOAA-20 (higher resolution, but zip)
FIRMS_ARCHIVE_URL = (
    "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/"
    "J1_VIIRS_C2_Global_7d.zip"
)


def download(
    bbox: str = "-180,-90,180,90",
    api_key: Optional[str] = None,
    out_dir: Path = OUT_DIR,
) -> Path:
    """
    Download VIIRS active fire detections from NASA FIRMS.

    Args:
        bbox: "west,south,east,north" bounding box (default: global)
        api_key: FIRMS MAP KEY (register free at firms.modaps.eosdis.nasa.gov)
        out_dir: Output directory

    Returns:
        Path to downloaded CSV file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "viirs_active_fire_7d.csv"

    key = api_key or FIRMS_API_KEY

    if key:
        url = FIRMS_URL.format(api_key=key, bbox=bbox)
        logger.info("[FIRMS] Downloading active fire data from API...")
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            out_path.write_text(resp.text)
            rows = resp.text.count("\n") - 1
            logger.info("[FIRMS] Downloaded %d fire detections → %s", rows, out_path)
            return out_path
        except Exception as e:
            logger.warning("[FIRMS] API download failed: %s — trying archive", e)

    # Primary fallback: MODIS 7-day global CSV (no key, no zip)
    logger.info("[FIRMS] Downloading MODIS 7-day global CSV (no API key required)...")
    try:
        resp = requests.get(FIRMS_MODIS_URL, timeout=60,
                            headers={"User-Agent": "Heli.OS/1.0"})
        resp.raise_for_status()
        out_path.write_text(resp.text)
        rows = resp.text.count("\n") - 1
        logger.info("[FIRMS] MODIS CSV downloaded: %d fire detections → %s", rows, out_path)
        return out_path
    except Exception as e:
        logger.warning("[FIRMS] MODIS download failed: %s — trying VIIRS archive", e)

    # Secondary fallback: VIIRS zip
    import zipfile
    zip_path = out_dir / "J1_VIIRS_7d.zip"
    try:
        resp = requests.get(FIRMS_ARCHIVE_URL, timeout=120, stream=True)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        with zipfile.ZipFile(zip_path) as zf:
            csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
            zf.extract(csv_name, out_dir)
            (out_dir / csv_name).rename(out_path)
        zip_path.unlink(missing_ok=True)
        logger.info("[FIRMS] VIIRS archive downloaded → %s", out_path)
    except Exception as e:
        logger.warning("[FIRMS] Archive download failed: %s — generating synthetic", e)
        _generate_synthetic(out_path)

    return out_path


def load_as_training_samples(csv_path: Path) -> list[dict]:
    """
    Parse FIRMS CSV into training samples for the fire/smoke detection model.

    Each sample represents one active fire detection with:
      - lat, lon: coordinates
      - frp: Fire Radiative Power (MW) — proxy for fire intensity
      - confidence: detection confidence (nominal/high/low)
      - label: 'fire' (class index 2 in our detection model)
    """
    samples = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    frp = float(row.get("frp", 0) or 0)
                    conf = str(row.get("confidence", "nominal")).lower()
                    samples.append({
                        "lat": float(row["latitude"]),
                        "lon": float(row["longitude"]),
                        "frp_mw": frp,
                        "confidence": conf,
                        "daynight": row.get("daynight", "D"),
                        "label": "fire",
                        "class_idx": 2,
                    })
                except (KeyError, ValueError):
                    continue
    except Exception as e:
        logger.error("[FIRMS] Failed to parse %s: %s", csv_path, e)

    logger.info("[FIRMS] Loaded %d fire samples", len(samples))
    return samples


def _generate_synthetic(out_path: Path) -> None:
    """Generate realistic synthetic fire detections as fallback."""
    import random

    random.seed(42)
    rows = ["latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight"]
    fire_regions = [
        (36.7, -119.4),  # California
        (40.0, -123.0),  # Northern CA
        (-33.8, 150.9),  # Australia
        (55.0, 82.0),    # Siberia
        (-15.0, -55.0),  # Brazil
        (0.0, 25.0),     # Central Africa
    ]
    for _ in range(2000):
        center = random.choice(fire_regions)
        lat = center[0] + random.uniform(-3, 3)
        lon = center[1] + random.uniform(-3, 3)
        frp = random.uniform(5, 500)
        brightness = random.uniform(310, 420)
        conf = random.choice(["nominal", "high", "high", "high"])
        rows.append(
            f"{lat:.4f},{lon:.4f},{brightness:.1f},1.0,1.0,"
            f"2026-01-01,1200,NOAA-20,VIIRS,{conf},2.0NRT,{brightness - 20:.1f},{frp:.1f},D"
        )
    out_path.write_text("\n".join(rows))
    logger.info("[FIRMS] Generated %d synthetic fire samples", len(rows) - 1)
