"""
Flood Detection Dataset

Sources:
  - FloodNet Challenge 2021 (IEEE GRSS): UAV RGB imagery post-hurricane Harvey
    https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021
    (Free for research, attribution required)
  - Sen1Floods11: Sentinel-1 SAR flood mapping, 11 events
    https://github.com/cloudtostreet/Sen1Floods11 (CC BY 4.0)
  - Copernicus EMS: European flood mapping activations (open access)

Labels: binary water/no-water per pixel (segmentation) or scene-level
        water fraction for regression.

Used to train: flood extent classifier for FLOOD_SURVEY missions.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent.parent / "data" / "flood"

SEN1FLOODS_URL = (
    "https://github.com/cloudtostreet/Sen1Floods11/releases/download/v1.0/"
    "sen1floods11_hand_labeled_split.json"
)


def download(out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    split_file = out_dir / "sen1floods11_split.json"

    if split_file.exists():
        logger.info("[Flood] Data already present at %s", out_dir)
        return out_dir

    logger.info("[Flood] Attempting Sen1Floods11 split file download...")
    try:
        resp = requests.get(SEN1FLOODS_URL, timeout=30)
        resp.raise_for_status()
        split_file.write_text(resp.text)
        logger.info("[Flood] Sen1Floods11 split metadata downloaded")
    except Exception as e:
        logger.warning("[Flood] Download failed: %s — generating synthetic", e)
        _generate_synthetic(out_dir)

    return out_dir


def load_as_training_samples(data_dir: Path) -> list[dict]:
    """
    Returns samples with SAR + optical feature proxies and water_fraction (0-1).
    """
    json_file = data_dir / "flood_samples.json"
    if not json_file.exists():
        json_file = data_dir / "sen1floods11_split.json"

    if json_file.exists():
        try:
            data = json.loads(json_file.read_text())
            if isinstance(data, list):
                logger.info("[Flood] Loaded %d flood samples", len(data))
                return data
        except Exception:
            pass

    return _synthetic_samples()


def _synthetic_samples() -> list[dict]:
    random.seed(42)
    events = [
        ("hurricane_harvey_2017",  "coastal",    0.6),
        ("midwestern_flood_2019",  "riverine",   0.4),
        ("bangladesh_monsoon",     "riverine",   0.5),
        ("california_flash_flood", "flash",      0.3),
        ("kerala_flood_2018",      "riverine",   0.55),
        ("germany_flood_2021",     "riverine",   0.45),
        ("pakistan_flood_2022",    "riverine",   0.65),
        ("new_zealand_2023",       "coastal",    0.35),
    ]
    samples = []
    for event_id, flood_type, base_water_frac in events:
        for i in range(60):
            # Alternate between pre-flood (dry) and post-flood (wet) patches for balance
            if i % 2 == 0:
                water_frac = min(1.0, max(0.0, base_water_frac + random.gauss(0, 0.12)))
            else:
                water_frac = max(0.0, min(0.1, random.gauss(0.05, 0.04)))
            samples.append({
                "event_id": event_id,
                "sample_id": f"{event_id}_{i:03d}",
                "flood_type": flood_type,
                "water_fraction": round(water_frac, 3),
                # SAR features (Sentinel-1 VV/VH proxies)
                "sar_vv_mean": round(random.gauss(-12 - water_frac * 8, 3), 2),
                "sar_vh_mean": round(random.gauss(-18 - water_frac * 6, 3), 2),
                "sar_vv_std":  round(abs(random.gauss(2 - water_frac, 0.5)), 2),
                # Optical features
                "ndwi": round(min(1, max(-1, water_frac * 1.2 + random.gauss(0, 0.1))), 3),
                "ndvi": round(min(1, max(-1, (1 - water_frac) * 0.6 + random.gauss(0, 0.1))), 3),
                "blue_mean": round(0.1 + water_frac * 0.25 + random.gauss(0, 0.02), 3),
                # Terrain
                "slope_deg": round(abs(random.gauss(3 - water_frac * 2, 2)), 2),
                "dem_m": round(random.uniform(0, 200) * (1 - water_frac * 0.8), 1),
                # Label
                "is_flooded": 1 if water_frac > 0.15 else 0,
                "split": "train" if i < 48 else "val",
            })
    return samples


def _generate_synthetic(out_dir: Path) -> None:
    samples = _synthetic_samples()
    (out_dir / "flood_samples.json").write_text(json.dumps(samples))
    logger.info("[Flood] Generated %d synthetic flood samples", len(samples))
