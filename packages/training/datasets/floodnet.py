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

# Kaggle Hurricane Harvey flood segmentation dataset
HARVEY_DIR = Path(__file__).parent.parent / "data" / "flood_harvey" / "harvey_damage_satelite"

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


def _load_harvey_samples() -> list[dict]:
    """
    Load Hurricane Harvey flood segmentation dataset (real satellite imagery).
    Derives water_fraction and is_flooded from binary segmentation masks.
    White pixels (>128) in mask = flooded; black = dry.
    """
    mask_dir  = HARVEY_DIR / "train_masks"
    image_dir = HARVEY_DIR / "train_images"
    if not mask_dir.exists():
        return []

    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        logger.warning("[Flood] PIL/numpy not available — skipping Harvey real data")
        return []

    samples = []
    masks = list(mask_dir.glob("*.png"))
    logger.info("[Flood] Loading %d Harvey flood masks...", len(masks))

    for mask_path in masks:
        try:
            # Masks are class-indexed segmentation maps, not binary.
            # Class 0 = no-data/background, Class 23 = undamaged land.
            # Classes 1-22 represent flood damage / water / damaged structures.
            mask = np.array(Image.open(mask_path))
            flooded_pixels = np.isin(mask, [1, 2, 3, 4, 5])  # damaged buildings/roads + water
            water_frac = float(flooded_pixels.mean())

            # Derive tabular proxy features from image if available
            img_path = image_dir / (mask_path.stem + ".jpg")
            blue_mean = 0.12
            if img_path.exists():
                img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
                blue_mean = float(img[:, :, 2].mean())
                red_mean  = float(img[:, :, 0].mean())
                green_mean = float(img[:, :, 1].mean())
                ndwi_proxy = (green_mean - red_mean) / (green_mean + red_mean + 1e-6)
            else:
                ndwi_proxy = water_frac * 0.8 - 0.2

            samples.append({
                "event_id": "hurricane_harvey_2017",
                "sample_id": mask_path.stem,
                "flood_type": "coastal",
                "water_fraction": round(water_frac, 3),
                "sar_vv_mean": round(-12 - water_frac * 8, 2),   # proxy
                "sar_vh_mean": round(-18 - water_frac * 6, 2),
                "sar_vv_std":  round(max(0.5, 2 - water_frac), 2),
                "ndwi": round(min(1, max(-1, ndwi_proxy)), 3),
                "ndvi": round(min(1, max(-1, (1 - water_frac) * 0.5)), 3),
                "blue_mean": round(blue_mean, 3),
                "slope_deg": round(max(0, 3 - water_frac * 2), 2),
                "dem_m": round(max(0, 50 * (1 - water_frac)), 1),
                "is_flooded": 1 if water_frac > 0.10 else 0,
                "split": "train",
                "source": "kaggle_harvey_real",
            })
        except Exception:
            continue

    logger.info("[Flood] Loaded %d real Harvey flood samples", len(samples))
    return samples


def load_as_training_samples(data_dir: Path) -> list[dict]:
    """
    Returns samples with SAR + optical feature proxies and water_fraction (0-1).
    """
    # Blend real Harvey data with synthetic to ensure class balance.
    # Harvey post-event imagery is ~97% flooded — synthetic provides dry counterexamples.
    harvey_samples = _load_harvey_samples()
    if harvey_samples:
        synthetic = _synthetic_samples()
        blended = harvey_samples + synthetic
        flooded = sum(1 for s in blended if s["is_flooded"])
        logger.info("[Flood] Blended %d real Harvey + %d synthetic samples (%.0f%% flooded)",
                    len(harvey_samples), len(synthetic), 100 * flooded / len(blended))
        return blended

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
