"""
USGS Earthquake Catalog — Real Building Damage Proxy Dataset

USGS ComCat API: https://earthquake.usgs.gov/fdsnws/event/1/
License: US Government work, public domain.

Maps USGS alert levels to xBD-compatible damage classes:
  green  → no-damage (0)
  yellow → minor-damage (1)
  orange → major-damage (2)
  red    → destroyed (3)
  None   → estimated from magnitude + significance score

Features extracted per event:
  - disaster_type: earthquake / tsunami (from tsunami flag)
  - magnitude, mmi (Modified Mercalli Intensity), cdi (Community Intensity)
  - sig: USGS significance score (0-1000+)
  - tsunami: bool
  - depth_km: hypocentre depth

Used as real ground truth for train_damage.py alongside xBD synthetic labels.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent.parent / "data" / "usgs_earthquakes"

# USGS ComCat FDSN Event API — M4.5+ past 365 days, all regions
_API_URL = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query"
    "?format=geojson&minmagnitude=4.5&orderby=time&limit=2000"
    "&starttime=2024-04-01&endtime=2026-04-23"
)

_ALERT_TO_DAMAGE = {"green": 0, "yellow": 1, "orange": 2, "red": 3}


def _mag_to_damage(mag: float, sig: int, mmi: float | None, tsunami: bool) -> int:
    """Estimate damage class from magnitude + significance when no USGS alert available."""
    if tsunami:
        return max(2, min(3, int((mag - 6.0) + 2)))
    if mmi is not None:
        if mmi >= 8:  return 3
        if mmi >= 6:  return 2
        if mmi >= 4:  return 1
        return 0
    if mag >= 7.5: return 3
    if mag >= 6.5: return 2
    if mag >= 5.5: return 1
    return 0


def download(out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "usgs_m45_events.json"

    if out_file.exists():
        data = json.loads(out_file.read_text())
        logger.info("[USGS] Cached: %d earthquake events", len(data))
        return out_dir

    logger.info("[USGS] Downloading M4.5+ earthquake catalog from USGS ComCat...")
    try:
        resp = requests.get(_API_URL, timeout=30, headers={"User-Agent": "Heli.OS/1.0"})
        resp.raise_for_status()
        raw = resp.json()
        features = raw.get("features", [])
        logger.info("[USGS] Downloaded %d earthquake events", len(features))
        out_file.write_text(json.dumps(features))
    except Exception as e:
        logger.warning("[USGS] Download failed: %s", e)
        out_file.write_text("[]")

    return out_dir


def load_as_training_samples(data_dir: Path) -> list[dict]:
    out_file = data_dir / "usgs_m45_events.json"
    if not out_file.exists():
        return []

    try:
        features = json.loads(out_file.read_text())
    except Exception:
        return []

    samples = []
    for feat in features:
        props = feat.get("properties", {})
        geom  = feat.get("geometry", {})
        coords = geom.get("coordinates", [0, 0, 10])

        mag     = float(props.get("mag") or 4.5)
        alert   = (props.get("alert") or "").lower()
        mmi     = props.get("mmi")
        cdi     = props.get("cdi")
        sig     = int(props.get("sig") or 0)
        tsunami = bool(props.get("tsunami", 0))
        place   = props.get("place", "")
        depth   = float(coords[2]) if len(coords) > 2 else 10.0

        if alert in _ALERT_TO_DAMAGE:
            damage_class = _ALERT_TO_DAMAGE[alert]
        else:
            damage_class = _mag_to_damage(mag, sig, mmi, tsunami)

        disaster_type = "tsunami" if tsunami and mag >= 7.0 else "earthquake"

        # Derive sensor proxy features correlated with event magnitude
        sar_base = min((mag - 4.5) / 4.0, 1.0)  # 4.5→0, 8.5→1.0

        samples.append({
            "damage_class":          damage_class,
            "disaster_type":         disaster_type,
            "magnitude":             mag,
            "mmi":                   float(mmi) if mmi else None,
            "cdi":                   float(cdi) if cdi else None,
            "sig":                   sig,
            "tsunami":               int(tsunami),
            "depth_km":              depth,
            "place":                 place,
            # Sensor proxies (derived from magnitude/impact)
            "sar_incoherence":       min(sar_base + 0.1 * (damage_class / 3.0), 1.0),
            "thermal_anomaly":       1 if damage_class >= 2 and mag >= 6.5 else 0,
            "zone_area_m2":          min(1000 * (10 ** (mag - 4.0)), 50000),
            "structure_density":     500.0,
            "time_since_event_h":    12.0,
            "pre_event_pop_density": min(sig * 2.0, 10000),
            "source":                "usgs_real",
        })

    logger.info("[USGS] Parsed %d training samples (real earthquake events)", len(samples))
    return samples
