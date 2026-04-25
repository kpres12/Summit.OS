"""
NASA EONET Natural Hazard Classifier

Trains a multi-class classifier on NASA EONET (Earth Observatory Natural
Event Tracker) — a curated catalog of currently-occurring and recent
natural events worldwide.

Categories include:
  Wildfires, Severe Storms, Volcanoes, Drought, Floods, Earthquakes,
  Sea & Lake Ice, Snow, Manmade events, Dust & Haze, Landslides,
  Water Color, Temperature Extremes.

Per-event features:
  geo: lat, lon, lat_abs, lon_sin/cos
  time: day_of_year (sin/cos), hour (sin/cos)
  count: event has multiple geometry points (= active over time)
  text: title token-bag (event-keyword presence flags)
  source: number of EONET sources (cross-confirmation)

Target: EONET category (id of the category most associated with the event)

Data: https://eonet.gsfc.nasa.gov/api/v3/events  (public, no auth)

Output: packages/c2_intel/models/eonet_hazard_classifier.joblib
        packages/c2_intel/models/eonet_hazard_classifier_meta.json

Usage:
    python train_eonet_classifier.py [--limit 5000 --days 365]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "c2_intel" / "models"
MODEL_PATH = MODELS_DIR / "eonet_hazard_classifier.joblib"
META_PATH  = MODELS_DIR / "eonet_hazard_classifier_meta.json"
CACHE_DIR  = Path(__file__).parent / "data" / "eonet"

EONET_URL = "https://eonet.gsfc.nasa.gov/api/v3/events"

# Bag-of-words feature flags — keep small, tuned to common EONET vocabulary
KEYWORDS = [
    "fire", "wildfire", "flood", "flooding", "rain",
    "storm", "cyclone", "hurricane", "typhoon", "tornado",
    "volcano", "volcanic", "eruption", "ash",
    "earthquake", "tsunami",
    "drought", "heat", "cold",
    "ice", "iceberg", "snow", "blizzard",
    "dust", "haze", "smoke",
    "landslide", "mudslide",
    "oil", "spill", "industrial",
    "algal", "bloom",
]


def _download(days: int = 365, limit: int = 5000, status: str = "all") -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"eonet_d{days}_l{limit}_{status}.json"
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
            if data:
                logger.info("[EONET] Loaded %d cached events", len(data))
                return data
        except Exception:
            pass

    logger.info("[EONET] Downloading: days=%d limit=%d status=%s", days, limit, status)
    try:
        r = requests.get(EONET_URL, params={
            "days": days, "limit": limit, "status": status,
        }, timeout=60, headers={"User-Agent": "Heli.OS/1.0"})
        r.raise_for_status()
        events = r.json().get("events", [])
        cache.write_text(json.dumps(events))
        logger.info("[EONET] Cached %d events", len(events))
        return events
    except Exception as e:
        logger.error("[EONET] Download failed: %s", e)
        return []


def _featurize(events: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    samples = []
    cat_to_id: dict[str, int] = {}
    id_to_cat: list[str] = []

    feat_names = [
        "lat", "lon", "lat_abs", "lon_sin", "lon_cos",
        "day_sin", "day_cos", "n_geometry_points",
        "n_sources", "title_len",
    ] + [f"kw::{k}" for k in KEYWORDS]

    for ev in events:
        geom = ev.get("geometry") or []
        if not geom:
            continue
        # Use first geometry point for spatial features
        first = geom[0]
        coords = first.get("coordinates") or [0.0, 0.0]
        if not isinstance(coords, list) or len(coords) < 2:
            continue
        try:
            lon = float(coords[0])
            lat = float(coords[1])
        except (TypeError, ValueError):
            continue

        cats = ev.get("categories") or []
        if not cats:
            continue
        cat = cats[0].get("title") or cats[0].get("id")
        if cat is None:
            continue
        cat = str(cat).strip()

        if cat not in cat_to_id:
            cat_to_id[cat] = len(id_to_cat)
            id_to_cat.append(cat)

        # Time features from the first geometry's date
        date_str = first.get("date") or ""
        try:
            dt = datetime.strptime(date_str.replace("Z", "+00:00"),
                                   "%Y-%m-%dT%H:%M:%S%z")
        except (ValueError, TypeError):
            try:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except Exception:
                dt = datetime.now(timezone.utc)
        doy = dt.timetuple().tm_yday
        day_sin = math.sin(2 * math.pi * doy / 365.25)
        day_cos = math.cos(2 * math.pi * doy / 365.25)

        sources = ev.get("sources") or []
        n_sources = len(sources)
        title = (ev.get("title") or "").lower()
        title_clean = re.sub(r"[^a-z0-9 ]+", " ", title)
        words = set(title_clean.split())
        kw_flags = [1.0 if k in words or any(k in w for w in words) else 0.0
                    for k in KEYWORDS]

        # Use the description too — it has more signal
        desc = (ev.get("description") or "").lower()
        if desc:
            for i, k in enumerate(KEYWORDS):
                if kw_flags[i] == 0 and k in desc:
                    kw_flags[i] = 0.5  # half-weight if only in description

        feat = np.array([
            lat,
            lon / 180.0,
            abs(lat) / 90.0,
            math.sin(math.radians(lon)),
            math.cos(math.radians(lon)),
            day_sin, day_cos,
            min(len(geom) / 50.0, 1.0),
            min(n_sources / 5.0, 1.0),
            min(len(title) / 100.0, 1.0),
        ] + kw_flags, dtype=np.float32)

        samples.append((feat, cat_to_id[cat]))

    if not samples:
        return np.zeros((0, len(feat_names)), dtype=np.float32), np.zeros(0, dtype=np.int64), feat_names, id_to_cat

    X = np.stack([s[0] for s in samples])
    y = np.array([s[1] for s in samples], dtype=np.int64)
    return X, y, feat_names, id_to_cat


def train(days: int = 365, limit: int = 5000) -> None:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    import joblib

    events = _download(days=days, limit=limit, status="all")
    if not events:
        # Try larger window
        events = _download(days=days * 3, limit=limit, status="all")
    if not events:
        raise RuntimeError("EONET returned no events.")

    X, y, feat_names, id_to_cat = _featurize(events)
    if len(X) < 50:
        raise RuntimeError(f"Only {len(X)} usable EONET samples — too few.")

    # Drop ultra-rare classes (<5 events) to avoid degenerate folds
    counts = Counter(y.tolist())
    keep = {c for c, n in counts.items() if n >= 5}
    if len(keep) < len(counts):
        keep_mask = np.array([yi in keep for yi in y])
        X, y = X[keep_mask], y[keep_mask]
        # Reindex labels
        old_to_new: dict[int, int] = {}
        new_id_to_cat: list[str] = []
        for c in sorted(keep):
            old_to_new[c] = len(new_id_to_cat)
            new_id_to_cat.append(id_to_cat[c])
        y = np.array([old_to_new[yi] for yi in y], dtype=np.int64)
        id_to_cat = new_id_to_cat
        logger.info("[EONET] Dropped rare classes — kept %d classes / %d samples",
                    len(keep), len(X))

    logger.info("[EONET] Class distribution: %s",
                {id_to_cat[c]: int(n) for c, n in Counter(y.tolist()).items()})

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.08,
        subsample=0.8, random_state=7)

    f1 = cross_val_score(clf, X, y, cv=3, scoring="f1_weighted", n_jobs=-1)
    logger.info("[EONET] CV f1_weighted = %.4f +/- %.4f", f1.mean(), f1.std())

    clf.fit(X, y)

    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    top = {feat_names[i]: round(float(importances[i]), 4) for i in top_idx}
    logger.info("[EONET] Top 10 features: %s", top)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": clf, "feature_names": feat_names,
                 "categories": id_to_cat}, MODEL_PATH)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": "GradientBoostingClassifier",
        "task": "EONET natural hazard category",
        "categories": id_to_cat,
        "n_classes": len(id_to_cat),
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "data_source": "nasa_eonet_real",
        "days_window": days,
        "metrics": {
            "f1_cv_weighted": round(float(f1.mean()), 4),
            "f1_cv_weighted_std": round(float(f1.std()), 4),
        },
        "top_features": top,
        "class_distribution": {id_to_cat[c]: int(n) for c, n in Counter(y.tolist()).items()},
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("[EONET] Saved -> %s", MODEL_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--limit", type=int, default=5000)
    args = p.parse_args()
    train(days=args.days, limit=args.limit)
