"""
Building Damage Assessment Model Trainer

Trains a gradient-boosted classifier to estimate building damage severity
from xBD annotations. This is a lightweight tabular model (not a vision model)
that runs on metadata features extracted from imagery or sensor data.

Output: packages/c2_intel/models/damage_classifier.joblib
        packages/c2_intel/models/damage_classifier_meta.json

Input features (per building/zone):
  - disaster_type (one-hot, 8 types)
  - zone_area_m2: estimated affected area
  - structure_density: buildings per km² (from world model)
  - thermal_anomaly: bool — any thermal hot-spots detected
  - sar_incoherence: float 0-1 (SAR change detection score; 0=no change)
  - time_since_event_h: hours since disaster event
  - pre_event_pop_density: population density (from public data)

Target: damage_class 0-3 (no-damage → destroyed)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "c2_intel" / "models"
META_PATH = MODELS_DIR / "damage_classifier_meta.json"
MODEL_PATH = MODELS_DIR / "damage_classifier.joblib"

DISASTER_TYPES = [
    "hurricane", "wildfire", "flooding", "earthquake",
    "tsunami", "tornado", "volcano", "other",
]


def _load_xbd_samples() -> tuple[np.ndarray, np.ndarray]:
    """Load xBD + synthetic samples and extract tabular features."""
    from datasets.xbd import download, load_as_training_samples

    labels_dir = download(skip_images=True)
    samples = load_as_training_samples(labels_dir)

    if not samples:
        logger.warning("[DamageModel] No xBD samples — using synthetic")
        samples = _synthetic_samples(n=2000)

    X, y = [], []
    for s in samples:
        features = _extract_features(s)
        X.append(features)
        y.append(int(s.get("damage_class", 0)))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def _extract_features(sample: dict) -> list[float]:
    # Disaster type one-hot (8 features)
    dtype = sample.get("disaster_type", "other").lower()
    dtype_oh = [1.0 if dt in dtype else 0.0 for dt in DISASTER_TYPES]

    # Continuous features
    zone_area = float(sample.get("zone_area_m2", 5000))
    struct_density = float(sample.get("structure_density", 500))
    thermal = float(sample.get("thermal_anomaly", 0))
    sar_incoh = float(sample.get("sar_incoherence", 0.5))
    time_h = float(sample.get("time_since_event_h", 12))
    pop_density = float(sample.get("pre_event_pop_density", 2000))

    # Normalize
    zone_area = min(zone_area / 100_000, 1.0)
    struct_density = min(struct_density / 10_000, 1.0)
    time_h = min(time_h / 72.0, 1.0)
    pop_density = min(pop_density / 20_000, 1.0)

    return dtype_oh + [zone_area, struct_density, thermal, sar_incoh, time_h, pop_density]


def _synthetic_samples(n: int = 2000) -> list[dict]:
    import random
    random.seed(42)
    samples = []
    disaster_types = DISASTER_TYPES
    for _ in range(n):
        dtype = random.choice(disaster_types)
        # Simulate realistic damage distributions by disaster type
        if dtype in ("earthquake", "tsunami"):
            damage = random.choices([0, 1, 2, 3], weights=[0.2, 0.25, 0.3, 0.25])[0]
            sar = random.uniform(0.5, 1.0)
        elif dtype == "wildfire":
            damage = random.choices([0, 1, 2, 3], weights=[0.15, 0.2, 0.35, 0.3])[0]
            sar = random.uniform(0.4, 0.9)
        elif dtype in ("hurricane", "tornado"):
            damage = random.choices([0, 1, 2, 3], weights=[0.25, 0.3, 0.3, 0.15])[0]
            sar = random.uniform(0.3, 0.8)
        else:
            damage = random.choices([0, 1, 2, 3], weights=[0.4, 0.3, 0.2, 0.1])[0]
            sar = random.uniform(0.1, 0.6)

        samples.append({
            "disaster_type": dtype,
            "damage_class": damage,
            "zone_area_m2": random.uniform(100, 50000),
            "structure_density": random.uniform(50, 5000),
            "thermal_anomaly": 1 if (dtype == "wildfire" and damage >= 2) else random.choice([0, 1]),
            "sar_incoherence": sar,
            "time_since_event_h": random.uniform(0.5, 72),
            "pre_event_pop_density": random.uniform(100, 10000),
        })
    return samples


def train(n_estimators: int = 200) -> None:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    import joblib

    logger.info("[DamageModel] Loading training data...")
    X, y = _load_xbd_samples()
    logger.info("[DamageModel] %d samples, %d features, %d classes", len(X), X.shape[1], len(set(y)))

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )

    scores = cross_val_score(model, X, y, cv=5, scoring="f1_macro")
    logger.info("[DamageModel] CV F1-macro: %.3f ± %.3f", scores.mean(), scores.std())

    model.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": "GradientBoostingClassifier",
        "n_estimators": n_estimators,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "disaster_types": DISASTER_TYPES,
        "classes": ["no-damage", "minor-damage", "major-damage", "destroyed"],
        "metrics": {
            "f1_macro_cv": round(float(scores.mean()), 4),
            "f1_macro_cv_std": round(float(scores.std()), 4),
        },
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("[DamageModel] Model saved → %s", MODEL_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).parent))
    train()
